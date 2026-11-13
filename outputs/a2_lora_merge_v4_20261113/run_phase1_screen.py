"""Phase 1: Build exact weighted LoRA merges (epoch_0.5 + epoch_1.0) & run fast screening on RAG probe + conflict slice."""
from __future__ import annotations

import os

os.environ.update(USE_TF="0", USE_FLAX="0", HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false")

import argparse
import gc
import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

import torch
from peft import PeftModel
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.a2_sentence_io import parse_sample, score_answer, summarize_results
from scripts.train_a2 import configure_tokenizer
from scripts.train_a2_v4_kaggle import _breakdown_by_slice, _load_split, checkpoint_composite
from src.training.a2_sentence_protocol import make_messages, normalize_payload_evidence, resolve_selection

OUT_DIR = ROOT / "outputs/a2_lora_merge_v4_20261113"
SRC_05 = OUT_DIR / "sources/epoch_0.5"
SRC_10 = OUT_DIR / "sources/epoch_1.0"
BASE_MODEL = ROOT / "data/models/Qwen--Qwen3-1.7B/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
V4_DATA = ROOT / "data/processed/a2_sentence_selection_v4"
PROBE_DIR = ROOT / "outputs/a2_probe_v4_selected_20261113"
ZIP_RESULTS = ROOT / "outputs/a2_kaggle_sentence_v4_fp16/results/a2_sentence_v4_fp16_results.zip"


def ensure_sources() -> None:
    if (SRC_05 / "adapter_model.safetensors").exists() and (SRC_10 / "adapter_model.safetensors").exists():
        return
    with zipfile.ZipFile(ZIP_RESULTS) as zf:
        for ck, dst in [("epoch_0.5", SRC_05), ("epoch_1.0", SRC_10)]:
            dst.mkdir(parents=True, exist_ok=True)
            prefix = f"checkpoint_adapters/{ck}/"
            for name in zf.namelist():
                if name.startswith(prefix) and not name.endswith("/"):
                    (dst / name[len(prefix):]).write_bytes(zf.read(name))


def build_exact_cat_merge(w05: float, tag: str) -> Path:
    """Create an exact weight-space LoRA merge: ΔW = w05 * ΔW_0.5 + (1 - w05) * ΔW_1.0 via rank-32 cat."""
    w10 = 1.0 - w05
    dst = OUT_DIR / "adapters" / tag
    dst.mkdir(parents=True, exist_ok=True)

    cfg05 = json.loads((SRC_05 / "adapter_config.json").read_text(encoding="utf-8"))
    cfg10 = json.loads((SRC_10 / "adapter_config.json").read_text(encoding="utf-8"))
    assert cfg05["r"] == cfg10["r"] == 16 and cfg05["lora_alpha"] == cfg10["lora_alpha"] == 32
    scale_src = cfg05["lora_alpha"] / cfg05["r"]  # 2.0

    s05 = load_file(str(SRC_05 / "adapter_model.safetensors"))
    s10 = load_file(str(SRC_10 / "adapter_model.safetensors"))
    merged = {}
    max_err = 0.0

    for k in s05:
        if k.endswith(".lora_A.weight"):
            kb = k.replace(".lora_A.weight", ".lora_B.weight")
            a05, b05 = s05[k], s05[kb]
            a10, b10 = s10[k], s10[kb]
            # Target merged config uses r=32, lora_alpha=32 => scaling = 1.0
            # Note keeping same dtype as sources (float32 / float16)
            dtype = a05.dtype
            a_cat = torch.cat([w05 * scale_src * a05.float(), w10 * scale_src * a10.float()], dim=0).to(dtype)
            b_cat = torch.cat([b05.float(), b10.float()], dim=1).to(dtype)
            merged[k] = a_cat
            merged[kb] = b_cat

            # Sanity check exact ΔW equivalence
            dw_expected = w05 * scale_src * (b05.float() @ a05.float()) + w10 * scale_src * (b10.float() @ a10.float())
            dw_actual = 1.0 * (b_cat.float() @ a_cat.float())
            err = (dw_expected - dw_actual).abs().max().item()
            if err > max_err:
                max_err = err
        elif not k.endswith(".lora_B.weight"):
            raise ValueError(f"Unexpected tensor key in adapter: {k}")

    save_file(merged, str(dst / "adapter_model.safetensors"))
    new_cfg = dict(cfg10)
    new_cfg["r"] = cfg05["r"] + cfg10["r"]
    new_cfg["lora_alpha"] = new_cfg["r"]  # scaling = 1.0
    (dst / "adapter_config.json").write_text(json.dumps(new_cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    # Copy tokenizer & README files cleanly
    for fname in ["README.md", "added_tokens.json", "chat_template.jinja", "merges.txt", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.json"]:
        src_f = SRC_10 / fname
        if src_f.exists():
            (dst / fname).write_bytes(src_f.read_bytes())

    sha = hashlib.sha256((dst / "adapter_model.safetensors").read_bytes()).hexdigest()
    print(f"[{tag}] w05={w05:.2f}, w10={w10:.2f} -> max |ΔW error| = {max_err:.2e}, sha256={sha[:16]}")
    return dst


def load_conflict_indices() -> dict[str, list[int]]:
    """Select exact cases where epoch_0.5 & epoch_1.0 disagree or fail (NEVER touching test.jsonl)."""
    import zipfile

    out: dict[str, list[int]] = {}
    with zipfile.ZipFile(ZIP_RESULTS) as zf:
        for sp in ("val", "challenge"):
            p05 = [json.loads(x) for x in zf.read(f"checkpoint_evals/epoch_0.5/{sp}/predictions.jsonl").decode().splitlines() if x.strip()]
            p10 = [json.loads(x) for x in zf.read(f"checkpoint_evals/epoch_1.0/{sp}/predictions.jsonl").decode().splitlines() if x.strip()]
            idxs = []
            for i, (a, b) in enumerate(zip(p05, p10)):
                a_ok = a["status_correct"] and (a["expected_status"] != "draft" or a["selection_set_exact"])
                b_ok = b["status_correct"] and (b["expected_status"] != "draft" or b["selection_set_exact"])
                if (a["predicted_status"] != b["predicted_status"]) or (a_ok != b_ok) or (not b_ok):
                    idxs.append(i)
            out[sp] = idxs
    return out


def generate_one(model, tok, messages: list[dict]) -> tuple[str, float, int]:
    ids = tok.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
    ).to("cuda")
    in_tok = int(ids.shape[1])
    t0 = time.perf_counter()
    with torch.inference_mode():
        gen = model.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            max_new_tokens=384,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )
    sec = time.perf_counter() - t0
    raw = tok.decode(gen[0, in_tok:], skip_special_tokens=True).strip()
    del ids, gen
    torch.cuda.empty_cache()
    return raw, sec, in_tok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=float, default=[0.50])
    parser.add_argument("--mode", choices=["probe", "screen", "full"], default="screen")
    parser.add_argument("--skip-probe", action="store_true")
    args = parser.parse_args()

    ensure_sources()
    torch.set_num_threads(6)
    configs = [(f"merge_w05_{w:.2f}", w) for w in args.weights]
    adapter_dirs = {tag: build_exact_cat_merge(w, tag) for tag, w in configs}

    conflicts = load_conflict_indices()
    print(f"Conflict/error slice sizes -> val: {len(conflicts['val'])}/84, challenge: {len(conflicts['challenge'])}/80")

    val_records = _load_split(V4_DATA, "val")
    chal_records = _load_split(V4_DATA, "challenge")
    probe_cases = json.loads((PROBE_DIR / "cases.json").read_text(encoding="utf-8"))

    tok = configure_tokenizer(AutoTokenizer.from_pretrained(str(BASE_MODEL), local_files_only=True), "qwen3_17b")

    summary_out = {}
    for tag, adir in adapter_dirs.items():
        print(f"\n=== RUNNING {tag} (mode={args.mode}) ===", flush=True)
        base = AutoModelForCausalLM.from_pretrained(
            str(BASE_MODEL),
            local_files_only=True,
            trust_remote_code=False,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            ),
            device_map={"": 0},
            dtype=torch.float16,
            attn_implementation="sdpa",
        )
        model = PeftModel.from_pretrained(base, str(adir)).eval()
        model.requires_grad_(False)
        model.config.use_cache = True

        probe_out = []
        if not args.skip_probe:
            probe_path = OUT_DIR / f"probe_{tag}.json"
            if probe_path.exists():
                probe_out = json.loads(probe_path.read_text(encoding="utf-8"))
                print(f"  Loaded cached probe ({len(probe_out)} cases)", flush=True)
            else:
                for idx, c in enumerate(probe_cases):
                    payload = normalize_payload_evidence({"question": c["question"], "evidence": [{k: e[k] for k in ["evidence_id", "title", "content"]} for e in c["evidence"]]})
                    for e in payload["evidence"]:
                        e["content"] = " ".join(e["content"].split())
                    msgs = make_messages(payload)
                    raw, sec, in_tok = generate_one(model, tok, msgs)
                    resolved = None
                    res_err = None
                    try:
                        resolved = resolve_selection(json.loads(raw), payload).model_dump()
                    except Exception as exc:
                        res_err = str(exc)
                    probe_out.append({
                        "idx": idx,
                        "question": c["question"],
                        "in_tok": in_tok,
                        "seconds": round(sec, 2),
                        "raw_output": raw,
                        "resolved_status": resolved["status"] if resolved else None,
                        "resolved_output": resolved,
                        "res_err": res_err,
                    })
                    print(f"  probe[{idx}] ({sec:.1f}s, {in_tok}t) -> status={resolved['status'] if resolved else 'ERR'} | {raw[:90]}", flush=True)
                probe_path.write_text(json.dumps(probe_out, ensure_ascii=False, indent=2), encoding="utf-8")

        split_results = {}
        if args.mode == "screen":
            for sp, recs in [("val", val_records), ("challenge", chal_records)]:
                rows = []
                for k_i, idx in enumerate(conflicts[sp], 1):
                    rec = recs[idx]
                    raw, sec, in_tok = generate_one(model, tok, rec["messages"][:2])
                    payload, gold = parse_sample(rec)
                    sc = score_answer(rec, raw)
                    rows.append({"idx": idx, "seconds": round(sec, 2), "in_tok": in_tok, "raw_output": raw, "expected": gold, **sc})
                    print(f"  [{tag}/{sp}-screen {k_i}/{len(conflicts[sp])}] idx={idx} ({sec:.1f}s, {in_tok}t) st={sc['predicted_status']}/{gold['status']} sel={sc['selection_set_exact']}", flush=True)
                st_ok = sum(r["status_correct"] for r in rows)
                ans = [r for r in rows if r["expected_status"] == "draft"]
                sel_ok = sum(r["selection_set_exact"] for r in ans)
                avg_s = sum(r["seconds"] for r in rows) / max(1, len(rows))
                split_results[sp] = {
                    "n": len(rows),
                    "status_ok": st_ok,
                    "ans_n": len(ans),
                    "sel_exact": sel_ok,
                    "avg_seconds": round(avg_s, 2),
                    "rows": rows,
                }
                print(f"  {sp} conflict slice ({len(rows)}): status={st_ok}/{len(rows)}, ans_sel={sel_ok}/{len(ans)}, avg={avg_s:.1f}s", flush=True)
            (OUT_DIR / f"screen_{tag}.json").write_text(json.dumps({"probe": probe_out, "slices": split_results}, ensure_ascii=False, indent=2), encoding="utf-8")
            summary_out[tag] = {"probe": probe_out, "slices": split_results}
        elif args.mode == "full":
            full_dir = OUT_DIR / f"full_{tag}"
            full_dir.mkdir(parents=True, exist_ok=True)
            metrics_by_split = {}
            for sp, recs in [("val", val_records), ("challenge", chal_records)]:
                sp_dir = full_dir / sp
                sp_dir.mkdir(parents=True, exist_ok=True)
                pred_file = sp_dir / "predictions.jsonl"
                existing = []
                if pred_file.exists():
                    existing = [json.loads(x) for x in pred_file.read_text(encoding="utf-8").splitlines() if x.strip()]
                rows = list(existing)
                with pred_file.open("a", encoding="utf-8") as f_out:
                    for idx_1, rec in enumerate(recs, 1):
                        if idx_1 <= len(existing):
                            continue
                        raw, sec, in_tok = generate_one(model, tok, rec["messages"][:2])
                        payload, gold = parse_sample(rec)
                        row = {
                            "sample_id": rec.get("metadata", {}).get("id"),
                            "kind": rec.get("metadata", {}).get("kind"),
                            "partial_direction": rec.get("metadata", {}).get("partial_direction"),
                            "evidence_count": len(payload["evidence"]),
                            "question": payload["question"],
                            "raw_output": raw,
                            "expected": gold,
                            "generation_limit": 384,
                            "seconds": round(sec, 2),
                            "in_tok": in_tok,
                            **score_answer(rec, raw),
                        }
                        rows.append(row)
                        f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                        f_out.flush()
                        if idx_1 == 1 or idx_1 % 10 == 0 or idx_1 == len(recs):
                            print(f"  [{tag}/{sp}] {idx_1}/{len(recs)} ({sec:.1f}s, {in_tok}t)", flush=True)
                summary = summarize_results(rows)
                summary["slice_breakdown"] = _breakdown_by_slice(rows, recs)
                (sp_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
                metrics_by_split[sp] = summary
            comp = checkpoint_composite(metrics_by_split["val"], metrics_by_split["challenge"])
            full_report = {"tag": tag, "composite_score": comp, **metrics_by_split}
            (full_dir / "full_summary.json").write_text(json.dumps(full_report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  [{tag}] FULL DONE -> composite={comp:.4f} | val_st={metrics_by_split['val']['status_correct']:.4f} val_sel={metrics_by_split['val']['selection_set_exact']:.4f} | chal_st={metrics_by_split['challenge']['status_correct']:.4f} chal_sel={metrics_by_split['challenge']['selection_set_exact']:.4f}", flush=True)
            summary_out[tag] = full_report

        del model, base
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
