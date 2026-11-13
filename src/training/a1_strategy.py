"""Lấy mẫu có phủ toàn bộ train và tăng trọng số token điều phối A1."""
from collections import Counter, defaultdict
import math
import random

from src.training.a1_evaluation import DECISIONS, selection_score


def plan_epoch(records, seed, strategy=None):
    if not records or any(r["split"]!="train" for r in records):
        raise ValueError("Sampler chỉ nhận mẫu train không rỗng")
    rng=random.Random(seed)
    if strategy is None:
        order=list(range(len(records)))
        rng.shuffle(order)
    else:
        total=strategy["samples_per_epoch"]
        if not isinstance(total,int) or total<len(records):
            raise ValueError("samples_per_epoch phải nguyên và đủ phủ mọi mẫu train")
        distribution=strategy["decision_distribution"]
        if (set(distribution)!=set(DECISIONS) or any(not math.isfinite(v) or v<=0 for v in distribution.values())
                or not math.isclose(sum(distribution.values()),1.0)):
            raise ValueError("Phân bố sampling cần đủ năm decision và tổng bằng 1")
        raw={k:total*v for k,v in distribution.items()}
        quotas={k:math.floor(v) for k,v in raw.items()}
        for key in sorted(raw,key=lambda k:raw[k]-quotas[k],reverse=True)[:total-sum(quotas.values())]:
            quotas[key]+=1
        counts=Counter(r["target"]["decision"] for r in records)
        if any(quotas[k]<counts[k] for k in DECISIONS):
            raise ValueError("Quota không đủ phủ mọi mẫu; tăng samples_per_epoch hoặc sửa phân bố")
        if any(counts[k]==0 for k in DECISIONS):
            raise ValueError("Thiếu decision trong train; không thể replay")
        # Mọi mẫu có một lượt gốc. Chỉ replay phần quota còn thiếu.
        order=list(range(len(records)))
        buckets=defaultdict(lambda:defaultdict(list))
        for index,row in enumerate(records):
            decision=row["target"]["decision"]
            kind=tuple(row["target"]["targets"]) if decision=="route" else row["category"]
            buckets[(decision,kind)][row["group_id"]].append(index)
        for decision in DECISIONS:
            keys=[key for key in buckets if key[0]==decision]
            rng.shuffle(keys)
            group_cycles={key:list(buckets[key]) for key in keys}
            for names in group_cycles.values():
                rng.shuffle(names)
            for n in range(quotas[decision]-counts[decision]):
                key=keys[n%len(keys)]
                cycle=group_cycles[key]
                group=cycle[(n//len(keys))%len(cycle)]
                order.append(rng.choice(buckets[key][group]))
        rng.shuffle(order)
    repeats=Counter(order)
    return order,{"samples":len(order),"unique_train_samples":len(repeats),"available_train_samples":len(records),
                  "max_repeats":max(repeats.values()),
                  "decision_counts":dict(Counter(records[i]["target"]["decision"] for i in order)),
                  "seed":seed,"sample_ids":[records[i]["id"] for i in order]}


def assistant_weights(tokenizer, messages, ids, prompt_length, config=None):
    length=len(ids)-prompt_length
    weights=[1.0]*length
    if config is None:
        return weights
    prefix_weight,decision_weight=config["control_prefix_weight"],config["decision_value_weight"]
    if not all(math.isfinite(v) and v>=1 for v in (prefix_weight,decision_weight)):
        raise ValueError("Trọng số token cần hữu hạn và >= 1")
    raw=messages[-1]["content"]
    if not raw.startswith('{"decision":"'):
        raise ValueError("Cần JSON assistant canonical, không suy đoán vị trí token")
    prefix_end=raw.index(',"resolved_request":')
    start=len('{"decision":"')
    end=raw.index('"',start)
    encoding=tokenizer(raw,add_special_tokens=False,return_offsets_mapping=True)
    raw_ids=encoding["input_ids"]
    if ids[prompt_length:prompt_length+len(raw_ids)]!=raw_ids:
        # SentencePiece may tokenize the same JSON differently after the assistant header.
        # Align against offsets of the entire exact rendered chat, never approximate token indices.
        if not hasattr(tokenizer, "apply_chat_template"):
            raise ValueError("Token assistant không khớp template; không được gán trọng số xấp xỉ")
        rendered=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=False,enable_thinking=False)
        start_raw=rendered.rfind(raw)
        complete=tokenizer(rendered,add_special_tokens=False,return_offsets_mapping=True)
        if start_raw<0 or complete["input_ids"]!=ids:
            raise ValueError("Token assistant không khớp template; không được gán trọng số xấp xỉ")
        for i,(left,right) in enumerate(complete["offset_mapping"][prompt_length:]):
            if right>start_raw and left<start_raw+prefix_end:
                weights[i]=prefix_weight
            if left<start_raw+end and right>start_raw+start:
                weights[i]=max(weights[i],decision_weight)
        return weights
    for i,(left,right) in enumerate(encoding["offset_mapping"]):
        if right>left and left<prefix_end:
            weights[i]=prefix_weight
        if left<end and right>start:
            weights[i]=max(weights[i],decision_weight)
    return weights


def training_score(metrics, profile="control_exact"):
    if profile=="control_exact":
        return selection_score(metrics)
    if profile!="critical_recall":
        raise ValueError("Profile lựa chọn checkpoint không hợp lệ")
    clarify=metrics["per_decision"]["clarify"]["recall"]
    handoff=metrics["per_decision"]["handoff"]["recall"]
    eligible=(clarify>=.5 and handoff>=.75 and metrics["rates"]["schema_valid"]>=.98
              and metrics["rates"]["context_valid"]>=.98)
    return (int(eligible),min(clarify,handoff),metrics["decision_macro_f1"],*selection_score(metrics))
