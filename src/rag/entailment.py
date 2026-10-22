"""Đối chiếu premise–hypothesis bằng NLI đa ngôn ngữ, bổ sung cho nhận xét của LLM."""
from src.core.model_registry import checked_model


def aggregate_predictions(predictions, threshold=0.8):
    if not 0.5 < threshold <= 1:
        raise ValueError("Ngưỡng NLI phải thuộc (0.5, 1]")
    # Không dùng một cặp đã cắt ngắn để tự kết luận hỗ trợ hoặc mâu thuẫn.
    support = list(dict.fromkeys(item["evidence_id"] for item in predictions if not item["truncated"] and item["scores"]["entailment"] >= threshold))
    contradiction = list(dict.fromkeys(item["evidence_id"] for item in predictions if not item["truncated"] and item["scores"]["contradiction"] >= threshold))
    conflict = bool(support and contradiction)
    status = "insufficient" if conflict else ("contradicted" if contradiction else ("supported" if support else "insufficient"))
    return {"status": status, "support_ids": support, "contradiction_ids": contradiction,
            "conflicting_evidence": conflict, "threshold": threshold, "predictions": predictions}


class LocalEntailment:
    def __init__(self, threshold=0.8, threads=6):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        if not 0.5 < threshold <= 1:
            raise ValueError("Ngưỡng NLI phải thuộc (0.5, 1]")
        torch.set_num_threads(threads)
        directory, self.identity = checked_model("entailment")
        self.tokenizer = AutoTokenizer.from_pretrained(directory, use_fast=False,
            local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(directory, torch_dtype=torch.float32,
            local_files_only=True, trust_remote_code=False).eval()
        self.labels = [self.model.config.id2label[i] for i in range(3)]
        if set(self.labels) != {"entailment", "neutral", "contradiction"}:
            raise ValueError("Model NLI có nhãn không đúng cấu hình")
        self.threshold = threshold

    def assess(self, claim, evidence):
        import torch
        hypothesis = self.tokenizer(claim, add_special_tokens=False, verbose=False)["input_ids"]
        if len(hypothesis) > 128:
            raise ValueError("Nhận định quá dài cho NLI; cần tách nhỏ hơn")
        budget = min(384, 512 - len(hypothesis) - self.tokenizer.num_special_tokens_to_add(pair=True))
        predictions = []
        for item in evidence:
            premise = self.tokenizer(item.content, add_special_tokens=False, verbose=False)["input_ids"]
            for start in range(0, max(1, len(premise)), budget - 48):
                end = min(start + budget, len(premise))
                inputs = self.tokenizer.prepare_for_model(premise[start:end], pair_ids=hypothesis,
                    add_special_tokens=True, return_attention_mask=True, return_tensors="pt")
                inputs = {key: value.unsqueeze(0) if value.ndim == 1 else value for key, value in inputs.items()}
                if inputs["input_ids"].shape[-1] > 512:
                    raise ValueError("Cửa sổ NLI vượt giới hạn model")
                with torch.inference_mode():
                    values = torch.softmax(self.model(**inputs).logits[0].float(), dim=-1).tolist()
                predictions.append({"evidence_id": item.evidence_id, "truncated": False,
                    "premise_token_start": start, "premise_token_end": end,
                    "scores": dict(zip(self.labels, values)), "label": self.labels[max(range(3), key=lambda i: values[i])]})
                if end == len(premise):
                    break
        return aggregate_predictions(predictions, self.threshold)
