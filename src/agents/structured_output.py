"""Kiểm tra JSON của LLM, thử sửa định dạng một lần rồi báo lỗi rõ ràng."""
import hashlib
import json

from langchain_core.messages import HumanMessage, SystemMessage


class ModelOutputError(ValueError):
    pass


def parse_json(text):
    text = text.strip()
    if text.startswith("```json\n") and text.endswith("```"):
        text = text[8:-3].strip()
    elif text.startswith("```\n") and text.endswith("```"):
        text = text[4:-3].strip()
    return json.loads(text)


def invoke_structured(llm, schema, system, user, trace, attempts=2, validate=None):
    schema_text = json.dumps(schema.model_json_schema(), ensure_ascii=False)
    original = user + "\nJSON schema:\n" + schema_text
    error = ""
    for attempt in range(attempts):
        prompt = original if not error else original + "\nLần trước sai định dạng: " + error[:400] + "\nHãy trả lại đúng một JSON theo schema."
        response = llm.invoke([SystemMessage(content=system), HumanMessage(content=prompt)])
        text = response.content
        if not isinstance(text, str):
            raise ModelOutputError("Model không trả nội dung văn bản")
        entry = {"attempt": attempt + 1, "schema": schema.__name__,
                 "prompt_sha256": hashlib.sha256((system + prompt).encode()).hexdigest(), "raw": text}
        try:
            result = schema.model_validate(parse_json(text))
            if validate is not None:
                validate(result)
        except (ValueError, TypeError) as exc:
            error = str(exc)
            entry["error"] = error
            trace.append(entry)
            continue
        entry["valid"] = True
        trace.append(entry)
        return result
    raise ModelOutputError("Model trả JSON không hợp lệ sau các lần thử: " + error[:400])
