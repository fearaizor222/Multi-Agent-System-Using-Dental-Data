"""Explicit prompt formats for reviewed A1 checkpoints; never modify downloaded files."""
import hashlib

VINALLAMA_REPO = "vilm/vinallama-2.7b-chat"
VINALLAMA_CHATML = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' }}"
    "{% if not loop.last or message['role'] != 'assistant' %}{{ '\\n' }}{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)


def configure_tokenizer(tokenizer, config):
    format_name = config.get("chat_format", "native")
    if format_name == "vinallama_chatml_v1":
        if config["base_model"] != VINALLAMA_REPO:
            raise ValueError("VinaLLaMA ChatML requires the reviewed VinaLLaMA checkpoint")
        for token in ("<|im_start|>", "<|im_end|>", "<pad>"):
            ids = tokenizer.encode(token, add_special_tokens=False)
            if len(ids) != 1 or ids[0] == tokenizer.unk_token_id:
                raise ValueError("Missing VinaLLaMA special token: " + token)
        tokenizer.chat_template = VINALLAMA_CHATML
        tokenizer.eos_token = "<|im_end|>"
        tokenizer.pad_token = "<pad>"
    elif format_name != "native":
        raise ValueError("Unsupported A1 chat format: " + format_name)
    elif config["base_model"] == VINALLAMA_REPO:
        raise ValueError("VinaLLaMA needs chat_format=vinallama_chatml_v1")
    if not tokenizer.chat_template:
        raise ValueError("Checkpoint has no reviewed chat template")
    return tokenizer


def load_tokenizer(config, root):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(root / config["model_dir"], local_files_only=True, trust_remote_code=False)
    return configure_tokenizer(tokenizer, config)


def tokenizer_identity(tokenizer, config):
    return {"chat_format": config.get("chat_format", "native"),
            "template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
            "eos_token_id": tokenizer.eos_token_id, "pad_token_id": tokenizer.pad_token_id}
