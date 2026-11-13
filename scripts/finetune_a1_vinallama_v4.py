"""Prepare or explicitly fine-tune A1 v4 on VinaLLaMA-2.7B-chat, then export figures."""
from scripts.finetune_a1_qwen3_v4 import main
from src.training.a1_runtime import ROOT


if __name__ == "__main__":
    main(default_config=ROOT / "config/a1_vinallama_27b_v4.json", run_prefix="vinallama_27b_v4")
