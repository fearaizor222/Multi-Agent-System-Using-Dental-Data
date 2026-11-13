"""Không được huấn luyện vào prompt hoặc âm thầm cắt/mask sai đáp án."""
import pytest

from scripts.check_a1_environment import encode_assistant_sample, completion_only_loss


MESSAGES = [{"role": "system", "content": "policy"}, {"role": "user", "content": "question"},
            {"role": "assistant", "content": '{"decision":"route"}'}]


class TokenizerStub:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        return [10, 11, 12] if kwargs["add_generation_prompt"] else [10, 11, 12, 20, 21, 99]


def test_system_and_user_masked_but_answer_and_eos_supervised():
    tokens, labels, prompt_length = encode_assistant_sample(TokenizerStub(), MESSAGES)
    assert prompt_length == 3
    assert tokens == [10, 11, 12, 20, 21, 99]
    assert labels == [-100, -100, -100, 20, 21, 99]


def test_template_prefix_drift_rejected_before_training():
    class BadTokenizer(TokenizerStub):
        def apply_chat_template(self, messages, **kwargs):
            return [10, 11, 12] if kwargs["add_generation_prompt"] else [10, 11, 88, 20, 21, 99]
    with pytest.raises(ValueError, match="prefix"):
        encode_assistant_sample(BadTokenizer(), MESSAGES)


def test_empty_completion_and_wrong_roles_rejected():
    class EmptyCompletion(TokenizerStub):
        def apply_chat_template(self, messages, **kwargs):
            return [10, 11, 12]
    with pytest.raises(ValueError, match="Không có token"):
        encode_assistant_sample(EmptyCompletion(), MESSAGES)
    with pytest.raises(ValueError, match="system/user/assistant"):
        encode_assistant_sample(TokenizerStub(), MESSAGES[:-1])


def test_selective_logits_loss_and_gradients_match_full_masked_causal_loss():
    import torch
    from types import SimpleNamespace
    torch.manual_seed(7)
    full_logits = torch.randn(1, 9, 13, requires_grad=True)
    sparse_logits = full_logits.detach().clone().requires_grad_()
    tokens = torch.randint(0, 13, (1, 9))
    labels = tokens.clone()
    labels[:, :6] = -100
    batch = {"input_ids": tokens, "labels": labels, "attention_mask": torch.ones_like(tokens)}
    reference = torch.nn.functional.cross_entropy(full_logits[:, :-1].reshape(-1, 13), labels[:, 1:].reshape(-1), ignore_index=-100)

    class FakeModel:
        def __call__(self, input_ids, attention_mask, logits_to_keep):
            return SimpleNamespace(logits=sparse_logits[:, logits_to_keep, :])

    selected = completion_only_loss(FakeModel(), batch, 6)
    assert torch.allclose(reference, selected)
    reference.backward()
    selected.backward()
    assert torch.allclose(full_logits.grad, sparse_logits.grad)
