"""Tests for tokenizer wrapper utilities (mocked — no model downloads)."""

import pytest

from docextract.data.tokenizer_utils import (
    apply_chat_template,
    get_max_length,
    load_tokenizer,
)


class FakeTokenizer:
    """Duck-typed tokenizer stand-in for unit tests."""

    model_max_length = 4096

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        return (
            "[SYS] [USER] [ASSISTANT]"
            if add_generation_prompt
            else "[SYS] [USER] [ASSISTANT]\nresolved"
        )


@pytest.fixture
def fake_tokenizer() -> FakeTokenizer:
    """Return a fresh fake tokenizer."""
    return FakeTokenizer()


def test_load_tokenizer_returns_tokenizer(monkeypatch, fake_tokenizer) -> None:
    monkeypatch.setattr(
        "docextract.data.tokenizer_utils.AutoTokenizer.from_pretrained",
        lambda model_id, revision=None, local_files_only=False: fake_tokenizer,
    )
    monkeypatch.setattr(
        "docextract.data.tokenizer_utils.PreTrainedTokenizerFast",
        FakeTokenizer,
    )
    loaded = load_tokenizer("mock/model")
    assert isinstance(loaded, FakeTokenizer)


def test_load_tokenizer_oserror_raises(monkeypatch) -> None:
    def _fail(model_id: str, revision: str | None = None, local_files_only: bool = False) -> None:
        _ = (model_id, revision, local_files_only)
        raise OSError("boom")

    monkeypatch.setattr("docextract.data.tokenizer_utils.AutoTokenizer.from_pretrained", _fail)
    with pytest.raises(OSError):
        load_tokenizer("mock/missing")


def test_apply_chat_template_returns_string(fake_tokenizer) -> None:
    messages = [
        {"role": "system", "content": "extract JSON"},
        {"role": "user", "content": "doc text"},
    ]
    rendered = apply_chat_template(messages, fake_tokenizer)
    assert isinstance(rendered, str)
    assert "resolved" in rendered


def test_get_max_length_positive(fake_tokenizer) -> None:
    assert get_max_length(fake_tokenizer) > 0
