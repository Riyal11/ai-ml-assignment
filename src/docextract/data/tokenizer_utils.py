"""Thin wrappers around ``transformers`` tokenizer loading and templating."""

import logging
from typing import Any, cast

from transformers import AutoTokenizer, PreTrainedTokenizerFast

logger = logging.getLogger(__name__)

# TODO: Hindi-specific tokenization fixes for Qwen3-4B yet to land.
# Investigate Devanagari token efficiency + whether to pad Hindi prompts.


def load_tokenizer(
    model_id: str,
    revision: str | None = None,
    *,
    local_files_only: bool = False,
) -> PreTrainedTokenizerFast:
    """Load a fast tokenizer for ``model_id``.

    Args:
        model_id: Hugging Face model identifier.
        revision: Optional pinned revision (branch, tag, or commit hash).

    Returns:
        The loaded tokenizer.

    Raises:
        OSError: If the tokenizer cannot be downloaded or loaded.
        ValueError: If the loaded tokenizer is not a fast tokenizer.
    """
    try:
        tokenizer: Any = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            local_files_only=local_files_only,
        )
    except OSError:
        logger.exception("Failed to load tokenizer for %s", model_id)
        raise
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            f"model {model_id} returned a slow tokenizer; require `fast` to "
            "support apply_chat_template reliably"
        )
    return tokenizer


def apply_chat_template(
    messages: list[dict[str, Any]],
    tokenizer: PreTrainedTokenizerFast,
    add_generation_prompt: bool = False,
) -> str:
    """Apply the tokenizer's built-in chat template.

    Args:
        messages: List of ``{"role": ..., "content": ...}`` dicts.
        tokenizer: Tokenizer whose chat template to apply.
        add_generation_prompt: Append the assistant-turn marker when True.

    Returns:
        The rendered chat string.
    """
    rendered = cast(
        str,
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        ),
    )
    return rendered


def get_max_length(tokenizer: PreTrainedTokenizerFast) -> int:
    """Return the model's maximum input length.

    Args:
        tokenizer: Loaded tokenizer.

    Returns:
        The tokenizer's declared ``model_max_length``.
    """
    return int(tokenizer.model_max_length)
