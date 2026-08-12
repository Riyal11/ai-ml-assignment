"""LoRA / QLoRA / DoRA adapter configuration builders."""

from typing import cast

import torch
from peft import LoraConfig
from transformers import BitsAndBytesConfig

from docextract.train.run_config import RunConfig, validate_mandatory_fields

QWEN3_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _require(config: RunConfig, field: str) -> int:
    """Return a validated int field, narrowing after mandatory-field checks."""
    value = cast(int | None, getattr(config, field))
    if value is None:
        raise ValueError(f"mandatory field {field!r} is None")
    return value


def get_lora_config(config: RunConfig) -> LoraConfig:
    """Build a standard LoRA ``LoraConfig``.

    Args:
        config: Validated run config.

    Returns:
        PEFT ``LoraConfig`` for LoRA training.
    """
    validate_mandatory_fields(config)
    return LoraConfig(
        r=_require(config, "r"),
        lora_alpha=_require(config, "alpha"),
        target_modules=config.target_modules or QWEN3_TARGET_MODULES,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )


def get_qlora_config(config: RunConfig) -> tuple[LoraConfig, BitsAndBytesConfig]:
    """Build 4-bit QLoRA (LoRA config + NF4 quantization config).

    Args:
        config: Validated run config.

    Returns:
        A ``(LoraConfig, BitsAndBytesConfig)`` pair for QLoRA training.
    """
    validate_mandatory_fields(config)
    lora = get_lora_config(config)
    bnb = BitsAndBytesConfig(  # type: ignore[no-untyped-call]
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    return lora, bnb


def get_dora_config(config: RunConfig) -> LoraConfig:
    """Build a DoRA ``LoraConfig`` (weight-decomposed LoRA).

    Args:
        config: Validated run config.

    Returns:
        A ``LoraConfig`` with ``use_dora=True``.

    Raises:
        NotImplementedError: If the installed PEFT does not support DoRA.
    """
    validate_mandatory_fields(config)
    if "use_dora" not in LoraConfig.__dataclass_fields__:
        raise NotImplementedError(
            "installed PEFT does not support DoRA (use_dora); upgrade peft>=0.13"
        )
    return LoraConfig(
        r=_require(config, "r"),
        lora_alpha=_require(config, "alpha"),
        target_modules=config.target_modules or QWEN3_TARGET_MODULES,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        use_dora=True,
    )


def get_adapter_config(
    config: RunConfig,
) -> LoraConfig | tuple[LoraConfig, BitsAndBytesConfig]:
    """Dispatch to the adapter config builder matching ``config.method``.

    Args:
        config: Validated run config.

    Returns:
        A ``LoraConfig`` for ``lora``/``dora``, or a
        ``(LoraConfig, BitsAndBytesConfig)`` pair for ``qlora``.
    """
    if config.method == "qlora":
        return get_qlora_config(config)
    if config.method == "dora":
        return get_dora_config(config)
    return get_lora_config(config)
