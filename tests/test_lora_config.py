"""Tests for LoRA / QLoRA / DoRA adapter configuration builders."""

import pytest
from peft import LoraConfig
from transformers import BitsAndBytesConfig

from docextract.train.lora_config import (
    get_adapter_config,
    get_dora_config,
    get_lora_config,
    get_qlora_config,
)
from docextract.train.run_config import RunConfig

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def config(method: str = "lora") -> RunConfig:
    """Build a valid RunConfig for adapter tests."""
    return RunConfig(
        method=method,  # type: ignore[arg-type]
        r=16,
        alpha=32,
        target_modules=TARGET_MODULES,
        learning_rate=0.0002,
        lr_scheduler_type="cosine",
        max_seq_length=2048,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=3.0,
    )


def test_get_lora_config_has_correct_rank_alpha() -> None:
    lora_config = get_lora_config(config())
    assert lora_config.r == 16
    assert lora_config.lora_alpha == 32


def test_get_qlora_config_returns_tuple() -> None:
    lora_config, bnb = get_qlora_config(config("qlora"))
    assert isinstance(lora_config, LoraConfig)
    assert isinstance(bnb, BitsAndBytesConfig)
    assert bnb.load_in_4bit is True
    assert bnb.bnb_4bit_quant_type == "nf4"
    assert bnb.load_in_4bit is True


def test_target_modules_match_input() -> None:
    lora_config = get_lora_config(config())
    assert set(lora_config.target_modules) == set(TARGET_MODULES)


def test_qlora_target_modules_match_input() -> None:
    lora_config, _ = get_qlora_config(config("qlora"))
    assert set(lora_config.target_modules) == set(TARGET_MODULES)


def test_get_adapter_config_lora_dispatches() -> None:
    result = get_adapter_config(config("lora"))
    assert isinstance(result, LoraConfig)


def test_get_adapter_config_qlora_dispatches() -> None:
    result = get_adapter_config(config("qlora"))
    assert isinstance(result, tuple)


def test_get_dora_config_use_dora() -> None:
    lora_config = get_dora_config(config("dora"))
    assert lora_config.use_dora is True


def test_get_adapter_config_dora_dispatches() -> None:
    result = get_adapter_config(config("dora"))
    assert isinstance(result, LoraConfig)


@pytest.mark.parametrize("method", ["lora", "qlora", "dora"])
def test_all_methods_construct(method: str) -> None:
    cfg = config(method)
    result = get_adapter_config(cfg)
    if cfg.method == "qlora":
        assert isinstance(result, tuple)
    else:
        assert isinstance(result, LoraConfig)
