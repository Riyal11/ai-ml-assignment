"""Tests for RunConfig construction and mandatory-field validation."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from docextract.train.run_config import RunConfig, run_config_to_dict, validate_mandatory_fields


def valid_config_fields() -> dict[str, object]:
    """Return a complete, valid field set for RunConfig."""
    return {
        "method": "lora",
        "r": 16,
        "alpha": 32,
        "target_modules": ["q_proj", "k_proj"],
        "learning_rate": 0.0002,
        "lr_scheduler_type": "cosine",
        "max_seq_length": 2048,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "num_train_epochs": 3.0,
    }


def test_valid_config_constructs() -> None:
    config = RunConfig(**valid_config_fields())
    assert config.r == 16
    assert config.alpha == 32
    assert config.method == "lora"


def test_missing_mandatory_field_raises() -> None:
    fields = valid_config_fields()
    del fields["learning_rate"]
    with pytest.raises(ValidationError):
        RunConfig(**fields)


def test_invalid_method_raises() -> None:
    fields = valid_config_fields()
    fields["method"] = "svd"
    with pytest.raises(ValidationError):
        RunConfig(**fields)


def test_r_zero_raises() -> None:
    fields = valid_config_fields()
    fields["r"] = 0
    with pytest.raises(ValidationError):
        RunConfig(**fields)


def test_r_negative_raises() -> None:
    fields = valid_config_fields()
    fields["r"] = -2
    with pytest.raises(ValidationError):
        RunConfig(**fields)


def test_empty_target_modules_raises() -> None:
    fields = valid_config_fields()
    fields["target_modules"] = []
    with pytest.raises(ValidationError):
        RunConfig(**fields)


def test_validate_mandatory_fields_passes_for_complete() -> None:
    config = RunConfig(**valid_config_fields())
    validate_mandatory_fields(config)


def test_validate_mandatory_fields_fails_for_incomplete() -> None:
    fields = valid_config_fields()
    fields["method"] = None
    fields["num_train_epochs"] = None
    config = RunConfig(**fields)
    with pytest.raises(ValueError) as excinfo:
        validate_mandatory_fields(config)
    assert "method" in str(excinfo.value)
    assert "num_train_epochs" in str(excinfo.value)


def test_run_id_auto_generated() -> None:
    config = RunConfig(**valid_config_fields())
    assert config.run_id is not None
    assert config.run_id.startswith(datetime.now().strftime("%Y%m%d-"))
    assert config.run_id.endswith("-lora")


def test_run_id_preserved_when_given() -> None:
    fields = valid_config_fields()
    fields["run_id"] = "custom-run-1"
    config = RunConfig(**fields)
    assert config.run_id == "custom-run-1"


def test_output_dir_default() -> None:
    config = RunConfig(**valid_config_fields())
    assert str(config.output_dir) == "artifacts"


def test_run_config_to_dict() -> None:
    fields = valid_config_fields()
    fields["run_id"] = "run-123"
    config = RunConfig(**fields)
    result = run_config_to_dict(config)
    assert result["run_id"] == "run-123"
    assert result["target_modules"] == "q_proj,k_proj"
    assert result["method"] == "lora"
