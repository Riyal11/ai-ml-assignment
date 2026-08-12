"""Pydantic config dataclass for a single fine-tuning run."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import ConfigDict, Field, model_validator
from pydantic.dataclasses import dataclass

Method = Literal["lora", "qlora", "dora"]

_MANDATED_FIELDS: tuple[str, ...] = (
    "run_id",
    "method",
    "r",
    "alpha",
    "target_modules",
    "learning_rate",
    "lr_scheduler_type",
    "max_seq_length",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "num_train_epochs",
)


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class RunConfig:
    """Configuration for one supervised fine-tuning run.

    The ten mandated fields mirror the assignment's hyperparameter log
    and must all be non-null before training starts (see
    ``validate_mandatory_fields``).

    Attributes:
        method: PEFT method — ``lora``, ``qlora`` or ``dora``.
        r: LoRA rank, must be > 0.
        alpha: LoRA alpha, must be > 0.
        target_modules: Modules to adapt, must be non-empty.
        learning_rate: Peak learning rate, must be > 0.
        lr_scheduler_type: Scheduler name (e.g. ``cosine``, ``linear``).
        max_seq_length: Maximum token sequence length.
        per_device_train_batch_size: Batch size per device.
        gradient_accumulation_steps: Gradient accumulation steps.
        num_train_epochs: Number of training epochs.
        run_id: Auto-generated ``YYYYMMDD-HHMMSS-<method>`` if not given.
        base_model: Hugging Face model identifier.
        seed: Random seed.
        torch_dtype: Model dtype (e.g. ``bfloat16``).
        trainable_params: Filled in after training; param count pulled from log.
        total_params: Filled in after training; total param count.
        output_dir: Directory for adapter checkpoints.
    """

    method: Method | None = Field(...)
    r: int | None = Field(gt=0)
    alpha: int | None = Field(gt=0)
    target_modules: list[str] | None = Field(min_length=1)
    learning_rate: float | None = Field(gt=0)
    lr_scheduler_type: str | None = Field(...)
    max_seq_length: int | None = Field(gt=0)
    per_device_train_batch_size: int | None = Field(gt=0)
    gradient_accumulation_steps: int | None = Field(gt=0)
    num_train_epochs: float | None = Field(gt=0)

    run_id: str | None = None
    base_model: str = "Qwen/Qwen3-4B-Instruct"
    seed: int = 42
    torch_dtype: str = "bfloat16"
    trainable_params: int | None = None
    total_params: int | None = None
    output_dir: Path = Path("artifacts")

    @model_validator(mode="after")
    def _ensure_run_id(self) -> RunConfig:
        """Generate ``run_id`` from timestamp + method when not supplied."""
        if self.run_id is None:
            self.run_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{self.method}"
        return self


def validate_mandatory_fields(config: RunConfig) -> None:
    """Raise ``ValueError`` if any mandated field is None.

    Args:
        config: Run config to validate.

    Raises:
        ValueError: Listing every mandated field that is ``None``.
    """
    missing = [name for name in _MANDATED_FIELDS if getattr(config, name) is None]
    if missing:
        raise ValueError(f"mandatory fields missing from config: {', '.join(missing)}")


def run_config_to_dict(config: RunConfig) -> dict[str, Any]:
    """Flatten a ``RunConfig`` to a plain dict of its fields."""
    fields: dict[str, Any] = {}
    for name in config.__dataclass_fields__:
        field_value = getattr(config, name)
        if name in ("run_id", "output_dir"):
            fields[name] = str(field_value)
        elif isinstance(field_value, list):
            fields[name] = ",".join(field_value)
        else:
            fields[name] = field_value
    return fields
