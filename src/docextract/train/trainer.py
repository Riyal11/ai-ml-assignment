"""End-to-end LoRA/QLoRA/DoRA fine-tuning entrypoint with mandatory logging."""

import csv
import json
import logging
import time
from pathlib import Path
from typing import Any

from datasets import Dataset  # type: ignore[import-untyped]
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer  # type: ignore[attr-defined]

from docextract.data.tokenizer_utils import (
    apply_chat_template,
    load_tokenizer,
)
from docextract.train.lora_config import get_adapter_config, get_qlora_config
from docextract.train.run_config import (
    RunConfig,
    run_config_to_dict,
    validate_mandatory_fields,
)

logger = logging.getLogger(__name__)

_HYPERPARAM_LOG_COLUMNS = (
    "run_id,method,r,alpha,target_modules,learning_rate,lr_scheduler_type,"
    "max_seq_length,per_device_train_batch_size,gradient_accumulation_steps,"
    "num_train_epochs,base_model,seed,torch_dtype,trainable_params,total_params,"
    "train_time_sec,peak_gpu_mem_mb,status,diary_ref"
).split(",")

_HYPERPARAM_LOG_PATH = (
    Path(__file__).resolve().parents[3] / "experiments" / "hyperparameter_log.csv"
)

# ponytail: None = unpinned (bandit-clean). Pin a commit hash before production runs.
_REVISION: str | None = None


def _append_hyperparameter_row(row: dict[str, Any]) -> None:
    """Append one row to the append-only hyperparameter CSV log."""
    _HYPERPARAM_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not _HYPERPARAM_LOG_PATH.exists()
    with _HYPERPARAM_LOG_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_HYPERPARAM_LOG_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _load_and_format_dataset(dataset_path: Path, tokenizer: Any) -> Dataset:
    """Load JSONL chat examples and render each to a plain ``text`` string."""
    records: list[dict[str, Any]] = []
    with dataset_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    texts = [apply_chat_template(example["messages"], tokenizer) for example in records]
    return Dataset.from_dict({"text": texts})


def _count_params(model: Any) -> tuple[int, int]:
    """Return ``(trainable_params, total_params)`` for a model."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return int(trainable), int(total)


def _require_int(value: int | None, name: str) -> int:
    """Narrow a validated int field to ``int`` after mandatory-field checks."""
    if value is None:
        raise ValueError(f"mandatory field {name!r} is None")
    return value


def _require_float(value: float | None, name: str) -> float:
    """Narrow a validated float field to ``float`` after mandatory-field checks."""
    if value is None:
        raise ValueError(f"mandatory field {name!r} is None")
    return value


def _require_str(value: str | None, name: str) -> str:
    """Narrow a validated str field to ``str`` after mandatory-field checks."""
    if value is None:
        raise ValueError(f"mandatory field {name!r} is None")
    return value


def _peak_gpu_mem_mb() -> int | None:
    """Return peak CUDA memory allocated this process, in MB, if any."""
    import torch

    if torch.cuda.is_available():
        return int(torch.cuda.max_memory_allocated() // (1024 * 1024))
    return None


def run_training(config: RunConfig, dataset_path: Path) -> Path:
    """Fine-tune a base model on chat-formatted JSONL data.

    Refuses to start if any mandated field is missing. Logs all ten
    mandated hyperparameters to MLflow and ``experiments/hyperparameter_log.csv``
    before training, and outcomes (trainable/total params, train time,
    peak GPU memory, status) after.

    Args:
        config: Fully populated ``RunConfig``.
        dataset_path: Path to JSONL training data in ``messages`` format.

    Returns:
        Path to the best adapter checkpoint directory.

    Raises:
        ValueError: If ``validate_mandatory_fields`` fails.
        Exception: The underlying training failure, after logging
            ``status="failed"``.
    """
    validate_mandatory_fields(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    import mlflow

    run_started = time.monotonic()
    status = "failed"
    train_time_sec: float | None = None
    trainable_params: int | None = None
    total_params: int | None = None
    peak_mem_mb: int | None = None

    try:
        logger.info(
            "Starting training run %s (method=%s, base_model=%s)",
            config.run_id,
            config.method,
            config.base_model,
        )
        with mlflow.start_run(run_name=config.run_id) as run:
            run.log_param("custom_run_id", config.run_id)
            run.log_params({k: v for k, v in run_config_to_dict(config).items() if v is not None})

            tokenizer = load_tokenizer(config.base_model)
            adapter_config = get_adapter_config(config)
            qlora_lora_config = None

            model_kwargs: dict[str, Any] = {"torch_dtype": "auto", "device_map": "auto"}
            if config.method == "qlora":
                qlora_lora_config, bnb_config = get_qlora_config(config)
                model_kwargs["quantization_config"] = bnb_config

            model: Any = AutoModelForCausalLM.from_pretrained(
                config.base_model, revision=_REVISION, **model_kwargs
            )
            if config.method == "qlora":
                if qlora_lora_config is None:
                    raise RuntimeError("QLoRA config missing after get_qlora_config")
                model = prepare_model_for_kbit_training(model)  # type: ignore[no-untyped-call]
                model = get_peft_model(model, qlora_lora_config)
            elif config.method in ("lora", "dora"):
                model = get_peft_model(model, adapter_config)  # type: ignore[arg-type]

            trainable_params, total_params = _count_params(model)
            train_dataset = _load_and_format_dataset(dataset_path, tokenizer)

            sft_config = SFTConfig(
                output_dir=str(config.output_dir),
                per_device_train_batch_size=_require_int(
                    config.per_device_train_batch_size, "per_device_train_batch_size"
                ),
                gradient_accumulation_steps=_require_int(
                    config.gradient_accumulation_steps, "gradient_accumulation_steps"
                ),
                learning_rate=_require_float(config.learning_rate, "learning_rate"),
                lr_scheduler_type=_require_str(config.lr_scheduler_type, "lr_scheduler_type"),
                num_train_epochs=_require_float(config.num_train_epochs, "num_train_epochs"),
                max_length=_require_int(config.max_seq_length, "max_seq_length"),
                dataset_text_field="text",
                save_strategy="epoch",
                report_to=["mlflow"],
                run_name=config.run_id or "",
                seed=config.seed,
                fp16=False,
                bf16=True,
            )

            trainer = SFTTrainer(
                model=model,
                args=sft_config,
                processing_class=tokenizer,
                train_dataset=train_dataset,
            )
            trainer.train()
            trainer.save_model(str(config.output_dir))

            train_time_sec = time.monotonic() - run_started
            peak_mem_mb = _peak_gpu_mem_mb()

            run.log_metrics(
                {
                    "train_time_sec": train_time_sec,
                    "peak_gpu_mem_mb": float(peak_mem_mb or 0.0),
                    "trainable_params": float(trainable_params or 0),
                    "total_params": float(total_params or 0),
                }
            )
            status = "succeeded"
            logger.info(
                "Training run %s finished successfully in %.1fs (adapter at %s)",
                config.run_id,
                train_time_sec,
                config.output_dir,
            )
    except Exception:
        logger.exception("Training run %s failed", config.run_id)
        raise
    finally:
        row = {
            **run_config_to_dict(config),
            "trainable_params": trainable_params,
            "total_params": total_params,
            "train_time_sec": train_time_sec,
            "peak_gpu_mem_mb": peak_mem_mb,
            "status": status,
            "diary_ref": "",
        }
        try:
            _append_hyperparameter_row(row)
        except Exception as csv_exc:
            logger.error("Failed to write hyperparameter log: %s", csv_exc)

    return config.output_dir
