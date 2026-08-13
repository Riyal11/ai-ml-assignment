"""Tests for training hyperparameter CSV logging and QLoRA device map."""

from pathlib import Path

import pytest

from docextract.train.trainer import (
    _HYPERPARAM_LOG_COLUMNS,
    _append_hyperparameter_row,
    _qlora_device_map,
)


def test_qlora_device_map_requires_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """QLoRA must fail fast when CUDA is unavailable."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="QLoRA requires a CUDA GPU"):
        _qlora_device_map()


def test_append_hyperparameter_row_ignores_extra_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rows with fields outside the CSV schema (e.g. output_dir) must not fail."""
    log_path = tmp_path / "experiments" / "hyperparameter_log.csv"
    monkeypatch.setattr("docextract.train.trainer._HYPERPARAM_LOG_PATH", log_path)

    _append_hyperparameter_row(
        {
            "run_id": "test-run",
            "method": "qlora",
            "r": 8,
            "alpha": 16,
            "target_modules": "q_proj,k_proj",
            "learning_rate": 0.0002,
            "lr_scheduler_type": "cosine",
            "max_seq_length": 512,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "num_train_epochs": 3,
            "base_model": "Qwen/Qwen3-4B-Instruct-2507",
            "seed": 42,
            "torch_dtype": "bfloat16",
            "trainable_params": 1000,
            "total_params": 4000000000,
            "train_time_sec": 12.5,
            "peak_gpu_mem_mb": 7000,
            "status": "failed",
            "diary_ref": "",
            "output_dir": "artifacts/run-001",
        }
    )

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert "output_dir" not in lines[0]
    assert "test-run" in lines[1]
    for col in _HYPERPARAM_LOG_COLUMNS:
        assert col in lines[0].split(",")
