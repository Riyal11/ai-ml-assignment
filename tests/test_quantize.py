"""Tests for the quantization CLI."""

from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.quantize import quantize_model


def test_missing_model_path_raises(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"
    with pytest.raises(FileNotFoundError, match="model path not found"):
        quantize_model(missing, tmp_path / "out")


def test_valid_path_returns_path(tmp_path: Path) -> None:
    model_dir = tmp_path / "merged-model"
    model_dir.mkdir()
    output_dir = tmp_path / "gguf"
    expected = output_dir / "merged-model.Q4_K_M.gguf"

    with (
        patch("scripts.quantize._check_llama_cpp_available"),
        patch("scripts.quantize._run_gguf_conversion") as mock_convert,
    ):
        result = quantize_model(model_dir, output_dir)

    assert result == expected
    mock_convert.assert_called_once_with(model_dir, expected)


def test_awq_raises_not_implemented(tmp_path: Path) -> None:
    model_dir = tmp_path / "merged-model"
    model_dir.mkdir()
    with pytest.raises(NotImplementedError, match="AWQ"):
        quantize_model(model_dir, tmp_path / "out", method="awq")


def test_gptq_raises_not_implemented(tmp_path: Path) -> None:
    model_dir = tmp_path / "merged-model"
    model_dir.mkdir()
    with pytest.raises(NotImplementedError, match="GPTQ"):
        quantize_model(model_dir, tmp_path / "out", method="gptq")


def test_invalid_method_raises_value_error(tmp_path: Path) -> None:
    model_dir = tmp_path / "merged-model"
    model_dir.mkdir()
    with pytest.raises(ValueError, match="unknown quantization method"):
        quantize_model(model_dir, tmp_path / "out", method="gguf-q8_0")
