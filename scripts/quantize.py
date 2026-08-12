"""CLI for converting a merged HuggingFace model to quantized formats."""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_GGUF_METHOD = "gguf-q4_k_m"
_SUPPORTED_METHODS = (_GGUF_METHOD, "awq", "gptq")
_DEFAULT_OUTPUT_DIR = Path("artifacts/gguf")


def _check_llama_cpp_available() -> None:
    """Verify ``llama_cpp`` is importable before GGUF conversion.

    Raises:
        SystemExit: With code 1 when ``llama-cpp-python`` is not installed.
    """
    try:
        import llama_cpp  # noqa: F401
    except ImportError:
        logger.error(
            "llama-cpp-python is required for GGUF conversion. "
            "Install with: uv pip install llama-cpp-python"
        )
        raise SystemExit(1) from None


def _find_convert_script() -> Path | None:
    """Locate ``convert_hf_to_gguf.py`` on ``PATH`` or beside ``llama_cpp``."""
    found = shutil.which("convert_hf_to_gguf.py")
    if found is not None:
        return Path(found)

    try:
        import llama_cpp
    except ImportError:
        return None

    package_root = Path(llama_cpp.__file__).resolve().parent
    for candidate in (
        package_root / "convert_hf_to_gguf.py",
        package_root.parent / "convert_hf_to_gguf.py",
    ):
        if candidate.is_file():
            return candidate
    return None


def _run_gguf_conversion(model_path: Path, output_path: Path) -> None:
    """Convert a HuggingFace model directory to GGUF Q4_K_M.

    Uses ``convert_hf_to_gguf.py`` from llama.cpp when available, then
    quantizes with ``llama-quantize`` if present. Conversion is intentionally
    thin — callers are expected to have llama.cpp tooling installed.

    Args:
        model_path: Merged HuggingFace model directory.
        output_path: Destination ``.Q4_K_M.gguf`` file path.

    Raises:
        RuntimeError: If conversion tooling is missing or the subprocess fails.
    """
    convert_script = _find_convert_script()
    if convert_script is None:
        msg = (
            "convert_hf_to_gguf.py not found on PATH or beside llama_cpp. "
            "Install llama.cpp and expose convert_hf_to_gguf.py, or run conversion manually."
        )
        logger.error(msg)
        raise RuntimeError(msg)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fp16_gguf = output_path.parent / f"{model_path.name}.f16.gguf"

    logger.info("Converting %s to intermediate GGUF via %s", model_path, convert_script)
    convert_cmd = [
        sys.executable,
        str(convert_script),
        str(model_path),
        "--outfile",
        str(fp16_gguf),
        "--outtype",
        "f16",
    ]
    subprocess.run(convert_cmd, check=True)  # noqa: S603

    quantize_bin = shutil.which("llama-quantize")
    if quantize_bin is None:
        logger.warning("llama-quantize not found; leaving intermediate GGUF at %s", fp16_gguf)
        if fp16_gguf != output_path:
            fp16_gguf.replace(output_path)
        return

    logger.info("Quantizing %s to Q4_K_M", fp16_gguf)
    quantize_cmd = [quantize_bin, str(fp16_gguf), str(output_path), "Q4_K_M"]
    subprocess.run(quantize_cmd, check=True)  # noqa: S603
    if fp16_gguf.exists() and fp16_gguf != output_path:
        fp16_gguf.unlink()


def _quantize_gguf_q4_k_m(model_path: Path, output_dir: Path) -> Path:
    """Run GGUF Q4_K_M quantization for ``model_path``.

    Args:
        model_path: Merged HuggingFace model directory.
        output_dir: Directory for the quantized artifact.

    Returns:
        Path to ``<model-name>.Q4_K_M.gguf``.
    """
    _check_llama_cpp_available()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_path.name}.Q4_K_M.gguf"
    _run_gguf_conversion(model_path, output_path)
    logger.info("Wrote quantized model to %s", output_path)
    return output_path


def quantize_model(model_path: Path, output_dir: Path, method: str = _GGUF_METHOD) -> Path:
    """Convert a merged HuggingFace model to a quantized artifact.

    Args:
        model_path: Path to merged HuggingFace model directory.
        output_dir: Directory for quantized output files.
        method: Quantization method (``gguf-q4_k_m``, ``awq``, ``gptq``).

    Returns:
        Path to the quantized model file.

    Raises:
        FileNotFoundError: If ``model_path`` does not exist.
        ValueError: If ``method`` is not a recognized quantization method.
        NotImplementedError: If ``method`` is ``awq`` or ``gptq``.
        RuntimeError: If GGUF conversion tooling fails.
        SystemExit: If ``llama-cpp-python`` is not installed for GGUF conversion.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"model path not found: {model_path}")

    if method == "awq":
        raise NotImplementedError(
            "AWQ quantization is not implemented in this script; use the official "
            "Qwen AWQ tooling or integrate autoawq separately."
        )
    if method == "gptq":
        raise NotImplementedError(
            "GPTQ quantization is not implemented in this script; GPTQ support for "
            "Qwen3 is less mature than GGUF Q4_K_M for this assignment."
        )
    if method not in _SUPPORTED_METHODS:
        raise ValueError(
            f"unknown quantization method: {method!r}; "
            f"expected one of {', '.join(_SUPPORTED_METHODS)}"
        )

    return _quantize_gguf_q4_k_m(model_path, output_dir)


def _build_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser for quantization."""
    parser = argparse.ArgumentParser(
        description="Convert a merged HuggingFace model to a quantized format",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to merged HuggingFace model directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        help="Directory for quantized output (default: artifacts/gguf/)",
    )
    parser.add_argument(
        "--method",
        choices=_SUPPORTED_METHODS,
        default=_GGUF_METHOD,
        help="Quantization method (default: gguf-q4_k_m)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the quantization CLI.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        ``0`` on success, ``1`` on failure.
    """
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        output_path = quantize_model(args.model_path, args.output_dir, args.method)
        logger.info("Quantization complete: %s", output_path)
        return 0
    except SystemExit as exc:
        return int(exc.code) if exc.code is not None else 1
    except (FileNotFoundError, ValueError, NotImplementedError):
        logger.exception("Quantization failed")
        return 1
    except RuntimeError:
        logger.exception("Quantization tooling error")
        return 1


if __name__ == "__main__":
    sys.exit(main())
