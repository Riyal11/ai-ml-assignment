"""Dataset split management and train/eval path isolation."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parents[3] / "data"

TRAIN_ROOTS = (
    (DATA_ROOT / "train").resolve(),
    (DATA_ROOT / "validation").resolve(),
)


class Split(str, Enum):
    """Dataset split identifiers."""

    TRAIN = "train"
    VALIDATION = "validation"
    GOLDEN = "golden"
    BENCHMARK = "benchmark"

    def __str__(self) -> str:
        """Return the enum value when formatted as a string."""
        return self.value


@dataclass(frozen=True)
class DocumentRecord:
    """A single (document, target JSON) example with its split."""

    example_id: str
    document: str
    target: dict[str, object]
    language: str
    split: Split


def guard_train_path(
    path: Path, roots: tuple[Path, ...] | None = None
) -> Path:
    """Resolve ``path`` and reject anything outside the allowed train roots.

    Uses ``Path.resolve()`` (which also resolves symlinks) and a
    case-insensitive ancestor check so ``data/golden`` and
    ``data/benchmark`` can never be used for training.
    """
    allowed = roots if roots is not None else TRAIN_ROOTS
    resolved = path.resolve()
    resolved_str = resolved.as_posix().lower()
    for root in allowed:
        root_resolved = Path(root).resolve()
        root_str = root_resolved.as_posix().lower()
        if resolved_str == root_str or resolved_str.startswith(root_str + "/"):
            return resolved
    raise ValueError(
        f"path {path} resolves to {resolved}, outside allowed train roots "
        f"({', '.join(Path(r).resolve().as_posix() for r in allowed)})"
    )


def validate_train_split(split: Split) -> None:
    """Raise if ``split`` is not allowed in the training pipeline."""
    if split in (Split.GOLDEN, Split.BENCHMARK):
        raise ValueError(
            f"split {split.value!r} is reserved and must not be used for training"
        )


def validate_eval_split(split: Split) -> None:
    """Return without error for any valid split (evaluation may use all)."""
    # All four splits are valid evaluation targets; nothing to reject.
    return None


def load_train_dataset() -> list[DocumentRecord]:
    """Load the training dataset.

    Not yet implemented. Will read ``data/train`` and ``data/validation``.
    """
    raise NotImplementedError("load_train_dataset not implemented yet")


def load_eval_dataset(split: Split) -> list[DocumentRecord]:
    """Load an evaluation dataset for ``split``.

    Not yet implemented. Will read from the corresponding ``data/``
    subdirectory (golden, benchmark, or validation).
    """
    raise NotImplementedError("load_eval_dataset not implemented yet")
