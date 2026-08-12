"""Tests for dataset split isolation logic."""

import os
from pathlib import Path

import pytest

from docextract.data.dataset import (
    Split,
    guard_train_path,
    load_eval_dataset,
    load_train_dataset,
    validate_eval_split,
    validate_train_split,
)

# Root directories that guard_train_path() is allowed to use.
ALLOWED = ("train", "validation")
FORBIDDEN = ("golden", "benchmark")


def _build_tmp_roots(tmp: Path) -> tuple[Path, tuple[Path, ...], tuple[Path, ...]]:
    """Create allowed/forbidden data roots under ``tmp``."""
    allowed = tuple((tmp / name).resolve() for name in ALLOWED)
    forbidden = tuple((tmp / name).resolve() for name in FORBIDDEN)
    for root in (*allowed, *forbidden):
        root.mkdir(parents=True)
    return tmp, allowed, forbidden


@pytest.fixture
def roots(tmp_path: Path) -> tuple[Path, tuple[Path, ...], tuple[Path, ...]]:
    """Yield temp dir with fresh allowed/forbidden data roots."""
    return _build_tmp_roots(tmp_path)


def test_train_path_accepted(roots: tuple[Path, tuple[Path, ...], tuple[Path, ...]]) -> None:
    _, allowed, _ = roots
    result = guard_train_path(allowed[0] / "file.json", roots=allowed)
    assert result == (allowed[0] / "file.json").resolve()


def test_train_directory_itself_accepted(
    roots: tuple[Path, tuple[Path, ...], tuple[Path, ...]],
) -> None:
    _, allowed, _ = roots
    result = guard_train_path(allowed[0], roots=allowed)
    assert result == allowed[0]


def test_validation_path_accepted(roots: tuple[Path, tuple[Path, ...], tuple[Path, ...]]) -> None:
    _, allowed, _ = roots
    result = guard_train_path(allowed[1] / "file.json", roots=allowed)
    assert result == (allowed[1] / "file.json").resolve()


def test_nested_subdir_under_train_accepted(
    roots: tuple[Path, tuple[Path, ...], tuple[Path, ...]],
) -> None:
    _, allowed, _ = roots
    nested = allowed[0] / "sub" / "dir" / "file.json"
    result = guard_train_path(nested, roots=allowed)
    assert result == nested.resolve()


def test_golden_path_rejected(roots: tuple[Path, tuple[Path, ...], tuple[Path, ...]]) -> None:
    _, allowed, forbidden = roots
    with pytest.raises(ValueError):
        guard_train_path(forbidden[0] / "file.json", roots=allowed)


def test_benchmark_path_rejected(roots: tuple[Path, tuple[Path, ...], tuple[Path, ...]]) -> None:
    _, allowed, forbidden = roots
    with pytest.raises(ValueError):
        guard_train_path(forbidden[1] / "file.json", roots=allowed)


def test_golden_directory_itself_rejected(
    roots: tuple[Path, tuple[Path, ...], tuple[Path, ...]],
) -> None:
    _, allowed, forbidden = roots
    with pytest.raises(ValueError):
        guard_train_path(forbidden[0], roots=allowed)


def test_parent_traversal_outside_roots_rejected(
    roots: tuple[Path, tuple[Path, ...], tuple[Path, ...]],
) -> None:
    _, allowed, forbidden = roots
    # ../golden traversal from train must be rejected even though the
    # target exists as a real directory.
    with pytest.raises(ValueError):
        guard_train_path(allowed[0] / ".." / "golden" / "file.json", roots=allowed)
    assert forbidden[0].exists()


def test_unknown_sibling_rejected(roots: tuple[Path, tuple[Path, ...], tuple[Path, ...]]) -> None:
    _, allowed, _ = roots
    with pytest.raises(ValueError):
        guard_train_path(allowed[0] / ".." / "sneaky" / "file.json", roots=allowed)


def test_case_insensitive_ancestor_check(
    roots: tuple[Path, tuple[Path, ...], tuple[Path, ...]],
) -> None:
    _, allowed, _ = roots
    # "GOLDEN" in an ancestor path must be rejected regardless of case.
    with pytest.raises(ValueError):
        guard_train_path(allowed[0] / ".." / "GOLDEN" / "file.json", roots=allowed)


def test_direct_data_root_not_allowed(
    roots: tuple[Path, tuple[Path, ...], tuple[Path, ...]],
) -> None:
    tmp, allowed, _ = roots
    # Only train/ and validation/ subdirs are allowed, not the data root itself.
    with pytest.raises(ValueError):
        guard_train_path(tmp, roots=allowed)


def test_symlink_to_golden_rejected(
    roots: tuple[Path, tuple[Path, ...], tuple[Path, ...]],
) -> None:
    _, allowed, forbidden = roots
    link = allowed[0] / "link_to_golden.json"
    try:
        os.symlink(forbidden[0] / "file.json", link)
    except (OSError, NotImplementedError):
        pytest.skip("symlink not supported on this platform")
    with pytest.raises(ValueError):
        guard_train_path(link, roots=allowed)


def test_symlink_to_train_accepted(
    roots: tuple[Path, tuple[Path, ...], tuple[Path, ...]],
) -> None:
    _, allowed, _ = roots
    real = allowed[0] / "real.json"
    real.write_text("{}")
    link = allowed[0] / "link_to_train.json"
    try:
        os.symlink(real, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlink not supported on this platform")
    result = guard_train_path(link, roots=allowed)
    assert result == real.resolve()


def test_exact_root_match_accepted(
    roots: tuple[Path, tuple[Path, ...], tuple[Path, ...]],
) -> None:
    _, allowed, _ = roots
    result = guard_train_path(Path(str(allowed[0])), roots=allowed)
    assert result == allowed[0]


# --- Symlink defense: target path that escapes through a resolved link ---


def test_symlinked_dir_inside_train_escaping_rejected(
    roots: tuple[Path, tuple[Path, ...], tuple[Path, ...]],
) -> None:
    _, allowed, forbidden = roots
    outer = allowed[0] / "esc"
    outer.mkdir()
    target = forbidden[0] / "sneaky"
    target.mkdir()
    try:
        os.symlink(target.resolve(), (outer / "link"))
    except (OSError, NotImplementedError):
        pytest.skip("symlink not supported on this platform")
    # The path under train/ actually resolves into golden/.
    with pytest.raises(ValueError):
        guard_train_path(outer / "link" / "file.json", roots=allowed)


# --- validate_train_split / validate_eval_split ---


def test_train_split_allowed() -> None:
    validate_train_split(Split.TRAIN)


def test_validation_split_allowed_for_training() -> None:
    validate_train_split(Split.VALIDATION)


def test_golden_split_rejected_for_training() -> None:
    with pytest.raises(ValueError):
        validate_train_split(Split.GOLDEN)


def test_benchmark_split_rejected_for_training() -> None:
    with pytest.raises(ValueError):
        validate_train_split(Split.BENCHMARK)


def test_eval_split_accepts_train() -> None:
    validate_eval_split(Split.TRAIN)


def test_eval_split_accepts_validation() -> None:
    validate_eval_split(Split.VALIDATION)


def test_eval_split_accepts_golden() -> None:
    validate_eval_split(Split.GOLDEN)


def test_eval_split_accepts_benchmark() -> None:
    validate_eval_split(Split.BENCHMARK)


def test_unknown_split_value_rejected() -> None:
    with pytest.raises(ValueError):
        Split("sneaky")


def test_split_membership_values() -> None:
    assert {s.value for s in Split} == {"train", "validation", "golden", "benchmark"}


def test_split_str_matches_value() -> None:
    assert str(Split.TRAIN) == "train"
    assert str(Split.GOLDEN) == "golden"


# --- load stubs ---


def test_load_train_dataset_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        load_train_dataset()


def test_load_eval_dataset_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        load_eval_dataset(Split.GOLDEN)
