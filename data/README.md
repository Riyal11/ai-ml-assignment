# data/

Holds the datasets. Splits are **hard-isolated** — code in `src/docextract/data/dataset.py` enforces which pipeline may read which split.

## Splits

| Directory | Purpose |
|-----------|---------|
| `train/` | **Training only.** The fine-tuning pipeline reads this. |
| `validation/` | **Checkpoint selection / early stopping.** Read during training (not for gradient updates) and during evaluation. |
| `golden/` | **Final held-out evaluation ONLY. NEVER used for training.** 50 examples from the brief, do-not-touch during training/validation. |
| `benchmark/` | **Catastrophic-forgetting checks.** General-capability subset, base vs. fine-tuned comparison only. |

## Rules

- Training may read: `train/` + `validation/`.
- Training must never read: `golden/`, `benchmark/`.
- `golden/` is for final certification only — never tune hyperparameters on it.
- `benchmark/` is read-only comparison data — never tune on it.