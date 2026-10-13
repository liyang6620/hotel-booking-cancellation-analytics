"""Rebuild model, evaluation, and reporting artifacts from source data."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hotel_cancellation.training import train_and_evaluate  # noqa: E402, I001


if __name__ == "__main__":
    metadata = train_and_evaluate()
    metrics = metadata["test_metrics"]
    print(
        f"Built {metadata['model_version']}: "
        f"ROC-AUC={metrics['roc_auc']:.4f}, PR-AUC={metrics['pr_auc']:.4f}"
    )
