"""
Check for sample-level overlap (data leakage) between preprocessed train/val/test splits.

Usage:
    python rad_dino/data/check_data_leakage.py --data-dir /path/to/preprocessed/RSNA-Pneumonia
    python rad_dino/data/check_data_leakage.py --data-dir /path/to/preprocessed/VinDr-Mammo/birads/multi_view
"""

import argparse
import logging
import os
import sys

import pandas as pd

from rad_dino.loggings.setup import init_logging

init_logging()
logger = logging.getLogger(__name__)


def _load_split(data_dir: str, filename: str) -> pd.DataFrame | None:
    path = os.path.join(data_dir, filename)
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path, index_col=0)
    return df


def _check_index_overlap(splits: dict[str, pd.DataFrame]) -> dict[str, set]:
    """Return a dict mapping each pair name to overlapping index values."""
    names = list(splits.keys())
    overlaps: dict[str, set] = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            common = set(splits[a].index) & set(splits[b].index)
            if common:
                overlaps[f"{a} vs {b}"] = common
    return overlaps


def _check_column_overlap(splits: dict[str, pd.DataFrame], column: str) -> dict[str, set]:
    """Check overlap on a specific column (e.g. patient_id) across splits."""
    relevant = {name: df for name, df in splits.items() if column in df.columns}
    if len(relevant) < 2:
        return {}
    names = list(relevant.keys())
    overlaps: dict[str, set] = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            common = set(relevant[a][column]) & set(relevant[b][column])
            if common:
                overlaps[f"{a} vs {b}"] = common
    return overlaps


def check_data_leakage(data_dir: str) -> bool:
    """Run all leakage checks and return True if no leakage is found."""
    splits: dict[str, pd.DataFrame] = {}
    for filename, label in [
        ("train_labels.csv", "train"),
        ("val_labels.csv", "val"),
        ("test_labels.csv", "test"),
    ]:
        df = _load_split(data_dir, filename)
        if df is not None:
            splits[label] = df

    if len(splits) < 2:
        logger.error(
            f"Need at least 2 split CSVs in {data_dir}, found: {list(splits.keys())}"
        )
        return False

    logger.info(f"Checking data leakage in: {data_dir}")
    for name, df in splits.items():
        logger.info(f"  {name}: {len(df)} samples (index: {df.index.name or 'unnamed'})")

    clean = True

    # 1) Sample-index overlap
    index_overlaps = _check_index_overlap(splits)
    if index_overlaps:
        clean = False
        for pair, ids in index_overlaps.items():
            logger.warning(f"DATA LEAKAGE — {len(ids)} overlapping sample IDs between {pair}")
            for sample_id in sorted(ids)[:10]:
                logger.warning(f"  - {sample_id}")
            if len(ids) > 10:
                logger.warning(f"  ... and {len(ids) - 10} more")
    else:
        logger.info("No sample-level leakage detected (index IDs are disjoint across splits).")

    # 2) Patient-level overlap (if column exists)
    patient_overlaps = _check_column_overlap(splits, "patient_id")
    if patient_overlaps:
        clean = False
        for pair, ids in patient_overlaps.items():
            logger.warning(f"PATIENT LEAKAGE — {len(ids)} overlapping patient_ids between {pair}")
            for pid in sorted(ids)[:10]:
                logger.warning(f"  - {pid}")
            if len(ids) > 10:
                logger.warning(f"  ... and {len(ids) - 10} more")
    elif any("patient_id" in df.columns for df in splits.values()):
        logger.info("No patient-level leakage detected.")

    if clean:
        logger.info("All checks passed — no data leakage detected.")
    return clean


def main():
    parser = argparse.ArgumentParser(
        description="Check for data leakage between preprocessed train/val/test splits."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to the preprocessed dataset directory containing "
             "train_labels.csv, test_labels.csv (and optionally val_labels.csv).",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        logger.error(f"Directory not found: {args.data_dir}")
        sys.exit(1)

    ok = check_data_leakage(args.data_dir)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
