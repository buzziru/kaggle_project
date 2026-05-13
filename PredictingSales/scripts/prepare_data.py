from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from predicting_sales.config import DEFAULT_DATA_DIR, DEFAULT_MATRIX_PATH  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the PredictingSales feature matrix.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_MATRIX_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from predicting_sales.pipeline import build_and_save_feature_matrix

    matrix = build_and_save_feature_matrix(args.data_dir, args.output)
    print(f"Saved feature matrix to {args.output}")
    print(f"Shape: {matrix.shape}")
    print(f"Duplicate keys: {matrix.duplicated(subset=['date_block_num', 'shop_id', 'item_id']).sum()}")


if __name__ == "__main__":
    main()
