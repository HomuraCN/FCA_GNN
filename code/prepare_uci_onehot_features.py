#!/usr/bin/env python3
"""Create label-free one-hot feature CSVs for UCI-style .data files.

The one-hot column order intentionally matches the Java UCIParser:
scan rows from top to bottom, columns from left to right, and assign a
new feature index the first time a "column_index:value" key appears.
Unlike UCIParser, the final column is treated as the label and excluded
from the generated feature matrix.
"""

from __future__ import annotations

import argparse
import csv
from collections import OrderedDict
from pathlib import Path


DEFAULT_INPUT = (
    Path(__file__).resolve().parents[2]
    / "FCA"
    / "src"
    / "main"
    / "java"
    / "data"
    / "uci"
    / "car.data"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "car"


def read_rows(path: Path) -> list[list[str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = [[cell.strip() for cell in row] for row in csv.reader(f)]
    return [row for row in rows if row]


def build_feature_map(rows: list[list[str]], label_column: int) -> OrderedDict[tuple[int, str], int]:
    feature_map: OrderedDict[tuple[int, str], int] = OrderedDict()
    for row in rows:
        for column_index, value in enumerate(row):
            if column_index == label_column:
                continue
            key = (column_index, value)
            if key not in feature_map:
                feature_map[key] = len(feature_map)
    return feature_map


def write_onehot_features(
    rows: list[list[str]],
    feature_map: OrderedDict[tuple[int, str], int],
    label_column: int,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_count = len(feature_map)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        for row in rows:
            encoded = [0] * feature_count
            for column_index, value in enumerate(row):
                if column_index == label_column:
                    continue
                encoded[feature_map[(column_index, value)]] = 1
            writer.writerow(encoded)


def write_raw_copy(rows: list[list[str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a UCI-style .data file into a label-free one-hot "
            "feature matrix for FCA_GNN notebooks."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--dataset",
        default="car",
        help="Dataset basename used for output files, e.g. car.",
    )
    parser.add_argument(
        "--label-column",
        type=int,
        default=-1,
        help="Zero-based label column. Defaults to the final column.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()

    rows = read_rows(input_path)
    if not rows:
        raise ValueError(f"No rows found in {input_path}")

    column_count = len(rows[0])
    bad_rows = [index + 1 for index, row in enumerate(rows) if len(row) != column_count]
    if bad_rows:
        preview = ", ".join(map(str, bad_rows[:10]))
        raise ValueError(f"Inconsistent column counts at rows: {preview}")

    label_column = args.label_column if args.label_column >= 0 else column_count + args.label_column
    if label_column < 0 or label_column >= column_count:
        raise ValueError(f"label-column out of range: {args.label_column}")

    feature_map = build_feature_map(rows, label_column)
    features_path = output_dir / f"{args.dataset}.data.cleaned.csv"
    raw_path = output_dir / f"{args.dataset}.data"
    raw_csv_path = output_dir / f"{args.dataset}.data.csv"

    write_onehot_features(rows, feature_map, label_column, features_path)
    write_raw_copy(rows, raw_path)
    write_raw_copy(rows, raw_csv_path)

    label_values = []
    for row in rows:
        value = row[label_column]
        if value not in label_values:
            label_values.append(value)

    print(f"Input: {input_path}")
    print(f"Rows: {len(rows)}")
    print(f"Columns: {column_count}")
    print(f"Label column: {label_column}")
    print(f"Label classes: {len(label_values)} ({', '.join(label_values)})")
    print(f"Feature columns: {len(feature_map)}")
    print(f"Wrote features: {features_path}")
    print(f"Wrote labels source: {raw_path}")
    print(f"Wrote labels source CSV: {raw_csv_path}")


if __name__ == "__main__":
    main()
