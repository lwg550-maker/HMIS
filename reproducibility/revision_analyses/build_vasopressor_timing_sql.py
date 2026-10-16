from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def quote_ts(value: object) -> str:
    stamp = pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S")
    return f"TIMESTAMP '{stamp}'"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = pd.read_csv(args.analysis)
    if len(data) != 176 or data.subject_id.nunique() != 176:
        raise AssertionError("expected frozen 176-person analysis dataset")
    values = []
    required = ["subject_id", "stay_id", "t_hmis"]
    if any(column not in data for column in required):
        raise KeyError(f"analysis dataset must contain {required}")
    for row in data[required].itertuples(index=False):
        values.append(f"({int(row.subject_id)}, {int(row.stay_id)}, {quote_ts(row.t_hmis)})")
    text = args.template.read_text(encoding="utf-8")
    args.output.write_text(text.replace("__LANDMARK_VALUES__", ",\n  ".join(values)), encoding="utf-8")


if __name__ == "__main__":
    main()
