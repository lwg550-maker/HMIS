from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sql_literal(value: object) -> str:
    if pd.isna(value):
        return "NULL"
    return "'" + str(value).replace("'", "''") + "'"


def build_values(risk: pd.DataFrame) -> str:
    rows: list[str] = []
    for r in risk.itertuples(index=False):
        rows.append(
            "    ("
            f"{int(r.subject_id)}::integer,{int(r.hadm_id)}::integer,{int(r.stay_id)}::integer,"
            f"{sql_literal(r.t_HMIS)}::timestamp,{int(r.landmark_gcsm)}::integer,"
            f"{sql_literal(r.target_window)}::text,{sql_literal(r.active_vent_start)}::timestamp,"
            f"{sql_literal(r.active_vent_end)}::timestamp)"
        )
    return ",\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--risk-audit", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    risk_all = pd.read_csv(args.risk_audit)
    truth = risk_all["risk_set"].astype(str).str.lower().isin({"true", "1"})
    risk = risk_all.loc[truth].copy()
    required = {
        "subject_id", "hadm_id", "stay_id", "t_HMIS", "landmark_gcsm",
        "target_window", "active_vent_start", "active_vent_end",
    }
    missing = required - set(risk.columns)
    if missing:
        raise ValueError(f"Gate A audit is missing columns: {sorted(missing)}")
    if len(risk) != 176 or risk.subject_id.nunique() != 176:
        raise ValueError(
            f"Gate C seed assertion failed: rows={len(risk)}, "
            f"unique_subjects={risk.subject_id.nunique()}, expected 176/176"
        )
    if risk[list(required)].isna().any().any():
        bad = risk[list(required)].isna().sum()
        raise ValueError(f"Gate C seed contains missing values: {bad[bad.gt(0)].to_dict()}")

    template = args.template.read_text(encoding="utf-8")
    if template.count("__SEED_VALUES__") != 1:
        raise ValueError("SQL template must contain exactly one __SEED_VALUES__ token")
    runtime = template.replace("__SEED_VALUES__", build_values(risk))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(runtime, encoding="utf-8", newline="\n")
    manifest = {
        "risk_audit": str(args.risk_audit),
        "risk_audit_sha256": sha256(args.risk_audit),
        "template": str(args.template),
        "template_sha256": sha256(args.template),
        "runtime_sql": str(args.output),
        "runtime_sql_sha256": sha256(args.output),
        "seed_rows": int(len(risk)),
        "seed_unique_subjects": int(risk.subject_id.nunique()),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Gate C private SQL built: seed={len(risk)}, output={args.output}")


if __name__ == "__main__":
    main()
