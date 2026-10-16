from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


EXPECTED_OUTCOMES = {
    "SUSTAINED_LIBERATION": 131,
    "DEATH_BEFORE_LIBERATION": 40,
    "NO_LIBERATION_BY_28D": 5,
}
YEAR_MAP = {
    "2008 - 2010": 0,
    "2011 - 2013": 1,
    "2014 - 2016": 2,
    "2017 - 2019": 3,
    "2020 - 2022": 4,
}
PRIMARY_COLUMNS = [
    "age_10", "recorded_sex_male", "phenotype_sah", "phenotype_ich",
    "phenotype_sdh_edh", "log2_1_current_invasive_h", "landmark_gcsm_6",
    "pupil_bilateral_reactive", "major_sedative_active",
    "any_vasopressor_exposed_6h", "pf_per_50", "peep_per_5",
    "anchor_year_group_ordinal",
]
CONTINUOUS_COLUMNS = [
    "age_10", "log2_1_current_invasive_h", "pf_per_50", "peep_per_5",
    "anchor_year_group_ordinal",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_time(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    frame = frame.copy()
    for column in columns:
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame


def normalize_year_group(value: object) -> str | None:
    if pd.isna(value):
        return None
    return " ".join(str(value).replace("–", "-").split()).replace("-", " - ").replace("  ", " ")


def strict_destination_class(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip().upper()
    if text in {"HOME", "HOME HEALTH CARE", "REHAB"}:
        return "EXPLICIT_RETAIN_DESTINATION"
    if "ACUTE HOSPITAL" in text or "LONG TERM ACUTE" in text or "LTACH" in text:
        return "TRANSFER_CENSOR_DESTINATION"
    if text in {"HOSPICE", "SKILLED NURSING FACILITY"}:
        return "CONSERVATIVE_CENSOR_DESTINATION"
    return "UNKNOWN_OR_OTHER_DESTINATION"


def assert_gate_b(outcome: pd.DataFrame) -> None:
    counts = outcome["outcome"].value_counts().to_dict()
    if len(outcome) != 176 or outcome.subject_id.nunique() != 176:
        raise ValueError(
            f"Gate B risk-set assertion failed: {len(outcome)} rows, "
            f"{outcome.subject_id.nunique()} unique subjects"
        )
    for key, expected in EXPECTED_OUTCOMES.items():
        if int(counts.get(key, 0)) != expected:
            raise ValueError(f"Gate B outcome {key}: {counts.get(key, 0)} != {expected}")
    sources = outcome["confirmation_source"].value_counts().to_dict()
    if int(sources.get("FULL_168H_OBSERVATION", 0)) != 101:
        raise ValueError("Gate B full-observation event count is not 101")
    if int(sources.get("ALIVE_DISCHARGE_SURROGATE", 0)) != 30:
        raise ValueError("Gate B alive-discharge surrogate count is not 30")


def audit_discharge_surrogates(
    outcome: pd.DataFrame, candidates: pd.DataFrame, intervals: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    outcome = parse_time(
        outcome,
        ["event_time", "hospital_discharge", "hospital_death", "t_HMIS"],
    )
    candidates = parse_time(
        candidates,
        ["candidate_stop", "confirmation_end", "next_invasive_start", "hospital_discharge"],
    )
    intervals = parse_time(intervals, ["event_start", "event_end"])
    surrogate = outcome.loc[
        outcome.confirmation_source.eq("ALIVE_DISCHARGE_SURROGATE")
    ].copy()
    if len(surrogate) != 30:
        raise ValueError(f"Expected 30 discharge-surrogate events, found {len(surrogate)}")

    rows: list[dict[str, object]] = []
    for row in surrogate.itertuples(index=False):
        person_candidates = candidates.loc[candidates.subject_id.eq(row.subject_id)]
        chosen = person_candidates.loc[
            person_candidates.candidate_stop.eq(row.event_time)
            & person_candidates.candidate_result.eq("QUALIFIED_SUCCESS")
            & person_candidates.confirmation_source.eq("ALIVE_DISCHARGE_SURROGATE")
        ]
        if len(chosen) != 1:
            raise ValueError(
                f"Subject {row.subject_id}: expected one successful discharge candidate, found {len(chosen)}"
            )
        cand = chosen.iloc[0]
        person_intervals = intervals.loc[intervals.subject_id.eq(row.subject_id)]
        active_at_discharge = bool(
            (
                person_intervals.event_start.le(row.hospital_discharge)
                & person_intervals.event_end.gt(row.hospital_discharge)
            ).any()
        )
        reinvasion_before_discharge = bool(
            (
                person_intervals.event_start.gt(row.event_time)
                & person_intervals.event_start.le(row.hospital_discharge)
            ).any()
        )
        destination_class = strict_destination_class(row.discharge_location)
        evidence_clear = not active_at_discharge and not reinvasion_before_discharge
        if destination_class == "EXPLICIT_RETAIN_DESTINATION" and evidence_clear:
            strict_action = "RETAIN_SURROGATE_EVENT"
            strict_reason = "explicit destination and no recorded invasive support at discharge"
        else:
            strict_action = "CENSOR_AT_DISCHARGE"
            reasons = []
            if destination_class != "EXPLICIT_RETAIN_DESTINATION":
                reasons.append(destination_class.lower())
            if active_at_discharge:
                reasons.append("invasive_support_active_at_discharge")
            if reinvasion_before_discharge:
                reasons.append("reinvasion_between_stop_and_discharge")
            strict_reason = ";".join(reasons) or "ventilatory_status_not_confirmed"
        rows.append(
            {
                "subject_id": row.subject_id,
                "t_HMIS": row.t_HMIS,
                "candidate_stop": row.event_time,
                "confirmation_end": cand.confirmation_end,
                "hospital_discharge": row.hospital_discharge,
                "discharge_location": row.discharge_location,
                "destination_class": destination_class,
                "invasive_support_active_at_discharge": active_at_discharge,
                "reinvasion_between_stop_and_discharge": reinvasion_before_discharge,
                "strict_action": strict_action,
                "strict_reason": strict_reason,
                "stop_to_discharge_h": (
                    row.hospital_discharge - row.event_time
                ).total_seconds() / 3600.0,
            }
        )
    audit = pd.DataFrame(rows).sort_values("subject_id")
    summary = (
        audit.groupby(["discharge_location", "destination_class", "strict_action"], dropna=False)
        .size().rename("n").reset_index()
        .sort_values(["strict_action", "n"], ascending=[True, False])
    )
    return audit, summary


def prepare_analysis(raw: pd.DataFrame, outcome: pd.DataFrame) -> pd.DataFrame:
    if len(raw) != 176 or raw.subject_id.nunique() != 176:
        raise ValueError(
            f"Gate C raw extraction assertion failed: {len(raw)} rows, "
            f"{raw.subject_id.nunique()} unique subjects"
        )
    data = raw.merge(
        outcome[["subject_id", "outcome", "event_time", "followup_h", "confirmation_source"]],
        on="subject_id", how="left", validate="one_to_one",
    )
    if data.outcome.isna().any():
        raise ValueError("Gate C extraction does not match all Gate B subjects")
    data["age_10"] = pd.to_numeric(data.age, errors="coerce") / 10.0
    sex = data.recorded_sex.astype("string").str.upper()
    data["recorded_sex_male"] = sex.map({"M": 1.0, "F": 0.0})
    phenotype = data.diagnosis_phenotype.astype("string")
    data["phenotype_sah"] = phenotype.eq("SAH").astype(float)
    data["phenotype_ich"] = phenotype.eq("ICH").astype(float)
    data["phenotype_sdh_edh"] = phenotype.eq("SDH_EDH").astype(float)
    vent_h = pd.to_numeric(data.current_invasive_segment_h, errors="coerce")
    data["log2_1_current_invasive_h"] = np.log2(1.0 + vent_h.where(vent_h.ge(0)))
    data["landmark_gcsm_6"] = pd.to_numeric(data.landmark_gcsm, errors="coerce").map({5: 0.0, 6: 1.0})
    data["major_sedative_active"] = pd.to_numeric(data.major_sedative_active, errors="coerce")
    data["any_vasopressor_exposed_6h"] = pd.to_numeric(
        data.any_vasopressor_exposed_6h, errors="coerce"
    )
    data["pf_per_50"] = pd.to_numeric(data.pf_ratio, errors="coerce") / 50.0
    data["peep_per_5"] = pd.to_numeric(data.peep, errors="coerce") / 5.0
    normalized_year = data.anchor_year_group.map(normalize_year_group)
    data["anchor_year_group_ordinal"] = normalized_year.map(YEAR_MAP)
    data["anchor_year_group_normalized"] = normalized_year
    for column in PRIMARY_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data


def _fill_for_structural_audit(matrix: pd.DataFrame) -> pd.DataFrame:
    result = matrix.copy()
    for column in result:
        series = result[column]
        if column in CONTINUOUS_COLUMNS:
            fill = series.median()
        else:
            modes = series.mode(dropna=True)
            fill = modes.iloc[0] if not modes.empty else np.nan
        result[column] = series.fillna(fill)
    return result


def covariate_audits(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    variables = {
        "age": "years",
        "current_invasive_segment_h": "hours",
        "landmark_gcsm": "score",
        "pupil_bilateral_reactive": "binary",
        "major_sedative_active": "binary",
        "any_vasopressor_exposed_6h": "binary",
        "pf_ratio": "mmHg",
        "peep": "cmH2O",
        "anchor_year_group_ordinal": "ordinal_0_to_4",
        "auxiliary_fio2": "fraction",
        "auxiliary_resp_rate": "breaths_per_min",
        "auxiliary_rass": "score",
        "auxiliary_creatinine": "mg_per_dL",
        "auxiliary_wbc": "K_per_uL",
    }
    missing_rows = []
    range_rows = []
    for variable, unit in variables.items():
        series = pd.to_numeric(data[variable], errors="coerce")
        observed = series.dropna()
        missing_rows.append(
            {
                "variable": variable,
                "nonmissing_n": int(observed.size),
                "missing_n": int(series.isna().sum()),
                "missing_pct": float(100 * series.isna().mean()),
            }
        )
        range_rows.append(
            {
                "variable": variable,
                "unit": unit,
                "minimum": float(observed.min()) if len(observed) else np.nan,
                "q1": float(observed.quantile(0.25)) if len(observed) else np.nan,
                "median": float(observed.median()) if len(observed) else np.nan,
                "q3": float(observed.quantile(0.75)) if len(observed) else np.nan,
                "maximum": float(observed.max()) if len(observed) else np.nan,
            }
        )
    staleness_rows = []
    for variable in ["pupil_staleness_h", "pf_pair_staleness_h", "peep_staleness_h"]:
        series = pd.to_numeric(data[variable], errors="coerce").dropna()
        staleness_rows.append(
            {
                "variable": variable,
                "n": int(len(series)),
                "median_h": float(series.median()) if len(series) else np.nan,
                "q1_h": float(series.quantile(0.25)) if len(series) else np.nan,
                "q3_h": float(series.quantile(0.75)) if len(series) else np.nan,
                "max_h": float(series.max()) if len(series) else np.nan,
            }
        )

    matrix = data[PRIMARY_COLUMNS].copy()
    filled = _fill_for_structural_audit(matrix)
    if filled.isna().any().any():
        unfilled = filled.columns[filled.isna().any()].tolist()
        raise ValueError(f"Cannot audit structural rank; wholly missing columns: {unfilled}")
    x = filled.to_numpy(dtype=float)
    x_with_intercept = np.column_stack([np.ones(len(x)), x])
    rank = int(np.linalg.matrix_rank(x_with_intercept))
    expected_rank = int(x_with_intercept.shape[1])
    correlations = filled.corr(numeric_only=True).abs()
    near_pairs = []
    for i, left in enumerate(correlations.columns):
        for right in correlations.columns[i + 1 :]:
            value = correlations.loc[left, right]
            if pd.notna(value) and value >= 0.90:
                near_pairs.append({"left": left, "right": right, "abs_r": float(value)})

    pinv = np.linalg.pinv(x_with_intercept.T @ x_with_intercept)
    leverage = np.einsum("ij,jk,ik->i", x_with_intercept, pinv, x_with_intercept)
    leverage_threshold = 2.0 * expected_rank / len(data)
    high_leverage_n = int((leverage > leverage_threshold).sum())

    phenotype_counts = data.diagnosis_phenotype.value_counts(dropna=False).to_dict()
    sparse = {str(k): int(v) for k, v in phenotype_counts.items() if int(v) < 10}
    complete_case_n = int(matrix.notna().all(axis=1).sum())
    observed_pf = int(data.pf_ratio.notna().sum())
    observed_pupil = int(data.pupil_bilateral_reactive.notna().sum())
    abnormal_ranges = {
        "age_outside_18_120_n": int((~pd.to_numeric(data.age, errors="coerce").between(18, 120)).sum()),
        "negative_vent_duration_n": int((pd.to_numeric(data.current_invasive_segment_h, errors="coerce") < 0).sum()),
        "pf_outside_20_800_n": int((data.pf_ratio.notna() & ~pd.to_numeric(data.pf_ratio, errors="coerce").between(20, 800)).sum()),
        "peep_outside_0_40_n": int((data.peep.notna() & ~pd.to_numeric(data.peep, errors="coerce").between(0, 40)).sum()),
        "pf_pair_gap_over_120min_n": int((pd.to_numeric(data.pf_pair_gap_min, errors="coerce") > 120).sum()),
        "pupil_staleness_over_4h_n": int((pd.to_numeric(data.pupil_staleness_h, errors="coerce") > 4).sum()),
        "peep_staleness_over_6h_n": int((pd.to_numeric(data.peep_staleness_h, errors="coerce") > 6).sum()),
    }
    report = {
        "n": int(len(data)),
        "unique_subjects": int(data.subject_id.nunique()),
        "primary_predictor_df": 13,
        "matrix_columns": PRIMARY_COLUMNS,
        "matrix_rank_with_intercept": rank,
        "expected_rank_with_intercept": expected_rank,
        "structurally_full_rank": bool(rank == expected_rank),
        "complete_case_n": complete_case_n,
        "pupil_4h_nonmissing_n": observed_pupil,
        "pupil_4h_coverage_pct": 100.0 * observed_pupil / len(data),
        "pf_24h_nonmissing_n": observed_pf,
        "pf_24h_coverage_pct": 100.0 * observed_pf / len(data),
        "peep_6h_nonmissing_n": int(data.peep.notna().sum()),
        "peep_6h_coverage_pct": 100.0 * data.peep.notna().mean(),
        "diagnosis_phenotype_counts": {str(k): int(v) for k, v in phenotype_counts.items()},
        "sparse_phenotype_levels_lt10": sparse,
        "near_collinear_pairs_abs_r_ge_0_90": near_pairs,
        "potential_high_leverage_n": high_leverage_n,
        "potential_high_leverage_threshold": leverage_threshold,
        "abnormal_or_contract_violation_counts": abnormal_ranges,
        "mi_auxiliary_nonmissing": {
            key: int(data[key].notna().sum())
            for key in [
                "auxiliary_fio2", "auxiliary_resp_rate", "auxiliary_rass",
                "auxiliary_creatinine", "auxiliary_wbc",
            ]
        },
        "manual_review_triggers": [],
    }
    if rank != expected_rank:
        report["manual_review_triggers"].append("13-df design matrix is structurally rank deficient")
    if sparse:
        report["manual_review_triggers"].append("one or more diagnosis phenotype levels have fewer than 10 patients")
    if observed_pupil < len(data):
        report["manual_review_triggers"].append(
            "four-hour pupil coverage must be compared with prior audit; no window relaxation is permitted"
        )
    if observed_pf < len(data):
        report["manual_review_triggers"].append(
            "P/F is incomplete and requires MI feasibility review plus the prespecified FiO2 replacement analysis"
        )
    if any(abnormal_ranges.values()):
        report["manual_review_triggers"].append("one or more temporal/value contracts were violated")
    return (
        pd.DataFrame(missing_rows),
        pd.DataFrame(range_rows),
        pd.DataFrame(staleness_rows),
        report,
    )


def medication_summary(data: pd.DataFrame) -> pd.DataFrame:
    fields = [
        "propofol_active", "dexmedetomidine_active", "midazolam_active",
        "lorazepam_active", "ketamine_active", "major_sedative_active",
        "any_vasopressor_exposed_6h",
    ]
    rows = []
    for field in fields:
        values = pd.to_numeric(data[field], errors="coerce")
        rows.append({"variable": field, "n": int(values.eq(1).sum()), "denominator": int(values.notna().sum())})
    overlap = pd.to_numeric(data.active_sedative_drug_n, errors="coerce")
    for n, count in overlap.value_counts(dropna=False).sort_index().items():
        rows.append({"variable": f"active_sedative_drug_n={n}", "n": int(count), "denominator": int(len(data))})
    return pd.DataFrame(rows)


@dataclass
class GateCResult:
    discharge_audit: pd.DataFrame
    discharge_summary: pd.DataFrame
    analysis_data: pd.DataFrame
    missingness: pd.DataFrame
    ranges: pd.DataFrame
    staleness: pd.DataFrame
    medication: pd.DataFrame
    report: dict


def run_gate_c(raw: pd.DataFrame, outcome: pd.DataFrame, candidates: pd.DataFrame, intervals: pd.DataFrame) -> GateCResult:
    assert_gate_b(outcome)
    discharge_audit, discharge_summary = audit_discharge_surrogates(outcome, candidates, intervals)
    analysis = prepare_analysis(raw, outcome)
    missingness, ranges, staleness, report = covariate_audits(analysis)
    medication = medication_summary(analysis)
    strict_counts = discharge_audit.strict_action.value_counts().to_dict()
    report["discharge_surrogate_n"] = int(len(discharge_audit))
    report["strict_discharge_actions"] = {str(k): int(v) for k, v in strict_counts.items()}
    report["gate_c_status"] = "REQUIRES_AUTHOR_REVIEW_AFTER_EXTRACTION"
    return GateCResult(
        discharge_audit, discharge_summary, analysis, missingness, ranges,
        staleness, medication, report,
    )


def environment_receipt(inputs: dict[str, Path], random_seed: int = 20261012) -> dict:
    versions = {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__}
    package_root = Path(__file__).resolve().parents[1]
    code_files = sorted(
        path for path in package_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".py", ".sql", ".ps1"}
        and "restricted" not in path.parts
    )
    return {
        "random_seed": random_seed,
        "software": versions,
        "code_sha256": {
            str(path.relative_to(package_root)): sha256(path) for path in code_files
        },
        "inputs": {
            name: {"path": str(path), "sha256": sha256(path)} for name, path in inputs.items()
        },
    }


def write_result(result: GateCResult, restricted: Path, aggregate: Path, receipt: dict) -> None:
    restricted.mkdir(parents=True, exist_ok=True)
    aggregate.mkdir(parents=True, exist_ok=True)
    result.discharge_audit.to_csv(restricted / "gate_c_discharge_surrogate_audit_internal.csv", index=False)
    result.analysis_data.to_csv(restricted / "gate_c_analysis_dataset_internal.csv", index=False)
    result.discharge_summary.to_csv(aggregate / "gate_c_discharge_surrogate_summary.csv", index=False)
    result.missingness.to_csv(aggregate / "gate_c_missingness.csv", index=False)
    result.ranges.to_csv(aggregate / "gate_c_value_ranges.csv", index=False)
    result.staleness.to_csv(aggregate / "gate_c_measurement_staleness.csv", index=False)
    result.medication.to_csv(aggregate / "gate_c_medication_summary.csv", index=False)
    (aggregate / "gate_c_qc.json").write_text(json.dumps(result.report, indent=2), encoding="utf-8")
    receipt = {**receipt, "output_rows": int(len(result.analysis_data)), "gate_c_report": result.report}
    (aggregate / "run_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
