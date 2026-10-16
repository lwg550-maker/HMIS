from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy


BASE = Path(r"F:\EVD\neuroresp-pilot\post-rejection_reframing_20261006")
GATE_AB = BASE / "execution/hmis_liberation_gates_ab_20261012_013840"
GATE_B = BASE / "execution/hmis_liberation_gate_b_evidence_20261012_161516"
GATE_C = BASE / "execution/hmis_liberation_gate_c_20261012_165736"
GATE_E = BASE / "execution/hmis_liberation_gate_e_20261012_213429"
GATE_F = BASE / "execution/hmis_liberation_gate_f_20261013_004523"

MODEL_COLUMNS = [
    "age_10", "recorded_sex_male", "phenotype_sah", "phenotype_ich",
    "phenotype_sdh_edh", "log2_1_current_invasive_h", "landmark_gcsm_6",
    "pupil_bilateral_reactive", "major_sedative_active",
    "any_vasopressor_exposed_6h", "pf_per_50", "peep_per_5",
    "anchor_year_group_ordinal",
]
VASOPRESSOR = "any_vasopressor_exposed_6h"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


gate_e = load_module("gate_e_core_revision", GATE_E / "src/gate_e.py")
sys.modules["gate_e_core"] = gate_e
gate_f = load_module("gate_f_core_revision", GATE_F / "src/gate_f.py")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_time(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column in result:
            result[column] = pd.to_datetime(result[column], errors="coerce")
    return result


def derive_confirmation_time_outcome(
    risk: pd.DataFrame, candidates: pd.DataFrame
) -> pd.DataFrame:
    """Derive a day-28 observable confirmation-time endpoint.

    Only candidate cessations starting on/before day 21 are eligible. Full
    observation is timed at stop+168 h; an eligible alive-discharge surrogate
    is timed at discharge. Death at or before an otherwise coincident
    confirmation takes precedence. No post-day-28 datum is used.
    """
    risk = parse_time(risk, ["t_HMIS", "death_time", "hospital_death", "hospital_discharge"])
    candidates = parse_time(
        candidates,
        ["t_HMIS", "candidate_stop", "confirmation_end", "death_time", "hospital_discharge"],
    )
    if len(risk) != 176 or risk.subject_id.nunique() != 176:
        raise AssertionError("confirmation-time analysis requires the frozen 176-person risk set")
    rows: list[dict[str, object]] = []
    grouped = {int(k): g.sort_values("candidate_sequence") for k, g in candidates.groupby("subject_id")}
    for record in risk.itertuples(index=False):
        subject_id = int(record.subject_id)
        anchor = pd.Timestamp(record.t_HMIS)
        day21 = anchor + pd.Timedelta(days=21)
        day28 = anchor + pd.Timedelta(days=28)
        death = pd.to_datetime(getattr(record, "hospital_death", pd.NaT), errors="coerce")
        if pd.isna(death):
            death = pd.to_datetime(getattr(record, "death_time", pd.NaT), errors="coerce")
        eligible_events: list[tuple[pd.Timestamp, str, pd.Timestamp]] = []
        for cand in grouped.get(subject_id, pd.DataFrame()).itertuples(index=False):
            stop = pd.Timestamp(cand.candidate_stop)
            if not (anchor < stop <= day21):
                continue
            if cand.candidate_result != "QUALIFIED_SUCCESS":
                continue
            if cand.confirmation_source == "FULL_168H_OBSERVATION":
                confirmation = pd.Timestamp(cand.confirmation_end)
            elif cand.confirmation_source == "ALIVE_DISCHARGE_SURROGATE":
                confirmation = pd.Timestamp(cand.hospital_discharge)
                if not stop < confirmation:
                    raise AssertionError("alive-discharge surrogate was not after candidate cessation")
            else:
                continue
            if confirmation <= day28:
                eligible_events.append((confirmation, str(cand.confirmation_source), stop))
        event = min(eligible_events, default=None, key=lambda value: value[0])
        if pd.notna(death) and anchor < death <= day28 and (event is None or death <= event[0]):
            outcome = "DEATH_BEFORE_LIBERATION"
            event_time = death
            source = "NONE"
            stop_time = pd.NaT
        elif event is not None:
            outcome = "SUSTAINED_LIBERATION"
            event_time, source, stop_time = event
        else:
            outcome = "NO_LIBERATION_BY_28D"
            event_time = day28
            source = "NONE"
            stop_time = pd.NaT
        followup_h = (event_time - anchor).total_seconds() / 3600.0
        if not (0 < followup_h <= 672.0 + 1e-9):
            raise AssertionError("confirmation-time follow-up outside (0, 672 h]")
        rows.append({
            "subject_id": subject_id,
            "outcome": outcome,
            "event_time": event_time,
            "followup_h": followup_h,
            "confirmation_source": source,
            "qualifying_candidate_stop": stop_time,
        })
    output = pd.DataFrame(rows)
    if len(output) != 176 or output.subject_id.nunique() != 176:
        raise AssertionError("confirmation-time outcome does not exhaust frozen risk set")
    return output


def pool_table(pool: dict, parameters: list[str], suffix: str = "") -> pd.DataFrame:
    table = gate_f._pooled_estimate_table(pool, parameters)
    if suffix:
        table = table.rename(columns={c: f"{c}_{suffix}" for c in table.columns if c != "parameter"})
    return table


def fit_mi_cox_fg(
    data: pd.DataFrame,
    outcome: pd.DataFrame,
    *,
    imputations: int = 50,
    iterations: int = 20,
    seed: int = 20261012,
    peep_time_interaction: bool = False,
) -> dict[str, object]:
    frame, durations, codes = gate_f.build_imputation_frame(data, outcome, MODEL_COLUMNS)
    completed = gate_e.mice_pmm_impute(frame, imputations, iterations, seed, k_pmm=5)
    cox_est, cox_cov, fg_est, fg_cov = [], [], [], []
    fg_scaled_residuals: list[np.ndarray] = []
    ext_est, ext_cov, ext_grad = [], [], []
    fine_time, fine_status = gate_e.fine_gray_pseudo_time(durations / 24.0, codes, 28.0)
    for complete in completed:
        matrix = complete.loc[:, MODEL_COLUMNS].to_numpy(float)
        cox = gate_e.fit_phreg(durations, (codes == 1).astype(int), matrix, robust=False)
        fg = gate_e.fit_phreg(fine_time, fine_status, matrix, robust=True)
        cox_est.append(cox["params"]); cox_cov.append(cox["covariance"])
        fg_est.append(fg["params"]); fg_cov.append(fg["covariance"])
        fg_residuals = np.asarray(fg["result"].schoenfeld_residuals)[fine_status == 1]
        fg_scaled_residuals.append(fg_residuals @ fg["covariance"] * int(fine_status.sum()))
        if peep_time_interaction:
            extended = gate_e.fit_extended_cox(
                durations,
                (codes == 1).astype(int),
                matrix,
                time_varying_index=MODEL_COLUMNS.index("peep_per_5"),
                initial=np.r_[cox["params"], 0.0],
            )
            ext_est.append(extended["params"]); ext_cov.append(extended["covariance"])
            ext_grad.append(extended["gradient_max_abs"])
    cox_pool = gate_e.rubin_pool(np.asarray(cox_est), np.asarray(cox_cov))
    fg_pool = gate_e.rubin_pool(np.asarray(fg_est), np.asarray(fg_cov))
    cox_table = pool_table(cox_pool, MODEL_COLUMNS, "cox")
    fg_table = pool_table(fg_pool, MODEL_COLUMNS, "finegray")
    merged = cox_table.merge(fg_table, on="parameter", validate="one_to_one")
    fg_ph = gate_e.schoenfeld_time_test(
        fine_time[fine_status == 1], np.mean(np.stack(fg_scaled_residuals), axis=0)
    )
    result: dict[str, object] = {
        "table": merged,
        "diagnostics": {
            "n": int(len(outcome)),
            "event_n": int((codes == 1).sum()),
            "death_n": int((codes == 2).sum()),
            "censored_n": int((codes == 0).sum()),
            "finegray_ph_method": "rank-time test of mean scaled Schoenfeld residuals from the pseudo-time PH representation across imputations",
            "finegray_ph_global_p": float(fg_ph["global_p"]),
            "finegray_ph_vasopressor_p": float(fg_ph["per_variable_p"][MODEL_COLUMNS.index(VASOPRESSOR)]),
        },
    }
    if peep_time_interaction:
        parameters = [*MODEL_COLUMNS, "peep_by_log1p_time"]
        ext_pool = gate_e.rubin_pool(np.asarray(ext_est), np.asarray(ext_cov))
        result["extended_table"] = pool_table(ext_pool, parameters)
        result["extended_max_gradient"] = float(max(ext_grad))
    return result


def checkpoint_selection_audit(
    checkpoint: pd.DataFrame,
    raw_events: pd.DataFrame,
    landmark_subjects: set[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cp = checkpoint.copy()
    cp["subject_id"] = pd.to_numeric(cp.subject_id, errors="raise").astype(int)
    cp["candidate_record_n"] = pd.to_numeric(cp.candidate_record_n, errors="coerce").fillna(0).astype(int)
    for c in ["same_timestamp_unresolved_conflict", "nmb_active_at_checkpoint"]:
        cp[c] = cp[c].astype(str).str.lower().eq("true")
    nontransition = sorted(set(cp.subject_id) - landmark_subjects)
    if len(nontransition) != 253:
        raise AssertionError(f"expected 253 nontransition patients, got {len(nontransition)}")
    raw = raw_events.loc[raw_events.event_type.eq("GCS_M")].copy()
    raw["subject_id"] = pd.to_numeric(raw.subject_id, errors="raise").astype(int)
    raw["value_num"] = pd.to_numeric(raw.value_num, errors="coerce")
    raw["charttime"] = pd.to_datetime(raw.charttime, errors="coerce")
    raw["t0"] = pd.to_datetime(raw.t0, errors="coerce")
    raw = raw.loc[raw.charttime.gt(raw.t0 - pd.Timedelta(hours=4)) & raw.charttime.le(raw.t0 + pd.Timedelta(hours=48))]
    counts = raw.groupby("subject_id").size()
    rows: list[dict[str, object]] = []
    for sid in nontransition:
        g = cp.loc[cp.subject_id.eq(sid)].sort_values("window_order")
        states = g.motor_state.astype(str).tolist()
        evaluable = g.motor_state.isin(["LOW", "HIGH"])
        high_any = g.motor_state.eq("HIGH").any()
        raw_high_any = raw.loc[raw.subject_id.eq(sid), "value_num"].isin([5, 6]).any()
        conflict = g.same_timestamp_unresolved_conflict.any()
        nmb = g.nmb_active_at_checkpoint.any()
        recordless = g.candidate_record_n.eq(0).sum()
        adjacent_evaluable = any(
            states[i] in {"LOW", "HIGH"} and states[i + 1] in {"LOW", "HIGH"}
            for i in range(len(states) - 1)
        )
        unknown_bridge = any(
            states[i] == "LOW" and states[j] == "HIGH" and any(x == "UNKNOWN" for x in states[i + 1:j])
            for i in range(len(states)) for j in range(i + 2, len(states))
        )
        if not raw_high_any:
            reason = "Never recorded GCS-M 5-6 in the (-4,48] h checkpoint observation span"
        elif conflict:
            reason = "Unresolved same-time GCS-M conflict"
        elif nmb:
            reason = "Continuous neuromuscular blockade made one or more checkpoints unevaluable"
        elif unknown_bridge:
            reason = "Unknown intermediate checkpoint prevented a nonadjacent low-to-high bridge"
        elif not adjacent_evaluable:
            reason = "No evaluable adjacent checkpoint pair"
        elif recordless >= 2 or int(counts.get(sid, 0)) < 2:
            reason = "Insufficient GCS-M recording frequency for the checkpoint algorithm"
        elif not high_any:
            reason = "Recorded GCS-M 5-6 did not populate an evaluable checkpoint"
        else:
            reason = "Other observed sequence without an eligible adjacent low-to-high transition"
        rows.append({
            "subject_id": sid,
            "reason": reason,
            "checkpoint_evaluable_n": int(evaluable.sum()),
            "raw_gcs_m_record_n": int(counts.get(sid, 0)),
        })
    patient = pd.DataFrame(rows)
    all_reasons = [
        "Never recorded GCS-M 5-6 in the (-4,48] h checkpoint observation span",
        "No evaluable adjacent checkpoint pair",
        "Unknown intermediate checkpoint prevented a nonadjacent low-to-high bridge",
        "Continuous neuromuscular blockade made one or more checkpoints unevaluable",
        "Unresolved same-time GCS-M conflict",
        "Insufficient GCS-M recording frequency for the checkpoint algorithm",
        "Recorded GCS-M 5-6 did not populate an evaluable checkpoint",
        "Other observed sequence without an eligible adjacent low-to-high transition",
    ]
    summary = patient.groupby("reason", as_index=False).size().rename(columns={"size": "n"})
    summary = pd.DataFrame({"reason": all_reasons}).merge(summary, on="reason", how="left")
    summary["n"] = summary.n.fillna(0).astype(int)
    summary["pct_of_253"] = summary.n / 253 * 100
    eval_rows = []
    for window, g in cp.groupby("window_id", sort=False):
        eval_rows.append({
            "checkpoint": window,
            "n": int(len(g)),
            "evaluable_n": int(g.motor_state.isin(["LOW", "HIGH"]).sum()),
            "evaluable_pct": float(g.motor_state.isin(["LOW", "HIGH"]).mean() * 100),
            "unknown_n": int(g.motor_state.eq("UNKNOWN").sum()),
        })
    evaluability = pd.DataFrame(eval_rows)
    frequency = pd.DataFrame({"subject_id": sorted(set(cp.subject_id))})
    frequency["gcs_m_record_n"] = frequency.subject_id.map(counts).fillna(0).astype(int)
    return patient, summary, evaluability, frequency


def unsampled_gap_audit(raw_events: pd.DataFrame, nontransition_subjects: set[int]) -> pd.DataFrame:
    raw = raw_events.loc[raw_events.event_type.eq("GCS_M")].copy()
    raw["subject_id"] = pd.to_numeric(raw.subject_id, errors="raise").astype(int)
    raw["charttime"] = pd.to_datetime(raw.charttime, errors="coerce")
    raw["t0"] = pd.to_datetime(raw.t0, errors="coerce")
    raw["value_num"] = pd.to_numeric(raw.value_num, errors="coerce")
    gaps = {
        "W3-to-W4 unsampled interval (12,20] h": (12, 20),
        "W4-to-W5 unsampled interval (24,44] h": (24, 44),
    }
    rows = []
    for label, (left, right) in gaps.items():
        start = raw.t0 + pd.to_timedelta(left, unit="h")
        end = raw.t0 + pd.to_timedelta(right, unit="h")
        g = raw.loc[raw.charttime.gt(start) & raw.charttime.le(end)]
        nt = g.loc[g.subject_id.isin(nontransition_subjects)]
        rows.append({
            "interval": label,
            "record_n_all_474": int(len(g)),
            "patient_n_all_474": int(g.subject_id.nunique()),
            "record_n_nontransition_253": int(len(nt)),
            "patient_n_nontransition_253": int(nt.subject_id.nunique()),
            "nontransition_patients_with_recorded_gcsm_5_6_n": int(nt.loc[nt.value_num.isin([5, 6]), "subject_id"].nunique()),
        })
    return pd.DataFrame(rows)


def post_landmark_construct_audit(
    analysis: pd.DataFrame, raw_events: pd.DataFrame, cohort: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, object]]:
    raw = raw_events.loc[raw_events.event_type.eq("GCS_M")].copy()
    raw["charttime"] = pd.to_datetime(raw.charttime, errors="coerce")
    raw["value_num"] = pd.to_numeric(raw.value_num, errors="coerce")
    analysis = analysis.copy()
    if "t_hmis" not in analysis:
        raise KeyError("Gate C analysis dataset is missing t_hmis")
    analysis["t_hmis"] = pd.to_datetime(analysis.t_hmis, errors="coerce")
    rows = []
    for r in analysis.itertuples(index=False):
        later = raw.loc[
            raw.subject_id.eq(r.subject_id)
            & raw.charttime.gt(r.t_hmis)
            & raw.value_num.isin([5, 6])
        ].sort_values("charttime")
        first = later.iloc[0].charttime if len(later) else pd.NaT
        rows.append({
            "subject_id": int(r.subject_id),
            "repeated_gcsm_5_6_after_landmark": bool(len(later)),
            "first_repeat_interval_h": ((first - r.t_hmis).total_seconds() / 3600.0) if pd.notna(first) else np.nan,
            "major_sedative_active_at_landmark": int(r.major_sedative_active),
        })
    internal = pd.DataFrame(rows)
    intervals = internal.first_repeat_interval_h.dropna()
    status_counts = cohort.final_status.fillna("MISSING").value_counts().to_dict()
    summary = {
        "risk_set_n": int(len(internal)),
        "repeat_gcsm_5_6_n": int(internal.repeated_gcsm_5_6_after_landmark.sum()),
        "repeat_gcsm_5_6_pct": float(internal.repeated_gcsm_5_6_after_landmark.mean() * 100),
        "repeat_interval_h_median": float(intervals.median()) if len(intervals) else None,
        "repeat_interval_h_q1": float(intervals.quantile(0.25)) if len(intervals) else None,
        "repeat_interval_h_q3": float(intervals.quantile(0.75)) if len(intervals) else None,
        "sedative_active_n": int(internal.major_sedative_active_at_landmark.sum()),
        "sedative_active_pct": float(internal.major_sedative_active_at_landmark.mean() * 100),
        "t0_record_type_counts_in_474": {str(k): int(v) for k, v in status_counts.items()},
    }
    return internal, summary


def vasopressor_descriptive(analysis: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    drug_columns = {
        "Norepinephrine (itemid 221906)": "norepinephrine_exposed_6h",
        "Epinephrine (itemid 221289)": "epinephrine_exposed_6h",
        "Phenylephrine (itemid 221749)": "phenylephrine_exposed_6h",
        "Vasopressin (itemid 222315)": "vasopressin_exposed_6h",
        "Dopamine (itemid 221662)": "dopamine_exposed_6h",
    }
    drug_rows = []
    flags = analysis.loc[:, list(drug_columns.values())].astype(int)
    for label, column in drug_columns.items():
        drug_rows.append({"drug": label, "n": int(analysis[column].sum()), "pct": float(analysis[column].mean() * 100)})
    drug_rows.append({"drug": "Two or more vasopressors", "n": int(flags.sum(axis=1).ge(2).sum()), "pct": float(flags.sum(axis=1).ge(2).mean() * 100)})
    strata_rows = []
    for exposure, g in analysis.groupby(VASOPRESSOR):
        strata_rows.append({
            "vasopressor_exposed": int(exposure),
            "n": int(len(g)),
            "primary_rule_event_n": int(g.outcome.eq("SUSTAINED_LIBERATION").sum()),
            "competing_death_n": int(g.outcome.eq("DEATH_BEFORE_LIBERATION").sum()),
            "censored_n": int(g.outcome.eq("NO_LIBERATION_BY_28D").sum()),
            "full_168h_n": int(g.confirmation_source.eq("FULL_168H_OBSERVATION").sum()),
            "alive_discharge_surrogate_n": int(g.confirmation_source.eq("ALIVE_DISCHARGE_SURROGATE").sum()),
        })
    return pd.DataFrame(drug_rows), pd.DataFrame(strata_rows)


def vasopressor_timing_summary(path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    data = pd.read_csv(path)
    if len(data) == 0:
        raise AssertionError("vasopressor timing extraction is empty")
    if data.subject_id.nunique() != 36:
        raise AssertionError("vasopressor timing extraction does not reproduce 36 exposed patients")
    for column in ["hours_from_infusion_start_to_landmark", "hours_from_landmark_to_infusion_end"]:
        data[column] = pd.to_numeric(data[column], errors="raise")
    per_patient_drug = (
        data.groupby(["subject_id", "drug_name"], as_index=False)
        .agg(
            most_recent_start_h_before_landmark=("hours_from_infusion_start_to_landmark", "min"),
            active_at_landmark=("hours_from_landmark_to_infusion_end", lambda x: bool((x > 0).any())),
        )
    )
    rows = []
    for drug, g in per_patient_drug.groupby("drug_name"):
        values = g.most_recent_start_h_before_landmark
        rows.append({
            "drug": drug,
            "patient_n": int(len(g)),
            "most_recent_start_h_before_landmark_median": float(values.median()),
            "most_recent_start_h_before_landmark_q1": float(values.quantile(.25)),
            "most_recent_start_h_before_landmark_q3": float(values.quantile(.75)),
            "active_at_landmark_n": int(g.active_at_landmark.sum()),
        })
    unit_counts = (
        data.groupby(["drug_name", "rateuom"], dropna=False).size().reset_index(name="infusion_record_n")
    )
    meta = {
        "qualifying_infusion_rows": int(len(data)),
        "exposed_patients": int(data.subject_id.nunique()),
        "rate_unit_counts": unit_counts.to_dict(orient="records"),
        "dose_conversion_decision": "not performed; vasopressin uses units/hour whereas catecholamines use mcg/kg/min, and no prespecified equivalence rule was available",
    }
    return pd.DataFrame(rows), meta


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-models", action="store_true")
    args = parser.parse_args()
    for name in ["aggregate", "restricted", "tables", "logs"]:
        (args.output_dir / name).mkdir(parents=True, exist_ok=True)

    paths = {
        "analysis": GATE_C / "restricted/gate_c_analysis_dataset_internal.csv",
        "primary": GATE_B / "restricted/evidence_aware_outcome_7d_internal.csv",
        "candidates": GATE_B / "restricted/evidence_aware_candidates_7d_internal.csv",
        "historical": GATE_B / "restricted/historical_outcome_7d_internal.csv",
        "all_trach": GATE_B / "restricted/all_trach_upper_outcome_7d_internal.csv",
        "checkpoint": GATE_AB / "restricted/gate_a_checkpoint_audit_internal.csv",
        "landmark": GATE_AB / "restricted/gate_a_landmark_patient_audit_internal.csv",
        "raw_events": GATE_AB / "restricted/01_gate_ab_events_20261012_013840.csv",
        "cohort": Path(r"F:\EVD\neuroresp-pilot\01_extract_cohort.csv"),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    analysis = pd.read_csv(paths["analysis"])
    primary = pd.read_csv(paths["primary"])
    candidates = pd.read_csv(paths["candidates"])
    checkpoint = pd.read_csv(paths["checkpoint"])
    landmarks = pd.read_csv(paths["landmark"])
    raw_events = pd.read_csv(paths["raw_events"], low_memory=False)
    cohort = pd.read_csv(paths["cohort"])
    if len(analysis) != 176 or analysis.subject_id.nunique() != 176:
        raise AssertionError("Gate C risk set drift")
    if primary.outcome.value_counts().to_dict() != {
        "SUSTAINED_LIBERATION": 131,
        "DEATH_BEFORE_LIBERATION": 40,
        "NO_LIBERATION_BY_28D": 5,
    }:
        raise AssertionError("primary 131/40/5 identity drift")

    confirmation = derive_confirmation_time_outcome(primary, candidates)
    confirmation.to_csv(args.output_dir / "restricted/confirmation_time_outcome_internal.csv", index=False)
    confirmation_counts = {str(k): int(v) for k, v in confirmation.outcome.value_counts().items()}
    write_json(args.output_dir / "aggregate/confirmation_time_counts.json", confirmation_counts)

    landmark_subjects = set(pd.to_numeric(landmarks.subject_id, errors="raise").astype(int))
    patient_reason, reason_summary, evaluability, frequency = checkpoint_selection_audit(
        checkpoint, raw_events, landmark_subjects
    )
    patient_reason.to_csv(args.output_dir / "restricted/nontransition_reason_internal.csv", index=False)
    reason_summary.to_csv(args.output_dir / "tables/nontransition_reason_summary.csv", index=False)
    evaluability.to_csv(args.output_dir / "tables/checkpoint_evaluability.csv", index=False)
    frequency.drop(columns="subject_id").describe(percentiles=[.25, .5, .75]).to_csv(
        args.output_dir / "tables/gcsm_record_frequency_summary.csv"
    )
    gap_audit = unsampled_gap_audit(raw_events, set(patient_reason.subject_id))
    gap_audit.to_csv(args.output_dir / "tables/unsampled_checkpoint_gap_audit.csv", index=False)

    construct_internal, construct_summary = post_landmark_construct_audit(analysis, raw_events, cohort)
    construct_internal.to_csv(args.output_dir / "restricted/post_landmark_construct_internal.csv", index=False)
    write_json(args.output_dir / "aggregate/post_landmark_construct_summary.json", construct_summary)
    construct_strata = construct_internal.merge(
        analysis[["subject_id", "outcome"]], on="subject_id", how="left", validate="one_to_one"
    )
    cohort_status = cohort[["subject_id", "final_status"]].drop_duplicates("subject_id")
    construct_strata = construct_strata.merge(cohort_status, on="subject_id", how="left", validate="one_to_one")
    strata_rows = []
    for variable, values in {
        "repeated_gcsm_5_6_after_landmark": construct_strata.repeated_gcsm_5_6_after_landmark.astype(str),
        "major_sedative_active_at_landmark": construct_strata.major_sedative_active_at_landmark.astype(str),
        "t0_record_type": construct_strata.final_status.fillna("MISSING").astype(str),
    }.items():
        for level in sorted(values.unique()):
            g = construct_strata.loc[values.eq(level)]
            strata_rows.append({
                "audit_variable": variable, "level": level, "n": int(len(g)),
                "primary_rule_event_n": int(g.outcome.eq("SUSTAINED_LIBERATION").sum()),
                "competing_death_n": int(g.outcome.eq("DEATH_BEFORE_LIBERATION").sum()),
                "censored_n": int(g.outcome.eq("NO_LIBERATION_BY_28D").sum()),
            })
    pd.DataFrame(strata_rows).to_csv(args.output_dir / "tables/landmark_construct_strata.csv", index=False)

    drug, strata = vasopressor_descriptive(analysis)
    drug.to_csv(args.output_dir / "tables/vasopressor_drug_counts.csv", index=False)
    strata.to_csv(args.output_dir / "tables/vasopressor_outcome_strata.csv", index=False)
    timing_path = args.output_dir / "restricted/vasopressor_timing_internal.csv"
    if timing_path.exists():
        timing, timing_meta = vasopressor_timing_summary(timing_path)
        timing.to_csv(args.output_dir / "tables/vasopressor_timing_summary.csv", index=False)
        write_json(args.output_dir / "aggregate/vasopressor_timing_qc.json", timing_meta)

    if not args.skip_models:
        confirmation_fit = fit_mi_cox_fg(analysis, confirmation)
        confirmation_fit["table"].to_csv(args.output_dir / "tables/confirmation_time_models.csv", index=False)
        write_json(args.output_dir / "aggregate/confirmation_time_model_diagnostics.json", confirmation_fit["diagnostics"])

        primary_fit = fit_mi_cox_fg(analysis, primary, peep_time_interaction=True)
        primary_fit["extended_table"].to_csv(args.output_dir / "tables/peep_time_extended_model.csv", index=False)
        write_json(args.output_dir / "aggregate/primary_finegray_time_stability.json", primary_fit["diagnostics"])

        sensitivity_rows = []
        variant_paths = {
            "unaugmented_ventilation_timeline": paths["historical"],
            "all_tracheostomy_positive_pressure_boundary": paths["all_trach"],
        }
        for label, path in variant_paths.items():
            outcome = pd.read_csv(path)
            result = gate_f.fit_mi_cause_specific(
                analysis, outcome, MODEL_COLUMNS,
                imputations=50, iterations=20, seed=20261012,
            )
            estimate = result["estimates"].loc[result["estimates"].parameter.eq(VASOPRESSOR)].iloc[0]
            d = result["diagnostics"]
            sensitivity_rows.append({
                "specification": label,
                "n": d["n"], "event_n": d["target_events"],
                "death_n": d["competing_deaths"], "censored_n": d["ordinary_or_administrative_censoring"],
                "vasopressor_csHR": estimate.hazard_ratio,
                "ci_low": estimate.ci_low, "ci_high": estimate.ci_high,
                "p_value": estimate.p_value,
            })
        existing = pd.read_csv(GATE_F / "tables/Supplementary_Table_GateF_all_models.csv")
        existing_counts = pd.read_csv(GATE_F / "aggregate/gate_f_outcome_counts.csv").set_index("analysis")
        for label in ["STRICT_DISCHARGE", "NO_DISCHARGE_SURROGATE", "SUSTAINED_48H"]:
            sub = existing.loc[existing.analysis.eq(label)]
            row = sub.loc[sub.parameter.eq(VASOPRESSOR)].iloc[0]
            counts = existing_counts.loc[label]
            sensitivity_rows.append({
                "specification": label,
                "n": int(counts.n), "event_n": int(counts.sustained_liberation_n),
                "death_n": int(counts.death_before_liberation_n),
                "censored_n": int(counts.no_liberation_by_28d_n) + int(counts.early_censor_without_confirmation_n),
                "vasopressor_csHR": row.hazard_ratio,
                "ci_low": row.ci_low, "ci_high": row.ci_high,
                "p_value": row.p_value,
            })
        pd.DataFrame(sensitivity_rows).to_csv(args.output_dir / "tables/outcome_evidence_model_sensitivity.csv", index=False)

    receipt = {
        "seed": 20261012,
        "imputations": 50,
        "iterations": 20,
        "software": {
            "python": sys.version, "platform": platform.platform(),
            "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
        },
        "inputs": {name: {"path": str(path), "sha256": sha256(path)} for name, path in paths.items()},
        "frozen_assertions": {"source_cohort": 474, "landmark_candidates": 221, "risk_set": 176, "primary_outcomes": "131/40/5"},
        "confirmation_time_counts": confirmation_counts,
    }
    write_json(args.output_dir / "aggregate/run_receipt.json", receipt)
    print(f"Revision analyses complete: {args.output_dir}")


if __name__ == "__main__":
    main()
