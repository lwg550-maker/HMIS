"""Gate F prespecified sensitivity-analysis pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats
import statsmodels


RUN_ROOT = Path(__file__).resolve().parents[1]
EXECUTION_ROOT = RUN_ROOT.parent
GATE_B_ROOT = EXECUTION_ROOT / "hmis_liberation_gate_b_dualsource_20261012_113518"
GATE_D_ROOT = EXECUTION_ROOT / "hmis_liberation_gate_d_20261012_184041"
GATE_E_ROOT = EXECUTION_ROOT / "hmis_liberation_gate_e_20261012_213429"
sys.path.insert(0, str(GATE_B_ROOT / "src"))
sys.path.insert(0, str(GATE_D_ROOT / "src"))
sys.path.insert(0, str(GATE_E_ROOT / "src"))

from hmis_liberation.gate_b import derive_patient_endpoint  # noqa: E402
import gate_d as gate_d_core  # noqa: E402
import gate_e as gate_e_core  # noqa: E402


PRIMARY_COLUMNS = [
    "age_10",
    "recorded_sex_male",
    "phenotype_sah",
    "phenotype_ich",
    "phenotype_sdh_edh",
    "log2_1_current_invasive_h",
    "landmark_gcsm_6",
    "pupil_bilateral_reactive",
    "major_sedative_active",
    "any_vasopressor_exposed_6h",
    "pf_per_50",
    "peep_per_5",
    "anchor_year_group_ordinal",
]


def _parse_times(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column in result:
            result[column] = pd.to_datetime(result[column], errors="coerce")
    return result


def apply_day21_candidate_cutoff(primary: pd.DataFrame) -> pd.DataFrame:
    result = _parse_times(
        primary,
        ["event_time", "t_HMIS", "hospital_death", "hospital_discharge"],
    )
    late = result["outcome"].eq("SUSTAINED_LIBERATION") & result["event_time"].gt(
        result["t_HMIS"] + pd.Timedelta(days=21)
    )
    for index in result.index[late]:
        anchor = result.at[index, "t_HMIS"]
        horizon = anchor + pd.Timedelta(days=28)
        death = result.at[index, "hospital_death"]
        discharge = result.at[index, "hospital_discharge"]
        if pd.notna(death) and anchor < death <= horizon:
            result.at[index, "outcome"] = "DEATH_BEFORE_LIBERATION"
            endpoint = death
        else:
            result.at[index, "outcome"] = "CENSORED_WITHOUT_CONFIRMED_LIBERATION"
            endpoint = min(discharge, horizon) if pd.notna(discharge) else horizon
        result.at[index, "event_time"] = endpoint
        result.at[index, "followup_h"] = (endpoint - anchor).total_seconds() / 3600.0
        result.at[index, "confirmation_source"] = "NONE"
        result.at[index, "late_start_after_day21"] = False
    return result


def merge_post_landmark_gaps(
    intervals: pd.DataFrame,
    anchors: pd.DataFrame,
    *,
    maximum_gap_hours: float,
) -> tuple[pd.DataFrame, dict[str, int | float]]:
    source = _parse_times(intervals, ["event_start", "event_end"])
    anchor_map = _parse_times(anchors, ["t_HMIS"]).set_index("subject_id")["t_HMIS"]
    output: list[dict[str, object]] = []
    merged_gap_n = 0
    affected_subjects: set[int] = set()
    maximum_gap = pd.Timedelta(hours=float(maximum_gap_hours))
    for subject_id, group in source.sort_values(["subject_id", "event_start", "event_end"]).groupby("subject_id", sort=False):
        if subject_id not in anchor_map.index:
            raise ValueError(f"missing landmark for subject {subject_id}")
        anchor = pd.Timestamp(anchor_map.loc[subject_id])
        current: dict[str, object] | None = None
        for row in group.to_dict("records"):
            if pd.isna(row["event_start"]) or pd.isna(row["event_end"]):
                raise ValueError("invasive interval contains missing boundary")
            if row["event_end"] <= row["event_start"]:
                raise ValueError("invasive interval has non-positive duration")
            if current is None:
                current = row.copy()
                continue
            current_end = pd.Timestamp(current["event_end"])
            next_start = pd.Timestamp(row["event_start"])
            gap = next_start - current_end
            exact_or_overlap = gap <= pd.Timedelta(0)
            eligible_short_gap = (
                pd.Timedelta(0) < gap <= maximum_gap and current_end > anchor
            )
            if exact_or_overlap or eligible_short_gap:
                current["event_end"] = max(current_end, pd.Timestamp(row["event_end"]))
                if eligible_short_gap:
                    merged_gap_n += 1
                    affected_subjects.add(int(subject_id))
            else:
                output.append(current)
                current = row.copy()
        if current is not None:
            output.append(current)
    result = pd.DataFrame(output, columns=source.columns)
    return result, {
        "maximum_gap_hours": float(maximum_gap_hours),
        "merged_gap_n": merged_gap_n,
        "affected_patient_n": len(affected_subjects),
    }


def apply_discharge_policy(
    primary: pd.DataFrame,
    audit: pd.DataFrame,
    *,
    policy: str,
) -> pd.DataFrame:
    if policy not in {"STRICT_DESTINATION", "NO_SURROGATE"}:
        raise ValueError("unknown discharge policy")
    result = _parse_times(primary, ["event_time", "t_HMIS", "hospital_discharge"])
    action = audit.set_index("subject_id")["strict_action"].to_dict()
    surrogate = result["confirmation_source"].eq("ALIVE_DISCHARGE_SURROGATE")
    for index in result.index[surrogate]:
        subject_id = result.at[index, "subject_id"]
        if subject_id not in action:
            raise ValueError(f"missing discharge audit for surrogate subject {subject_id}")
        censor = policy == "NO_SURROGATE" or action[subject_id] == "CENSOR_AT_DISCHARGE"
        if censor:
            endpoint = result.at[index, "hospital_discharge"]
            if pd.isna(endpoint) or endpoint <= result.at[index, "t_HMIS"]:
                raise ValueError("invalid discharge censoring time")
            result.at[index, "outcome"] = "CENSORED_WITHOUT_CONFIRMED_LIBERATION"
            result.at[index, "event_time"] = endpoint
            result.at[index, "followup_h"] = (
                endpoint - result.at[index, "t_HMIS"]
            ).total_seconds() / 3600.0
            result.at[index, "confirmation_source"] = "NONE"
    return result


def rederive_endpoint(
    base: pd.DataFrame,
    intervals: pd.DataFrame,
    *,
    confirm_hours: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    context = _parse_times(
        base,
        ["t_HMIS", "hospital_death", "hospital_discharge", "first_tracheostomy"],
    )
    source = _parse_times(intervals, ["event_start", "event_end"])
    groups = {int(k): v for k, v in source.groupby("subject_id", sort=False)}
    outcomes: list[dict[str, object]] = []
    candidates: list[dict[str, object]] = []
    for row in context.itertuples(index=False):
        subject_id = int(row.subject_id)
        if subject_id not in groups:
            raise ValueError(f"missing invasive intervals for subject {subject_id}")
        person_intervals = [
            (pd.Timestamp(item.event_start), pd.Timestamp(item.event_end))
            for item in groups[subject_id].sort_values(["event_start", "event_end"]).itertuples(index=False)
        ]
        outcome, person_candidates = derive_patient_endpoint(
            subject_id=subject_id,
            anchor=pd.Timestamp(row.t_HMIS),
            intervals=person_intervals,
            death=row.hospital_death,
            discharge=row.hospital_discharge,
            confirm_hours=int(confirm_hours),
        )
        outcome.update({
            "t_HMIS": row.t_HMIS,
            "landmark_gcsm": row.landmark_gcsm,
            "target_window": row.target_window,
            "first_tracheostomy": row.first_tracheostomy,
            "hospital_discharge": row.hospital_discharge,
            "hospital_death": row.hospital_death,
            "discharge_location": getattr(row, "discharge_location", ""),
            "confirm_hours": int(confirm_hours),
        })
        outcomes.append(outcome)
        for candidate in person_candidates:
            candidate.update({"t_HMIS": row.t_HMIS, "confirm_hours": int(confirm_hours)})
            candidates.append(candidate)
    return (
        pd.DataFrame(outcomes).sort_values("subject_id").reset_index(drop=True),
        pd.DataFrame(candidates),
    )


def assemble_sensitivity_outcomes(
    primary: pd.DataFrame,
    intervals: pd.DataFrame,
    discharge_audit: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[str, int | float]]:
    """Construct every prespecified alternative endpoint on the fixed risk set."""
    primary_sorted = primary.sort_values("subject_id").reset_index(drop=True).copy()
    if len(primary_sorted) != 176 or primary_sorted["subject_id"].nunique() != 176:
        raise AssertionError("Gate F requires the frozen 176-patient landmark risk set")

    anchors = primary_sorted[["subject_id", "t_HMIS"]]
    gap_intervals, gap_audit = merge_post_landmark_gaps(
        intervals, anchors, maximum_gap_hours=6.0
    )
    gap_outcome, _ = rederive_endpoint(
        primary_sorted, gap_intervals, confirm_hours=168
    )
    outcome_48h, _ = rederive_endpoint(
        primary_sorted, intervals, confirm_hours=48
    )
    variants = {
        "PRIMARY_7D": primary_sorted,
        "DAY21_CANDIDATE_CUTOFF": apply_day21_candidate_cutoff(primary_sorted),
        "GAP_LE_6H": gap_outcome,
        "STRICT_DISCHARGE": apply_discharge_policy(
            primary_sorted, discharge_audit, policy="STRICT_DESTINATION"
        ),
        "NO_DISCHARGE_SURROGATE": apply_discharge_policy(
            primary_sorted, discharge_audit, policy="NO_SURROGATE"
        ),
        "SUSTAINED_48H": outcome_48h,
    }
    expected_ids = set(primary_sorted["subject_id"])
    for label, frame in variants.items():
        if len(frame) != 176 or frame["subject_id"].nunique() != 176:
            raise AssertionError(f"{label} changed the frozen risk-set size")
        if set(frame["subject_id"]) != expected_ids:
            raise AssertionError(f"{label} changed risk-set membership")
    return variants, gap_audit


def outcome_arrays(outcome: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    durations = pd.to_numeric(outcome["followup_h"], errors="raise").to_numpy(float)
    codes = outcome["outcome"].map({
        "SUSTAINED_LIBERATION": 1,
        "DEATH_BEFORE_LIBERATION": 2,
        "NO_LIBERATION_BY_28D": 0,
        "CENSORED_WITHOUT_CONFIRMED_LIBERATION": 0,
    })
    if codes.isna().any() or np.any(durations <= 0):
        raise ValueError("invalid sensitivity outcome or follow-up")
    return durations, codes.to_numpy(int)


def fio2_replacement_matrix(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    columns = ["fio2_per_0_10" if name == "pf_per_50" else name for name in PRIMARY_COLUMNS]
    matrix = data.loc[:, [name for name in PRIMARY_COLUMNS if name != "pf_per_50"]].copy()
    matrix["fio2_per_0_10"] = pd.to_numeric(data["auxiliary_fio2"], errors="coerce") / 0.10
    return matrix.loc[:, columns], columns


def descriptive_strata(data: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "GCS-M 5": pd.to_numeric(data["landmark_gcsm"], errors="coerce").eq(5),
        "GCS-M 6": pd.to_numeric(data["landmark_gcsm"], errors="coerce").eq(6),
        "W1": data["target_window"].astype(str).eq("W1"),
        "W2-W5": data["target_window"].astype(str).isin(["W2", "W3", "W4", "W5"]),
    }


IMPUTATION_AUXILIARIES = [
    "auxiliary_fio2",
    "auxiliary_resp_rate",
    "auxiliary_rass",
    "auxiliary_creatinine",
    "auxiliary_wbc",
]


def build_imputation_frame(
    data: pd.DataFrame,
    outcome: pd.DataFrame,
    model_columns: list[str],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    required = ["subject_id", *model_columns, *IMPUTATION_AUXILIARIES]
    missing = [column for column in required if column not in data]
    if missing:
        raise KeyError(f"missing imputation inputs: {missing}")
    merged = data.loc[:, required].merge(
        outcome[["subject_id", "outcome", "followup_h"]],
        on="subject_id",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(data):
        raise ValueError("outcome does not exhaust the sensitivity risk set")
    durations, codes = outcome_arrays(merged)
    frame = merged.loc[:, model_columns + IMPUTATION_AUXILIARIES].astype(float).copy()
    frame["event_liberation"] = (codes == 1).astype(float)
    frame["event_death"] = (codes == 2).astype(float)
    frame["log_followup_h"] = np.log(durations)
    frame["na_liberation"] = gate_e_core.nelson_aalen_at_observed_times(
        durations, (codes == 1).astype(int)
    )
    frame["na_death"] = gate_e_core.nelson_aalen_at_observed_times(
        durations, (codes == 2).astype(int)
    )
    return frame, durations, codes


def _pooled_estimate_table(pool: dict, parameters: list[str]) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for index, parameter in enumerate(parameters):
        estimate = float(pool["estimate"][index])
        se = math.sqrt(float(pool["total_covariance"][index, index]))
        df = float(pool["degrees_freedom"][index])
        critical = float(scipy.stats.t.ppf(0.975, df)) if np.isfinite(df) else 1.959963984540054
        p_value = (
            float(2 * scipy.stats.t.sf(abs(estimate / se), df))
            if np.isfinite(df)
            else float(2 * scipy.stats.norm.sf(abs(estimate / se)))
        )
        rows.append({
            "parameter": parameter,
            "log_hazard_ratio": estimate,
            "standard_error": se,
            "hazard_ratio": math.exp(estimate),
            "ci_low": math.exp(estimate - critical * se),
            "ci_high": math.exp(estimate + critical * se),
            "p_value": p_value,
            "fraction_missing_information": float(pool["fraction_missing_information"][index]),
        })
    return pd.DataFrame(rows)


def fit_mi_cause_specific(
    data: pd.DataFrame,
    outcome: pd.DataFrame,
    model_columns: list[str],
    *,
    imputations: int,
    iterations: int,
    seed: int,
) -> dict[str, object]:
    frame, durations, codes = build_imputation_frame(data, outcome, model_columns)
    completed = gate_e_core.mice_pmm_impute(
        frame, m=imputations, iterations=iterations, seed=seed, k_pmm=5
    )
    estimates, covariances = [], []
    gradients, ranks, warnings = [], [], []
    scaled_residuals = []
    event_mask = codes == 1
    for complete in completed:
        matrix = complete.loc[:, model_columns].to_numpy(float)
        fit = gate_e_core.fit_phreg(
            durations, event_mask.astype(int), matrix, robust=False
        )
        estimates.append(fit["params"])
        covariances.append(fit["covariance"])
        gradients.append(fit["gradient_max_abs"])
        ranks.append(int(np.linalg.matrix_rank(np.column_stack([np.ones(len(matrix)), matrix]))))
        warnings.append(len(fit["warnings"]))
        residuals = np.asarray(fit["result"].schoenfeld_residuals)[event_mask]
        scaled_residuals.append(
            residuals @ fit["covariance"] * int(event_mask.sum())
        )
    pool = gate_e_core.rubin_pool(np.asarray(estimates), np.asarray(covariances))
    mean_residuals = np.mean(np.stack(scaled_residuals), axis=0)
    ph = gate_e_core.schoenfeld_time_test(durations[event_mask], mean_residuals)
    return {
        "estimates": _pooled_estimate_table(pool, model_columns),
        "diagnostics": {
            "n": int(len(data)),
            "target_events": int(event_mask.sum()),
            "competing_deaths": int((codes == 2).sum()),
            "ordinary_or_administrative_censoring": int((codes == 0).sum()),
            "imputations": int(imputations),
            "iterations": int(iterations),
            "all_models_converged": bool(max(gradients) < 1e-4 and sum(warnings) == 0),
            "maximum_gradient_abs": float(max(gradients)),
            "warning_total": int(sum(warnings)),
            "minimum_rank_with_intercept": int(min(ranks)),
            "expected_rank_with_intercept": int(len(model_columns) + 1),
            "ph_global_p": float(ph["global_p"]),
            "ph_variable_p": {
                parameter: float(value)
                for parameter, value in zip(model_columns, ph["per_variable_p"])
            },
        },
    }


def fit_complete_case_cox(
    data: pd.DataFrame,
    outcome: pd.DataFrame,
    model_columns: list[str],
) -> dict[str, object]:
    """Fit the locked cause-specific Cox model among complete cases only."""
    required = ["subject_id", *model_columns]
    missing = [column for column in required if column not in data]
    if missing:
        raise KeyError(f"missing complete-case inputs: {missing}")
    merged = data.loc[:, required].merge(
        outcome[["subject_id", "outcome", "followup_h"]],
        on="subject_id",
        how="inner",
        validate="one_to_one",
    )
    complete = merged.dropna(subset=model_columns).copy()
    durations, codes = outcome_arrays(complete)
    event_mask = codes == 1
    matrix = complete.loc[:, model_columns].to_numpy(float)
    fit = gate_e_core.fit_phreg(
        durations, event_mask.astype(int), matrix, robust=False
    )
    params = np.asarray(fit["params"], dtype=float)
    covariance = np.asarray(fit["covariance"], dtype=float)
    standard_error = np.sqrt(np.diag(covariance))
    estimates = pd.DataFrame({
        "parameter": model_columns,
        "log_hazard_ratio": params,
        "standard_error": standard_error,
        "hazard_ratio": np.exp(params),
        "ci_low": np.exp(params - 1.959963984540054 * standard_error),
        "ci_high": np.exp(params + 1.959963984540054 * standard_error),
        "p_value": 2.0 * scipy.stats.norm.sf(np.abs(params / standard_error)),
        "fraction_missing_information": np.nan,
    })
    residuals = np.asarray(fit["result"].schoenfeld_residuals)[event_mask]
    scaled_residuals = residuals @ covariance * int(event_mask.sum())
    ph = gate_e_core.schoenfeld_time_test(
        durations[event_mask], scaled_residuals
    )
    rank = int(
        np.linalg.matrix_rank(np.column_stack([np.ones(len(matrix)), matrix]))
    )
    return {
        "estimates": estimates,
        "diagnostics": {
            "n": int(len(complete)),
            "target_events": int(event_mask.sum()),
            "competing_deaths": int((codes == 2).sum()),
            "ordinary_or_administrative_censoring": int((codes == 0).sum()),
            "imputations": 0,
            "iterations": 0,
            "all_models_converged": bool(
                fit["gradient_max_abs"] < 1e-4 and len(fit["warnings"]) == 0
            ),
            "maximum_gradient_abs": float(fit["gradient_max_abs"]),
            "warning_total": int(len(fit["warnings"])),
            "minimum_rank_with_intercept": rank,
            "expected_rank_with_intercept": int(len(model_columns) + 1),
            "ph_global_p": float(ph["global_p"]),
            "ph_variable_p": {
                parameter: float(value)
                for parameter, value in zip(model_columns, ph["per_variable_p"])
            },
        },
    }


def stratified_aj(
    data: pd.DataFrame,
    outcome: pd.DataFrame,
    *,
    evaluation_days: list[float],
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    merged = data[["subject_id", "landmark_gcsm", "target_window"]].merge(
        outcome[["subject_id", "outcome", "followup_h"]],
        on="subject_id",
        how="inner",
        validate="one_to_one",
    )
    durations, codes = outcome_arrays(merged)
    masks = descriptive_strata(merged)
    evaluation = np.asarray(evaluation_days, dtype=float)
    rows: list[dict[str, object]] = []
    for stratum_index, (label, mask) in enumerate(masks.items()):
        selected = mask.to_numpy(bool)
        aj = gate_d_core.aalen_johansen(durations[selected] / 24.0, codes[selected])
        liberation = gate_d_core.step_at(aj, evaluation, 1)
        death = gate_d_core.step_at(aj, evaluation, 2)
        llo, lhi, dlo, dhi = gate_d_core.bootstrap_aj(
            durations[selected] / 24.0,
            codes[selected],
            evaluation,
            replicates=replicates,
            seed=seed + 1009 * stratum_index,
        )
        for time_index, day in enumerate(evaluation):
            rows.append({
                "stratum": label,
                "n": int(selected.sum()),
                "target_event_n": int((codes[selected] == 1).sum()),
                "competing_death_n": int((codes[selected] == 2).sum()),
                "day": float(day),
                "liberation_cif": float(liberation[time_index]),
                "liberation_ci_low": float(llo[time_index]),
                "liberation_ci_high": float(lhi[time_index]),
                "death_cif": float(death[time_index]),
                "death_ci_low": float(dlo[time_index]),
                "death_ci_high": float(dhi[time_index]),
            })
    return pd.DataFrame(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    raise TypeError(f"cannot serialize {type(value)}")


def _outcome_count_table(variants: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for label, frame in variants.items():
        counts = frame["outcome"].value_counts().to_dict()
        rows.append({
            "analysis": label,
            "n": int(len(frame)),
            "sustained_liberation_n": int(counts.get("SUSTAINED_LIBERATION", 0)),
            "death_before_liberation_n": int(counts.get("DEATH_BEFORE_LIBERATION", 0)),
            "no_liberation_by_28d_n": int(counts.get("NO_LIBERATION_BY_28D", 0)),
            "early_censor_without_confirmation_n": int(
                counts.get("CENSORED_WITHOUT_CONFIRMED_LIBERATION", 0)
            ),
        })
    return pd.DataFrame(rows)


def _attach_analysis_label(
    result: dict[str, object], analysis: str, estimand: str
) -> tuple[pd.DataFrame, dict[str, object]]:
    table = result["estimates"].copy()
    table.insert(0, "estimand", estimand)
    table.insert(0, "analysis", analysis)
    diagnostic = {"analysis": analysis, "estimand": estimand, **result["diagnostics"]}
    return table, diagnostic


def _gate_e_reference_rows(table: pd.DataFrame) -> pd.DataFrame:
    common = table[["parameter"]].copy()
    cause_specific = common.assign(
        analysis="PRIMARY_7D_MI_REFERENCE",
        estimand="cause-specific hazard",
        hazard_ratio=table["csHR"],
        ci_low=table["csHR_ci_low"],
        ci_high=table["csHR_ci_high"],
        p_value=table["cs_p_value"],
        fraction_missing_information=table["cs_fmi"],
    )
    fine_gray = common.assign(
        analysis="PRIMARY_7D_FINE_GRAY_REFERENCE",
        estimand="subdistribution hazard",
        hazard_ratio=table["sdHR"],
        ci_low=table["sdHR_ci_low"],
        ci_high=table["sdHR_ci_high"],
        p_value=table["sd_p_value"],
        fraction_missing_information=table["sd_fmi"],
    )
    ordered = [
        "analysis", "estimand", "parameter", "hazard_ratio", "ci_low",
        "ci_high", "p_value", "fraction_missing_information",
    ]
    return pd.concat([cause_specific, fine_gray], ignore_index=True)[ordered]


def _assert_frozen_preconditions(
    variants: dict[str, pd.DataFrame],
    analysis_data: pd.DataFrame,
    gate_e_qc: dict[str, object],
    model_lock: dict[str, object],
) -> None:
    if gate_e_qc.get("status") != "PASS":
        raise AssertionError("Gate E did not pass")
    if int(gate_e_qc.get("n", -1)) != 176:
        raise AssertionError("Gate E risk set is not 176")
    if int(model_lock.get("analysis_population_n", -1)) != 176:
        raise AssertionError("model lock risk set is not 176")
    if int(model_lock.get("primary_predictor_df_hard_cap", -1)) != 13:
        raise AssertionError("model lock does not preserve the 13-df cap")
    if len(PRIMARY_COLUMNS) != 13:
        raise AssertionError("Gate F primary matrix is not 13 df")
    if len(analysis_data) != 176 or analysis_data["subject_id"].nunique() != 176:
        raise AssertionError("Gate C analysis data are not the frozen risk set")
    counts = _outcome_count_table(variants).set_index("analysis")
    expected = {
        "PRIMARY_7D": (131, 40, 5, 0),
        "DAY21_CANDIDATE_CUTOFF": (125, 40, 5, 6),
        "STRICT_DISCHARGE": (118, 40, 5, 13),
        "NO_DISCHARGE_SURROGATE": (101, 40, 5, 30),
    }
    for label, values in expected.items():
        observed = tuple(
            int(counts.loc[label, column])
            for column in [
                "sustained_liberation_n", "death_before_liberation_n",
                "no_liberation_by_28d_n", "early_censor_without_confirmation_n",
            ]
        )
        if observed != values:
            raise AssertionError(
                f"{label} frozen outcome boundary differs: {observed} != {values}"
            )
    if int(counts.loc["SUSTAINED_48H", "sustained_liberation_n"]) != 145:
        raise AssertionError("48-hour sustained-liberation count differs from 145")


def _write_markdown_table(table: pd.DataFrame, path: Path) -> None:
    display = table.copy()
    numeric = ["hazard_ratio", "ci_low", "ci_high", "p_value"]
    for column in numeric:
        if column in display:
            display[column] = pd.to_numeric(display[column], errors="coerce").map(
                lambda x: "" if pd.isna(x) else f"{x:.4g}"
            )
    columns = [str(column) for column in display.columns]
    def clean(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = [
        "| " + " | ".join(clean(column) for column in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in display.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(clean(value) for value in row) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run locked Gate F analyses")
    parser.add_argument("--analysis-data", type=Path, required=True)
    parser.add_argument("--primary-outcome", type=Path, required=True)
    parser.add_argument("--intervals", type=Path, required=True)
    parser.add_argument("--discharge-audit", type=Path, required=True)
    parser.add_argument("--gate-e-table", type=Path, required=True)
    parser.add_argument("--gate-e-qc", type=Path, required=True)
    parser.add_argument("--model-lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--imputations", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261012)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.imputations < 2 or args.iterations < 1 or args.bootstrap_replicates < 1:
        raise ValueError("invalid computation settings")
    output = args.output_dir.resolve()
    for name in ["aggregate", "restricted", "tables", "figures", "logs"]:
        (output / name).mkdir(parents=True, exist_ok=True)

    inputs = [
        args.analysis_data, args.primary_outcome, args.intervals,
        args.discharge_audit, args.gate_e_table, args.gate_e_qc, args.model_lock,
    ]
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    analysis_data = pd.read_csv(args.analysis_data)
    primary = pd.read_csv(args.primary_outcome)
    intervals = pd.read_csv(args.intervals)
    discharge_audit = pd.read_csv(args.discharge_audit)
    gate_e_table = pd.read_csv(args.gate_e_table)
    gate_e_qc = json.loads(args.gate_e_qc.read_text(encoding="utf-8"))
    model_lock = json.loads(args.model_lock.read_text(encoding="utf-8"))

    variants, gap_audit = assemble_sensitivity_outcomes(
        primary, intervals, discharge_audit
    )
    _assert_frozen_preconditions(variants, analysis_data, gate_e_qc, model_lock)
    outcome_counts = _outcome_count_table(variants)
    outcome_counts.to_csv(output / "aggregate/gate_f_outcome_counts.csv", index=False)
    (output / "aggregate/gap_le_6h_audit.json").write_text(
        json.dumps(gap_audit, indent=2, default=_json_default), encoding="utf-8"
    )
    internal = pd.concat(
        [frame.assign(analysis=label) for label, frame in variants.items()],
        ignore_index=True,
    )
    internal.to_csv(
        output / "restricted/gate_f_outcome_variants_internal.csv", index=False
    )

    tables = [_gate_e_reference_rows(gate_e_table)]
    diagnostics: list[dict[str, object]] = []
    ordered_models = [
        ("DAY21_CANDIDATE_CUTOFF", variants["DAY21_CANDIDATE_CUTOFF"], analysis_data, PRIMARY_COLUMNS),
        ("GAP_LE_6H", variants["GAP_LE_6H"], analysis_data, PRIMARY_COLUMNS),
        ("STRICT_DISCHARGE", variants["STRICT_DISCHARGE"], analysis_data, PRIMARY_COLUMNS),
        ("NO_DISCHARGE_SURROGATE", variants["NO_DISCHARGE_SURROGATE"], analysis_data, PRIMARY_COLUMNS),
    ]
    for label, outcome, model_data, columns in ordered_models:
        print(f"Gate F: fitting {label} with {args.imputations} imputations", flush=True)
        fitted = fit_mi_cause_specific(
            model_data, outcome, columns,
            imputations=args.imputations,
            iterations=args.iterations,
            seed=args.seed,
        )
        table, diagnostic = _attach_analysis_label(
            fitted, label, "cause-specific hazard"
        )
        tables.append(table)
        diagnostics.append(diagnostic)

    print("Gate F: fitting COMPLETE_CASE", flush=True)
    complete_case = fit_complete_case_cox(
        analysis_data, variants["PRIMARY_7D"], PRIMARY_COLUMNS
    )
    table, diagnostic = _attach_analysis_label(
        complete_case, "COMPLETE_CASE", "cause-specific hazard"
    )
    tables.append(table)
    diagnostics.append(diagnostic)

    fio2_data, fio2_columns = fio2_replacement_matrix(analysis_data)
    fio2_data.insert(0, "subject_id", analysis_data["subject_id"].to_numpy())
    for auxiliary in IMPUTATION_AUXILIARIES:
        if auxiliary not in fio2_data:
            fio2_data[auxiliary] = analysis_data[auxiliary].to_numpy()
    print(f"Gate F: fitting FIO2_REPLACES_PF with {args.imputations} imputations", flush=True)
    fio2 = fit_mi_cause_specific(
        fio2_data, variants["PRIMARY_7D"], fio2_columns,
        imputations=args.imputations,
        iterations=args.iterations,
        seed=args.seed,
    )
    table, diagnostic = _attach_analysis_label(
        fio2, "FIO2_REPLACES_PF", "cause-specific hazard"
    )
    tables.append(table)
    diagnostics.append(diagnostic)

    print(f"Gate F: fitting SUSTAINED_48H with {args.imputations} imputations", flush=True)
    outcome48 = fit_mi_cause_specific(
        analysis_data, variants["SUSTAINED_48H"], PRIMARY_COLUMNS,
        imputations=args.imputations,
        iterations=args.iterations,
        seed=args.seed,
    )
    table, diagnostic = _attach_analysis_label(
        outcome48, "SUSTAINED_48H", "cause-specific hazard"
    )
    tables.append(table)
    diagnostics.append(diagnostic)

    print("Gate F: generating descriptive stratified cumulative incidence", flush=True)
    strata = stratified_aj(
        analysis_data, variants["PRIMARY_7D"],
        evaluation_days=[3, 7, 14, 28],
        replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    strata.to_csv(output / "aggregate/gate_f_descriptive_stratified_cif.csv", index=False)

    result_table = pd.concat(tables, ignore_index=True, sort=False)
    preferred = [
        "analysis", "estimand", "parameter", "hazard_ratio", "ci_low",
        "ci_high", "p_value", "fraction_missing_information",
    ]
    result_table = result_table[[column for column in preferred if column in result_table]]
    result_table.to_csv(output / "tables/Supplementary_Table_GateF_all_models.csv", index=False)
    _write_markdown_table(
        result_table, output / "tables/Supplementary_Table_GateF_all_models.md"
    )
    (output / "aggregate/gate_f_model_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, default=_json_default), encoding="utf-8"
    )
    diagnostic_flat = pd.DataFrame([
        {key: value for key, value in item.items() if key != "ph_variable_p"}
        for item in diagnostics
    ])
    diagnostic_flat.to_csv(output / "aggregate/gate_f_model_diagnostics.csv", index=False)

    expected_rank = 14
    qc = {
        "gate": "F",
        "status": "PASS" if all(
            bool(item["all_models_converged"])
            and int(item["minimum_rank_with_intercept"]) == expected_rank
            for item in diagnostics
        ) else "STOPPED",
        "risk_set_n": 176,
        "imputations": int(args.imputations),
        "iterations": int(args.iterations),
        "seed": int(args.seed),
        "bootstrap_replicates": int(args.bootstrap_replicates),
        "model_df": 13,
        "all_parameters_reported": bool(
            all(len(item) == 13 for item in tables[1:])
        ),
        "outcome_counts": outcome_counts.to_dict("records"),
        "gap_le_6h_audit": gap_audit,
        "complete_case_n": int(complete_case["diagnostics"]["n"]),
        "gcs_m_5_n": int((analysis_data["landmark_gcsm"] == 5).sum()),
        "gcs_m_6_n": int((analysis_data["landmark_gcsm"] == 6).sum()),
        "w1_n": int((analysis_data["target_window"] == "W1").sum()),
        "w2_w5_n": int(analysis_data["target_window"].isin(["W2", "W3", "W4", "W5"]).sum()),
        "descriptive_strata_no_hypothesis_tests": True,
        "fine_gray_used_as_complementary_reference": True,
        "no_variable_selection": True,
        "gate_g_not_run": True,
    }
    if qc["complete_case_n"] != 133:
        raise AssertionError("complete-case n differs from the frozen Gate C audit")
    if qc["status"] != "PASS":
        raise AssertionError("one or more Gate F models failed convergence or rank checks")
    (output / "aggregate/gate_f_qc.json").write_text(
        json.dumps(qc, indent=2, default=_json_default), encoding="utf-8"
    )
    receipt = {
        "gate": "F",
        "status": qc["status"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": [
            {"path": str(path.resolve()), "sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in inputs
        ],
        "code": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "settings": {
            "imputations": args.imputations,
            "iterations": args.iterations,
            "bootstrap_replicates": args.bootstrap_replicates,
            "seed": args.seed,
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
        },
    }
    (output / "aggregate/run_receipt.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )
    print(f"Gate F complete: status={qc['status']}, output={output}", flush=True)


if __name__ == "__main__":
    main()
