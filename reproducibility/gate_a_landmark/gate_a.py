from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


KEYS = ["subject_id", "hadm_id", "stay_id"]
WINDOW_END_HOURS = {
    "B0": 0.0,
    "W1": 4.0,
    "W2": 8.0,
    "W3": 12.0,
    "W4": 24.0,
    "W5": 48.0,
}
WINDOW_ORDER = {name: order for order, name in enumerate(WINDOW_END_HOURS)}
GCS_LOOKBACK_HOURS = 4.0
NMB_ITEMIDS = {221555, 229233, 222062}


@dataclass(frozen=True)
class GateAExpected:
    parent_n: int = 474
    landmark_candidate_n: int = 221
    risk_set_n: int = 176
    excluded_not_ventilated_n: int = 45


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "t", "yes"}


def _parse_times(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column in out.columns:
            out[column] = pd.to_datetime(out[column], errors="coerce")
    return out


def _parse_cohort_times(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Parse the frozen cohort's D/M/Y exports without locale ambiguity."""
    out = frame.copy()
    for column in columns:
        if column not in out.columns:
            continue
        if pd.api.types.is_datetime64_any_dtype(out[column]):
            continue
        parsed = pd.to_datetime(
            out[column], format="%d/%m/%Y %H:%M:%S", errors="coerce"
        )
        unresolved = parsed.isna() & out[column].notna() & out[column].astype(str).ne("")
        if unresolved.any():
            parsed.loc[unresolved] = pd.to_datetime(
                out.loc[unresolved, column],
                format="%Y-%m-%d %H:%M:%S",
                errors="coerce",
            )
        out[column] = parsed
    return out


def validate_cohort(cohort: pd.DataFrame) -> pd.DataFrame:
    required = {*KEYS, "t0", "outtime", "deathtime"}
    missing = sorted(required - set(cohort.columns))
    if missing:
        raise ValueError(f"cohort missing columns: {missing}")
    out = _parse_cohort_times(cohort, ["t0", "outtime", "deathtime"])
    if out["subject_id"].duplicated().any():
        raise ValueError("cohort subject_id must be unique")
    if out[KEYS + ["t0", "outtime"]].isna().any().any():
        raise ValueError("cohort contains missing identifiers, t0, or outtime")
    return out


def validate_events(events: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    required = {
        "subject_id",
        "event_type",
        "event_start",
        "event_end",
        "charttime",
        "storetime",
        "itemid",
        "value_num",
        "rate",
    }
    missing = sorted(required - set(events.columns))
    if missing:
        raise ValueError(f"event extract missing columns: {missing}")
    out = _parse_times(
        events,
        ["event_start", "event_end", "charttime", "storetime", "dischtime", "deathtime"],
    )
    out["itemid"] = pd.to_numeric(out["itemid"], errors="coerce")
    out["value_num"] = pd.to_numeric(out["value_num"], errors="coerce")
    out["rate"] = pd.to_numeric(out["rate"], errors="coerce")
    unknown = set(out["subject_id"].dropna().astype(int)) - set(cohort["subject_id"].astype(int))
    if unknown:
        raise ValueError(f"event extract contains {len(unknown)} subjects outside cohort")
    covered = set(out.loc[out["event_type"].eq("COHORT"), "subject_id"].astype(int))
    expected = set(cohort["subject_id"].astype(int))
    if covered != expected:
        raise ValueError(
            "COHORT sentinel rows do not cover the source cohort: "
            f"missing={len(expected-covered)}, extra={len(covered-expected)}"
        )
    if "seed_n" in out.columns:
        seed_counts = pd.to_numeric(out["seed_n"], errors="coerce").dropna().unique()
        if len(seed_counts) != 1 or int(seed_counts[0]) != len(cohort):
            raise ValueError(
                f"database seed assertion failed: seed_n={seed_counts.tolist()}, "
                f"cohort_n={len(cohort)}"
            )
    out = out.reset_index(drop=True)
    out["source_row_number"] = np.arange(1, len(out) + 1, dtype="int64")
    return out


def _terminal_state(row: pd.Series, checkpoint_end: pd.Timestamp) -> str:
    death = row["deathtime"]
    outtime = row["outtime"]
    if pd.notna(death) and death <= checkpoint_end and death <= outtime:
        return "DEAD"
    if outtime <= checkpoint_end:
        return "ICU_EXIT_ALIVE"
    return "NONTERMINAL"


def _nmb_active_at_checkpoint(events: pd.DataFrame, checkpoint_end: pd.Timestamp) -> bool:
    nmb = events.loc[
        events["event_type"].eq("NMB_INFUSION")
        & events["itemid"].isin(NMB_ITEMIDS)
        & events["rate"].fillna(0).gt(0)
        & events["event_start"].notna()
        & events["event_end"].notna()
    ].copy()
    if nmb.empty:
        return False
    valid_duration = nmb["event_end"] > nmb["event_start"] + pd.Timedelta(minutes=5)
    active = (
        nmb["event_start"].le(checkpoint_end)
        & nmb["event_end"].gt(checkpoint_end)
        & valid_duration
    )
    return bool(active.any())


def _select_gcs(
    events: pd.DataFrame,
    checkpoint_end: pd.Timestamp,
) -> dict[str, object]:
    lower = checkpoint_end - pd.Timedelta(hours=GCS_LOOKBACK_HOURS)
    candidates = events.loc[
        events["event_type"].eq("GCS_M")
        & events["charttime"].gt(lower)
        & events["charttime"].le(checkpoint_end)
        & events["value_num"].between(1, 6)
    ].copy()
    if candidates.empty:
        return {
            "raw_gcs": np.nan,
            "raw_charttime": pd.NaT,
            "raw_storetime": pd.NaT,
            "source_row_number": np.nan,
            "conflict": False,
            "candidate_record_n": 0,
        }

    latest_charttime = candidates["charttime"].max()
    latest = candidates.loc[candidates["charttime"].eq(latest_charttime)].copy()
    if latest["storetime"].notna().any():
        latest_storetime = latest["storetime"].max()
        latest = latest.loc[latest["storetime"].eq(latest_storetime)].copy()
    else:
        latest_storetime = pd.NaT
        latest = latest.loc[latest["storetime"].isna()].copy()

    values = latest["value_num"].dropna().unique()
    if len(values) != 1:
        return {
            "raw_gcs": np.nan,
            "raw_charttime": latest_charttime,
            "raw_storetime": latest_storetime,
            "source_row_number": np.nan,
            "conflict": True,
            "candidate_record_n": int(len(candidates)),
        }
    chosen = latest.sort_values("source_row_number", kind="stable").iloc[-1]
    return {
        "raw_gcs": float(values[0]),
        "raw_charttime": latest_charttime,
        "raw_storetime": latest_storetime,
        "source_row_number": int(chosen["source_row_number"]),
        "conflict": False,
        "candidate_record_n": int(len(candidates)),
    }


def build_checkpoint_audit(cohort: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    cohort = validate_cohort(cohort)
    events = validate_events(events, cohort)
    event_groups = {int(k): v for k, v in events.groupby("subject_id", sort=False)}
    rows: list[dict[str, object]] = []

    for patient in cohort.itertuples(index=False):
        patient_row = pd.Series(patient._asdict())
        subject_id = int(patient.subject_id)
        subject_events = event_groups[subject_id]
        t0 = pd.Timestamp(patient.t0)
        for window_id, end_h in WINDOW_END_HOURS.items():
            checkpoint_end = t0 + pd.Timedelta(hours=end_h)
            terminal = _terminal_state(patient_row, checkpoint_end)
            selected = _select_gcs(subject_events, checkpoint_end)
            nmb_active = _nmb_active_at_checkpoint(subject_events, checkpoint_end)
            usable = (
                terminal == "NONTERMINAL"
                and not bool(selected["conflict"])
                and not nmb_active
                and pd.notna(selected["raw_gcs"])
            )
            rows.append(
                {
                    "subject_id": subject_id,
                    "hadm_id": int(patient.hadm_id),
                    "stay_id": int(patient.stay_id),
                    "window_id": window_id,
                    "window_order": WINDOW_ORDER[window_id],
                    "checkpoint_end": checkpoint_end,
                    "search_start_open": checkpoint_end
                    - pd.Timedelta(hours=GCS_LOOKBACK_HOURS),
                    "search_end_closed": checkpoint_end,
                    "terminal_state": terminal,
                    "candidate_record_n": selected["candidate_record_n"],
                    "raw_gcsm": selected["raw_gcs"],
                    "raw_gcsm_charttime": selected["raw_charttime"],
                    "raw_gcsm_storetime": selected["raw_storetime"],
                    "selected_source_row_number": selected["source_row_number"],
                    "same_timestamp_unresolved_conflict": bool(selected["conflict"]),
                    "nmb_active_at_checkpoint": nmb_active,
                    "gcsm": int(selected["raw_gcs"]) if usable else np.nan,
                    "gcsm_charttime": selected["raw_charttime"] if usable else pd.NaT,
                    "motor_state": (
                        "LOW"
                        if usable and float(selected["raw_gcs"]) <= 4
                        else "HIGH"
                        if usable
                        else "UNKNOWN"
                    ),
                }
            )

    audit = pd.DataFrame(rows).sort_values(
        ["subject_id", "window_order"], kind="stable"
    )
    reused = (
        audit.dropna(subset=["selected_source_row_number"])
        .groupby("selected_source_row_number")["window_id"]
        .nunique()
    )
    if reused.gt(1).any():
        raise AssertionError("a GCS source row was assigned to multiple checkpoints")
    return audit.reset_index(drop=True)


def derive_landmarks(checkpoints: pd.DataFrame) -> pd.DataFrame:
    required = {
        "subject_id",
        "hadm_id",
        "stay_id",
        "window_id",
        "window_order",
        "motor_state",
        "gcsm",
        "gcsm_charttime",
    }
    missing = sorted(required - set(checkpoints.columns))
    if missing:
        raise ValueError(f"checkpoint audit missing columns: {missing}")
    rows: list[dict[str, object]] = []
    for subject_id, group in checkpoints.groupby("subject_id", sort=False):
        ordered = group.sort_values("window_order", kind="stable").reset_index(drop=True)
        for index in range(1, len(ordered)):
            previous = ordered.iloc[index - 1]
            current = ordered.iloc[index]
            if int(current["window_order"]) != int(previous["window_order"]) + 1:
                raise AssertionError("checkpoint sequence is not adjacent")
            if previous["motor_state"] == "LOW" and current["motor_state"] == "HIGH":
                rows.append(
                    {
                        "subject_id": int(subject_id),
                        "hadm_id": int(current["hadm_id"]),
                        "stay_id": int(current["stay_id"]),
                        "low_window": previous["window_id"],
                        "low_gcsm": int(previous["gcsm"]),
                        "low_gcsm_charttime": previous["gcsm_charttime"],
                        "target_window": current["window_id"],
                        "landmark_gcsm": int(current["gcsm"]),
                        "t_HMIS": current["gcsm_charttime"],
                    }
                )
                break
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(
            columns=[
                "subject_id",
                "hadm_id",
                "stay_id",
                "low_window",
                "low_gcsm",
                "low_gcsm_charttime",
                "target_window",
                "landmark_gcsm",
                "t_HMIS",
            ]
        )
    if out["subject_id"].duplicated().any():
        raise AssertionError("more than one landmark retained per patient")
    out["low_gcsm_charttime"] = pd.to_datetime(
        out["low_gcsm_charttime"], errors="raise"
    )
    out["t_HMIS"] = pd.to_datetime(out["t_HMIS"], errors="raise")
    if not (out["low_gcsm_charttime"] < out["t_HMIS"]).all():
        raise AssertionError("landmark target record must be strictly after the low record")
    return out.sort_values(["t_HMIS", "subject_id"], kind="stable").reset_index(drop=True)


def merge_exact_intervals(
    intervals: Iterable[tuple[pd.Timestamp, pd.Timestamp]],
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    valid = sorted(
        (pd.Timestamp(start), pd.Timestamp(end))
        for start, end in intervals
        if pd.notna(start) and pd.notna(end) and end > start
    )
    merged: list[list[pd.Timestamp]] = []
    for start, end in valid:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def classify_risk_set(landmarks: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    events = _parse_times(events, ["event_start", "event_end"])
    vent = events.loc[events["event_type"].eq("INVASIVE_VENT")].copy()
    groups = {int(k): v for k, v in vent.groupby("subject_id", sort=False)}
    rows: list[dict[str, object]] = []
    for landmark in landmarks.itertuples(index=False):
        subject_id = int(landmark.subject_id)
        group = groups.get(subject_id, pd.DataFrame(columns=vent.columns))
        intervals = merge_exact_intervals(zip(group["event_start"], group["event_end"]))
        t_hmis = pd.Timestamp(landmark.t_HMIS)
        containing = [(start, end) for start, end in intervals if start <= t_hmis < end]
        if containing:
            risk = True
            reason = "IN_INVASIVE_INTERVAL_AT_T_HMIS"
            active_start, active_end = containing[0]
        else:
            risk = False
            active_start = pd.NaT
            active_end = pd.NaT
            if any(end == t_hmis for _, end in intervals):
                reason = "AT_INVASIVE_INTERVAL_END_BOUNDARY"
            elif not intervals:
                reason = "NO_INVASIVE_INTERVAL_EXTRACTED"
            elif t_hmis < intervals[0][0]:
                reason = "BEFORE_FIRST_INVASIVE_INTERVAL"
            elif t_hmis >= intervals[-1][1]:
                reason = "AT_OR_AFTER_FINAL_INVASIVE_INTERVAL"
            else:
                reason = "BETWEEN_INVASIVE_INTERVALS"
        row = landmark._asdict()
        row.update(
            {
                "risk_set": risk,
                "risk_set_reason": reason,
                "active_vent_start": active_start,
                "active_vent_end": active_end,
                "merged_invasive_interval_n": len(intervals),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def compare_frozen_landmarks(
    landmarks: pd.DataFrame, frozen_patient: pd.DataFrame
) -> dict[str, int | bool]:
    required = {
        "subject_id",
        "motor_improvement_anchor_observed",
        "motor_improvement_anchor_window",
    }
    missing = sorted(required - set(frozen_patient.columns))
    if missing:
        raise ValueError(f"frozen patient file missing columns: {missing}")
    flag = pd.to_numeric(
        frozen_patient["motor_improvement_anchor_observed"], errors="coerce"
    ).fillna(0)
    expected = frozen_patient.loc[
        flag.eq(1), ["subject_id", "motor_improvement_anchor_window"]
    ].copy()
    merged = landmarks[["subject_id", "target_window"]].merge(
        expected, on="subject_id", how="outer", indicator=True
    )
    window_match = merged["target_window"].astype("string").eq(
        merged["motor_improvement_anchor_window"].astype("string")
    )
    return {
        "derived_n": int(len(landmarks)),
        "frozen_n": int(len(expected)),
        "subject_set_mismatch_n": int(merged["_merge"].ne("both").sum()),
        "window_mismatch_n": int((merged["_merge"].eq("both") & ~window_match).sum()),
        "exact_match": bool(merged["_merge"].eq("both").all() and window_match.all()),
    }


def assert_gate_a(
    cohort: pd.DataFrame,
    landmarks: pd.DataFrame,
    risk_audit: pd.DataFrame,
    expected: GateAExpected = GateAExpected(),
) -> dict[str, int]:
    observed = {
        "parent_n": int(cohort["subject_id"].nunique()),
        "landmark_candidate_n": int(len(landmarks)),
        "risk_set_n": int(risk_audit["risk_set"].sum()),
        "excluded_not_ventilated_n": int((~risk_audit["risk_set"]).sum()),
    }
    target = {
        "parent_n": expected.parent_n,
        "landmark_candidate_n": expected.landmark_candidate_n,
        "risk_set_n": expected.risk_set_n,
        "excluded_not_ventilated_n": expected.excluded_not_ventilated_n,
    }
    if observed["landmark_candidate_n"] != len(risk_audit):
        raise AssertionError("risk audit does not exhaust landmark candidates")
    if observed["risk_set_n"] + observed["excluded_not_ventilated_n"] != observed["landmark_candidate_n"]:
        raise AssertionError("Gate A flow is not exhaustive")
    mismatch = {key: (observed[key], target[key]) for key in target if observed[key] != target[key]}
    if mismatch:
        raise AssertionError(f"GATE_A_FROZEN_COUNT_MISMATCH: {mismatch}")
    return observed
