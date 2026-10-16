from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .gate_a import merge_exact_intervals


FOLLOWUP_HOURS = 28 * 24
CONFIRM_7D_HOURS = 7 * 24
CONFIRM_48H_HOURS = 48


@dataclass(frozen=True)
class GateBExpected:
    risk_set_n: int = 176
    sustained_7d_n: int = 131
    death_before_success_n: int = 40
    no_success_n: int = 5
    full_168h_confirmation_n: int = 101
    alive_discharge_surrogate_n: int = 30
    late_start_after_day21_n: int = 6
    sustained_48h_n: int = 145
    reinvasion_within_48h_n: int = 44
    reinvasion_within_7d_n: int = 61


def _valid_time(value: object) -> pd.Timestamp | pd.NaT:
    return pd.to_datetime(value, errors="coerce")


def _first_time(values: pd.Series) -> pd.Timestamp | pd.NaT:
    parsed = pd.to_datetime(values, errors="coerce").dropna()
    return parsed.min() if len(parsed) else pd.NaT


def _subject_context(subject_events: pd.DataFrame) -> dict[str, object]:
    death_column = (
        "hospital_deathtime"
        if "hospital_deathtime" in subject_events.columns
        else "deathtime"
    )
    death = _first_time(subject_events.get(death_column, pd.Series(dtype="object")))
    discharge = _first_time(subject_events.get("dischtime", pd.Series(dtype="object")))
    invasive = subject_events.loc[subject_events["event_type"].eq("INVASIVE_VENT")].copy()
    invasive["event_start"] = pd.to_datetime(invasive["event_start"], errors="coerce")
    invasive["event_end"] = pd.to_datetime(invasive["event_end"], errors="coerce")
    intervals = merge_exact_intervals(zip(invasive["event_start"], invasive["event_end"]))
    trach = subject_events.loc[subject_events["event_type"].eq("TRACHEOSTOMY")]
    first_trach = _first_time(trach["event_start"]) if len(trach) else pd.NaT
    discharge_location = ""
    if "discharge_location" in subject_events.columns:
        nonmissing = subject_events["discharge_location"].dropna().astype(str)
        if len(nonmissing):
            discharge_location = nonmissing.iloc[0]
    return {
        "death": death,
        "discharge": discharge,
        "intervals": intervals,
        "first_tracheostomy": first_trach,
        "discharge_location": discharge_location,
    }


def audit_tracheostomy_measurement(
    risk_audit: pd.DataFrame,
    events: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Audit derived tracheostomy intervals without assigning ventilatory state.

    ``mimiciv_derived.ventilation='Tracheostomy'`` may represent a patient with
    or without positive-pressure ventilation.  The frozen endpoint distinguishes
    these states, so this function identifies the affected intervals and raw
    respiratory evidence but deliberately does not reclassify them.
    """

    risk = risk_audit.loc[risk_audit["risk_set"], ["subject_id", "t_HMIS"]].copy()
    risk["t_HMIS"] = pd.to_datetime(risk["t_HMIS"], errors="raise")
    trach = events.loc[events["event_type"].eq("TRACHEOSTOMY_STATUS")].copy()
    evidence = events.loc[
        events["event_type"].isin(
            [
                "VENTILATOR_SETTING",
                "TRACH_OXYGEN_DEVICE",
                "TRACH_PPV_EVIDENCE",
                "TRACH_NO_MODE_EVIDENCE",
            ]
        )
    ].copy()
    for frame in (trach, evidence):
        frame["event_start"] = pd.to_datetime(frame["event_start"], errors="coerce")
        frame["event_end"] = pd.to_datetime(frame["event_end"], errors="coerce")

    rows: list[dict[str, object]] = []
    evidence_groups = {
        int(subject_id): group.copy()
        for subject_id, group in evidence.groupby("subject_id", sort=False)
    }
    for patient in risk.itertuples(index=False):
        anchor = pd.Timestamp(patient.t_HMIS)
        maximum = anchor + pd.Timedelta(days=35)
        patient_trach = trach.loc[trach["subject_id"].eq(patient.subject_id)]
        patient_evidence = evidence_groups.get(int(patient.subject_id), evidence.iloc[0:0])
        for interval in patient_trach.itertuples(index=False):
            start = pd.Timestamp(interval.event_start)
            end = pd.Timestamp(interval.event_end)
            if pd.isna(start) or pd.isna(end) or end <= anchor or start > maximum:
                continue
            ev = patient_evidence.loc[
                patient_evidence["event_start"].between(start, end, inclusive="both")
            ]
            ppv_n = int(
                ev["event_type"].isin(["VENTILATOR_SETTING", "TRACH_PPV_EVIDENCE"]).sum()
            )
            no_mode_n = int(
                ev["event_type"].isin(["TRACH_OXYGEN_DEVICE", "TRACH_NO_MODE_EVIDENCE"]).sum()
            )
            rows.append(
                {
                    "subject_id": int(patient.subject_id),
                    "t_HMIS": anchor,
                    "tracheostomy_interval_start": start,
                    "tracheostomy_interval_end": end,
                    "ppv_evidence_points": ppv_n,
                    "no_mode_evidence_points": no_mode_n,
                    "measurement_class": (
                        "DEFINITE_PPV_PRESENT_BUT_INTERVAL_BOUNDARIES_UNRESOLVED"
                        if ppv_n
                        else "TRACHEOSTOMY_STATUS_UNRESOLVED"
                    ),
                }
            )
    detail = pd.DataFrame(rows)
    summary = {
        "risk_set_n": int(len(risk)),
        "tracheostomy_status_interval_n": int(len(detail)),
        "patients_with_tracheostomy_status_n": int(
            detail["subject_id"].nunique() if len(detail) else 0
        ),
        "intervals_with_ppv_evidence_n": int(
            detail["ppv_evidence_points"].gt(0).sum() if len(detail) else 0
        ),
        "patients_with_ppv_evidence_n": int(
            detail.loc[detail["ppv_evidence_points"].gt(0), "subject_id"].nunique()
            if len(detail)
            else 0
        ),
        "unresolved_interval_n": int(len(detail)),
    }
    return detail, summary


def derive_patient_endpoint(
    *,
    subject_id: int,
    anchor: pd.Timestamp,
    intervals: list[tuple[pd.Timestamp, pd.Timestamp]],
    death: pd.Timestamp | pd.NaT,
    discharge: pd.Timestamp | pd.NaT,
    confirm_hours: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Apply the frozen sustained-liberation state machine to one patient.

    The event time is the qualifying candidate stop, not its future confirmation
    time. Information after day 28 may only confirm or reject a candidate that
    started on or before day 28.
    """

    anchor = pd.Timestamp(anchor)
    death = _valid_time(death)
    discharge = _valid_time(discharge)
    horizon = anchor + pd.Timedelta(hours=FOLLOWUP_HOURS)
    confirm_delta = pd.Timedelta(hours=int(confirm_hours))

    active_indices = [
        index for index, (start, end) in enumerate(intervals) if start <= anchor < end
    ]
    if len(active_indices) != 1:
        raise ValueError(
            f"subject {subject_id}: anchor must lie in exactly one merged invasive interval"
        )
    active_index = active_indices[0]
    candidates: list[dict[str, object]] = []
    eligible_indices = [
        index
        for index in range(active_index, len(intervals))
        if intervals[index][1] > anchor and intervals[index][1] <= horizon
    ]

    first_candidate_index = eligible_indices[0] if eligible_indices else None
    first_stop = (
        intervals[first_candidate_index][1]
        if first_candidate_index is not None
        else pd.NaT
    )
    first_reinvasion = (
        intervals[first_candidate_index + 1][0]
        if first_candidate_index is not None and first_candidate_index + 1 < len(intervals)
        else pd.NaT
    )
    reinvasion_before_death = pd.isna(death) or (
        pd.notna(first_reinvasion) and first_reinvasion <= death
    )
    reinvasion_48 = bool(
        pd.notna(first_reinvasion)
        and reinvasion_before_death
        and first_reinvasion <= first_stop + pd.Timedelta(hours=48)
    )
    reinvasion_7d = bool(
        pd.notna(first_reinvasion)
        and reinvasion_before_death
        and first_reinvasion <= first_stop + pd.Timedelta(hours=168)
    )

    failed_candidate_n = 0
    for sequence, index in enumerate(eligible_indices, start=1):
        cessation = intervals[index][1]
        if cessation <= anchor:
            raise AssertionError("candidate cessation must be strictly after t_HMIS")
        if pd.notna(death) and death <= cessation:
            if death <= horizon:
                outcome = {
                    "subject_id": subject_id,
                    "outcome": "DEATH_BEFORE_LIBERATION",
                    "event_time": death,
                    "followup_h": (death - anchor).total_seconds() / 3600,
                    "confirmation_source": "NONE",
                    "candidate_n": len(candidates),
                    "failed_candidate_n": failed_candidate_n,
                    "first_candidate_stop": first_stop,
                    "first_reinvasion": first_reinvasion,
                    "reinvasion_within_48h_of_first_stop": reinvasion_48,
                    "reinvasion_within_7d_of_first_stop": reinvasion_7d,
                    "late_start_after_day21": False,
                }
                return outcome, candidates
            break
        if pd.notna(discharge) and discharge <= cessation:
            break

        confirm_end = cessation + confirm_delta
        next_start = intervals[index + 1][0] if index + 1 < len(intervals) else pd.NaT
        reinvasion_in_window = bool(
            pd.notna(next_start)
            and next_start <= confirm_end
            and (pd.isna(discharge) or next_start <= discharge)
        )
        death_in_window = bool(
            pd.notna(death) and cessation <= death <= confirm_end
        )

        invalidator = ""
        invalidator_time = pd.NaT
        if reinvasion_in_window and death_in_window:
            if death <= next_start:
                invalidator, invalidator_time = "DEATH", death
            else:
                invalidator, invalidator_time = "REINVASION", next_start
        elif death_in_window:
            invalidator, invalidator_time = "DEATH", death
        elif reinvasion_in_window:
            invalidator, invalidator_time = "REINVASION", next_start

        candidate_row = {
            "subject_id": subject_id,
            "candidate_sequence": sequence,
            "candidate_stop": cessation,
            "confirmation_end": confirm_end,
            "next_invasive_start": next_start,
            "death_time": death,
            "hospital_discharge": discharge,
            "invalidator": invalidator or "NONE",
            "invalidator_time": invalidator_time,
            "candidate_result": "",
            "confirmation_source": "NONE",
            "candidate_after_day21": bool(
                cessation > anchor + pd.Timedelta(days=21)
            ),
        }

        if invalidator:
            candidate_row["candidate_result"] = "FAILED_" + invalidator
            candidates.append(candidate_row)
            failed_candidate_n += 1
            if invalidator == "DEATH":
                if death <= horizon:
                    outcome = {
                        "subject_id": subject_id,
                        "outcome": "DEATH_BEFORE_LIBERATION",
                        "event_time": death,
                        "followup_h": (death - anchor).total_seconds() / 3600,
                        "confirmation_source": "NONE",
                        "candidate_n": len(candidates),
                        "failed_candidate_n": failed_candidate_n,
                        "first_candidate_stop": first_stop,
                        "first_reinvasion": first_reinvasion,
                        "reinvasion_within_48h_of_first_stop": reinvasion_48,
                        "reinvasion_within_7d_of_first_stop": reinvasion_7d,
                        "late_start_after_day21": False,
                    }
                    return outcome, candidates
                break
            continue

        fully_observed = pd.notna(discharge) and discharge >= confirm_end
        alive_discharge_surrogate = (
            pd.notna(discharge)
            and cessation < discharge < confirm_end
            and (pd.isna(death) or death > discharge)
        )
        if fully_observed or alive_discharge_surrogate:
            source = (
                f"FULL_{int(confirm_hours)}H_OBSERVATION"
                if fully_observed
                else "ALIVE_DISCHARGE_SURROGATE"
            )
            candidate_row["candidate_result"] = "QUALIFIED_SUCCESS"
            candidate_row["confirmation_source"] = source
            candidates.append(candidate_row)
            outcome = {
                "subject_id": subject_id,
                "outcome": "SUSTAINED_LIBERATION",
                "event_time": cessation,
                "followup_h": (cessation - anchor).total_seconds() / 3600,
                "confirmation_source": source,
                "candidate_n": len(candidates),
                "failed_candidate_n": failed_candidate_n,
                "first_candidate_stop": first_stop,
                "first_reinvasion": first_reinvasion,
                "reinvasion_within_48h_of_first_stop": reinvasion_48,
                "reinvasion_within_7d_of_first_stop": reinvasion_7d,
                "late_start_after_day21": bool(
                    cessation > anchor + pd.Timedelta(days=21)
                ),
            }
            return outcome, candidates

        candidate_row["candidate_result"] = "UNCONFIRMED_OBSERVATION_ENDED"
        candidates.append(candidate_row)
        failed_candidate_n += 1
        break

    if pd.notna(death) and anchor < death <= horizon:
        outcome_name = "DEATH_BEFORE_LIBERATION"
        event_time = death
        confirmation_source = "NONE"
    else:
        outcome_name = "NO_LIBERATION_BY_28D"
        event_time = min(discharge, horizon) if pd.notna(discharge) else horizon
        confirmation_source = "NONE"
    outcome = {
        "subject_id": subject_id,
        "outcome": outcome_name,
        "event_time": event_time,
        "followup_h": (event_time - anchor).total_seconds() / 3600,
        "confirmation_source": confirmation_source,
        "candidate_n": len(candidates),
        "failed_candidate_n": failed_candidate_n,
        "first_candidate_stop": first_stop,
        "first_reinvasion": first_reinvasion,
        "reinvasion_within_48h_of_first_stop": reinvasion_48,
        "reinvasion_within_7d_of_first_stop": reinvasion_7d,
        "late_start_after_day21": False,
    }
    return outcome, candidates


def derive_endpoint_dataset(
    risk_audit: pd.DataFrame,
    events: pd.DataFrame,
    *,
    confirm_hours: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    risk = risk_audit.loc[risk_audit["risk_set"]].copy()
    if risk["subject_id"].duplicated().any():
        raise ValueError("risk set contains duplicate patients")
    groups = {int(k): v.copy() for k, v in events.groupby("subject_id", sort=False)}
    outcomes: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    for row in risk.itertuples(index=False):
        subject_id = int(row.subject_id)
        if subject_id not in groups:
            raise ValueError(f"missing events for risk-set subject {subject_id}")
        context = _subject_context(groups[subject_id])
        outcome, candidates = derive_patient_endpoint(
            subject_id=subject_id,
            anchor=pd.Timestamp(row.t_HMIS),
            intervals=context["intervals"],
            death=context["death"],
            discharge=context["discharge"],
            confirm_hours=confirm_hours,
        )
        outcome.update(
            {
                "t_HMIS": pd.Timestamp(row.t_HMIS),
                "landmark_gcsm": int(row.landmark_gcsm),
                "target_window": row.target_window,
                "first_tracheostomy": context["first_tracheostomy"],
                "tracheostomy_by_28d": bool(
                    pd.notna(context["first_tracheostomy"])
                    and context["first_tracheostomy"]
                    <= pd.Timestamp(row.t_HMIS) + pd.Timedelta(days=28)
                ),
                "hospital_discharge": context["discharge"],
                "hospital_death": context["death"],
                "discharge_location": context["discharge_location"],
                "confirm_hours": int(confirm_hours),
            }
        )
        outcomes.append(outcome)
        for candidate in candidates:
            candidate.update(
                {
                    "t_HMIS": pd.Timestamp(row.t_HMIS),
                    "confirm_hours": int(confirm_hours),
                }
            )
            candidate_rows.append(candidate)
    outcome_frame = pd.DataFrame(outcomes).sort_values("subject_id").reset_index(drop=True)
    candidate_frame = pd.DataFrame(candidate_rows)
    if len(outcome_frame) != len(risk):
        raise AssertionError("Gate B outcome derivation does not exhaust risk set")
    if outcome_frame["subject_id"].duplicated().any():
        raise AssertionError("Gate B produced duplicate patient outcomes")
    return outcome_frame, candidate_frame


def summarize_gate_b(
    outcome_7d: pd.DataFrame,
    outcome_48h: pd.DataFrame,
) -> dict[str, int]:
    counts = outcome_7d["outcome"].value_counts().to_dict()
    return {
        "risk_set_n": int(len(outcome_7d)),
        "sustained_7d_n": int(counts.get("SUSTAINED_LIBERATION", 0)),
        "death_before_success_n": int(counts.get("DEATH_BEFORE_LIBERATION", 0)),
        "no_success_n": int(counts.get("NO_LIBERATION_BY_28D", 0)),
        "full_168h_confirmation_n": int(
            outcome_7d["confirmation_source"].eq("FULL_168H_OBSERVATION").sum()
        ),
        "alive_discharge_surrogate_n": int(
            outcome_7d["confirmation_source"].eq("ALIVE_DISCHARGE_SURROGATE").sum()
        ),
        "late_start_after_day21_n": int(outcome_7d["late_start_after_day21"].sum()),
        "sustained_48h_n": int(
            outcome_48h["outcome"].eq("SUSTAINED_LIBERATION").sum()
        ),
        "reinvasion_within_48h_n": int(
            outcome_7d["reinvasion_within_48h_of_first_stop"].sum()
        ),
        "reinvasion_within_7d_n": int(
            outcome_7d["reinvasion_within_7d_of_first_stop"].sum()
        ),
    }


def assert_gate_b(
    outcome_7d: pd.DataFrame,
    outcome_48h: pd.DataFrame,
    expected: GateBExpected = GateBExpected(),
) -> dict[str, int]:
    observed = summarize_gate_b(outcome_7d, outcome_48h)
    target = {
        "risk_set_n": expected.risk_set_n,
        "sustained_7d_n": expected.sustained_7d_n,
        "death_before_success_n": expected.death_before_success_n,
        "no_success_n": expected.no_success_n,
        "full_168h_confirmation_n": expected.full_168h_confirmation_n,
        "alive_discharge_surrogate_n": expected.alive_discharge_surrogate_n,
        "late_start_after_day21_n": expected.late_start_after_day21_n,
        "sustained_48h_n": expected.sustained_48h_n,
        "reinvasion_within_48h_n": expected.reinvasion_within_48h_n,
        "reinvasion_within_7d_n": expected.reinvasion_within_7d_n,
    }
    if observed["sustained_7d_n"] + observed["death_before_success_n"] + observed["no_success_n"] != observed["risk_set_n"]:
        raise AssertionError("Gate B final states are not mutually exhaustive")
    if observed["full_168h_confirmation_n"] + observed["alive_discharge_surrogate_n"] != observed["sustained_7d_n"]:
        raise AssertionError("Gate B success evidence classes do not exhaust events")
    mismatch = {key: (observed[key], target[key]) for key in target if observed[key] != target[key]}
    if mismatch:
        raise AssertionError(f"GATE_B_FROZEN_COUNT_MISMATCH: {mismatch}")
    return observed
