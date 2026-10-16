from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from hmis_liberation.gate_a import merge_exact_intervals
from hmis_liberation.trach_evidence import ventilator_evidence_flags


HISTORICAL = "HISTORICAL_INVASIVEVENT_ONLY"
EVIDENCE_AWARE = "EVIDENCE_AWARE_BEDSIDE_PRIMARY"
ALL_TRACH_UPPER = "ALL_TRACHEOSTOMY_ACTIVE_UPPER_BOUND"
DEFINITIONS = {HISTORICAL, EVIDENCE_AWARE, ALL_TRACH_UPPER}
PERSISTENCE_HOURS = 14
OFF_DEVICES = {"Trach mask", "T-piece"}
DEVICE_COLUMNS = [
    "o2_delivery_device_1",
    "o2_delivery_device_2",
    "o2_delivery_device_3",
    "o2_delivery_device_4",
]


def _clean_text(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    result = str(value).strip()
    return result or None


def classify_charttime(records: pd.DataFrame) -> str:
    """Classify all raw records sharing one exact charttime.

    Standby is an explicit off state and therefore wins over co-recorded
    parameters. Otherwise explicit invasive modes or strong parameters win over
    oxygen-device fields. A tracheostomy tube alone remains unknown.
    """

    if records.empty:
        return "UNKNOWN"
    flags = [ventilator_evidence_flags(row) for row in records.to_dict("records")]
    if any(item["standby_mode"] for item in flags):
        return "OFF"
    if any(
        item["explicit_invasive_mode"] or item["strong_parameter_evidence"]
        for item in flags
    ):
        return "ACTIVE"
    devices: set[str] = set()
    for row in records.to_dict("records"):
        for column in DEVICE_COLUMNS:
            value = _clean_text(row.get(column))
            if value:
                devices.add(value)
    if devices & OFF_DEVICES:
        return "OFF"
    return "UNKNOWN"


def _point_states(
    raw: pd.DataFrame,
    interval_start: pd.Timestamp,
    interval_end: pd.Timestamp,
) -> pd.DataFrame:
    start, end = pd.Timestamp(interval_start), pd.Timestamp(interval_end)
    subset = raw.copy()
    subset["charttime"] = pd.to_datetime(subset["charttime"], errors="coerce")
    subset = subset.loc[
        subset["charttime"].ge(start) & subset["charttime"].lt(end)
    ].copy()
    rows = [
        {"charttime": time, "point_state": classify_charttime(group)}
        for time, group in subset.groupby("charttime", sort=True)
    ]
    return pd.DataFrame(rows, columns=["charttime", "point_state"])


def positive_subintervals(
    raw: pd.DataFrame,
    interval_start: pd.Timestamp,
    interval_end: pd.Timestamp,
    *,
    persistence_hours: int = PERSISTENCE_HOURS,
) -> tuple[list[tuple[pd.Timestamp, pd.Timestamp]], dict[str, object]]:
    start, end = pd.Timestamp(interval_start), pd.Timestamp(interval_end)
    if end <= start:
        raise ValueError("tracheostomy interval must have positive duration")
    points = _point_states(raw, start, end)
    active = list(points.loc[points["point_state"].eq("ACTIVE"), "charttime"])
    off = list(points.loc[points["point_state"].eq("OFF"), "charttime"])
    support: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    persistence = pd.Timedelta(hours=int(persistence_hours))
    for time in active:
        stop = min(pd.Timestamp(time) + persistence, end)
        later_off = [value for value in off if time < value < stop]
        if later_off:
            stop = min(later_off)
        if stop > time:
            support.append((pd.Timestamp(time), pd.Timestamp(stop)))
    merged = merge_exact_intervals(support)
    audit = {
        "point_n": int(len(points)),
        "active_point_n": int(points["point_state"].eq("ACTIVE").sum()),
        "off_point_n": int(points["point_state"].eq("OFF").sum()),
        "unknown_point_n": int(points["point_state"].eq("UNKNOWN").sum()),
        "positive_subinterval_n": int(len(merged)),
        "positive_supported_hours": float(
            sum((right - left).total_seconds() / 3600 for left, right in merged)
        ),
    }
    return merged, audit


def _merge_patient_intervals(rows: Iterable[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    columns = [
        "subject_id",
        "hadm_id",
        "source_stay_id",
        "event_type",
        "event_start",
        "event_end",
        "value_text",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    output: list[dict[str, object]] = []
    for (subject_id, hadm_id), group in frame.groupby(
        ["subject_id", "hadm_id"], sort=False
    ):
        intervals = merge_exact_intervals(zip(group["event_start"], group["event_end"]))
        for start, end in intervals:
            output.append(
                {
                    "subject_id": int(subject_id),
                    "hadm_id": int(hadm_id),
                    "source_stay_id": pd.NA,
                    "event_type": "INVASIVE_VENT",
                    "event_start": start,
                    "event_end": end,
                    "value_text": "EVIDENCE_AWARE_MERGED_INVASIVE",
                }
            )
    return pd.DataFrame(output, columns=columns).sort_values(
        ["subject_id", "event_start", "event_end"]
    ).reset_index(drop=True)


def restrict_events_to_risk_followup(
    events: pd.DataFrame,
    risk_audit: pd.DataFrame,
) -> pd.DataFrame:
    """Keep frozen risk-set patients and clip trach status to anchor-day35.

    Non-tracheostomy events are retained for the selected admissions because
    the frozen endpoint state machine needs pre-anchor invasive membership and
    admission/death/discharge context.
    """

    risk = risk_audit.copy()
    truth = risk["risk_set"].astype(str).str.lower().isin({"true", "1", "yes"})
    risk = risk.loc[truth, ["subject_id", "hadm_id", "t_HMIS"]].copy()
    if risk.duplicated(["subject_id", "hadm_id"]).any():
        raise ValueError("risk audit contains duplicate subject-admission anchors")
    risk["t_HMIS"] = pd.to_datetime(risk["t_HMIS"], errors="raise")
    source = events.merge(risk, on=["subject_id", "hadm_id"], how="inner", validate="many_to_one")
    source["event_start"] = pd.to_datetime(source["event_start"], errors="coerce")
    source["event_end"] = pd.to_datetime(source["event_end"], errors="coerce")
    is_trach = source["event_type"].eq("TRACHEOSTOMY_STATUS")
    horizon = source["t_HMIS"] + pd.Timedelta(days=35)
    relevant_trach = (
        is_trach
        & source["event_end"].gt(source["t_HMIS"])
        & source["event_start"].le(horizon)
    )
    source = source.loc[~is_trach | relevant_trach].copy().reset_index(drop=True)
    trach = source["event_type"].eq("TRACHEOSTOMY_STATUS")
    horizon = source["t_HMIS"] + pd.Timedelta(days=35)
    source.loc[trach, "event_start"] = source.loc[trach, ["event_start", "t_HMIS"]].max(axis=1)
    source.loc[trach, "event_end"] = pd.concat(
        [source.loc[trach, "event_end"], horizon.loc[trach].rename("horizon")], axis=1
    ).min(axis=1)
    return source.drop(columns=["t_HMIS"]).reset_index(drop=True)


def build_definition_events(
    events: pd.DataFrame,
    raw: pd.DataFrame,
    definition: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if definition not in DEFINITIONS:
        raise ValueError(f"unknown measurement definition: {definition}")
    source = events.copy()
    for column in ["event_start", "event_end"]:
        source[column] = pd.to_datetime(source[column], errors="coerce")
    raw = raw.copy()
    raw["charttime"] = pd.to_datetime(raw["charttime"], errors="coerce")
    base = source.loc[source["event_type"].eq("INVASIVE_VENT")]
    rows = [
        {
            "subject_id": int(row.subject_id),
            "hadm_id": int(row.hadm_id),
            "source_stay_id": row.source_stay_id,
            "event_type": "INVASIVE_VENT",
            "event_start": row.event_start,
            "event_end": row.event_end,
            "value_text": HISTORICAL,
        }
        for row in base.itertuples(index=False)
    ]
    audit_rows: list[dict[str, object]] = []
    trach = source.loc[source["event_type"].eq("TRACHEOSTOMY_STATUS")]
    if definition != HISTORICAL:
        for sequence, interval in enumerate(trach.itertuples(index=False), start=1):
            start, end = pd.Timestamp(interval.event_start), pd.Timestamp(interval.event_end)
            matching = raw.loc[
                raw["subject_id"].eq(interval.subject_id)
                & raw["hadm_id"].eq(interval.hadm_id)
            ]
            if pd.notna(interval.source_stay_id):
                matching = matching.loc[matching["stay_id"].eq(interval.source_stay_id)]
            if definition == ALL_TRACH_UPPER:
                additions = [(start, end)]
                matching_times = matching.loc[
                    matching["charttime"].ge(start) & matching["charttime"].lt(end),
                    "charttime",
                ]
                audit = {
                    "point_n": int(matching_times.nunique()),
                    "active_point_n": pd.NA,
                    "off_point_n": pd.NA,
                    "unknown_point_n": pd.NA,
                    "positive_subinterval_n": 1,
                    "positive_supported_hours": (end - start).total_seconds() / 3600,
                }
            else:
                additions, audit = positive_subintervals(matching, start, end)
            for left, right in additions:
                rows.append(
                    {
                        "subject_id": int(interval.subject_id),
                        "hadm_id": int(interval.hadm_id),
                        "source_stay_id": interval.source_stay_id,
                        "event_type": "INVASIVE_VENT",
                        "event_start": left,
                        "event_end": right,
                        "value_text": definition,
                    }
                )
            audit_rows.append(
                {
                    "subject_id": int(interval.subject_id),
                    "hadm_id": int(interval.hadm_id),
                    "source_stay_id": interval.source_stay_id,
                    "trach_interval_sequence_global": int(sequence),
                    "trach_interval_start": start,
                    "trach_interval_end": end,
                    "definition": definition,
                    **audit,
                }
            )
    return _merge_patient_intervals(rows), pd.DataFrame(audit_rows)


def replace_invasive_events(
    original_events: pd.DataFrame,
    invasive_intervals: pd.DataFrame,
) -> pd.DataFrame:
    retained = original_events.loc[
        ~original_events["event_type"].eq("INVASIVE_VENT")
    ].copy()
    intervals = invasive_intervals.copy()
    for column in retained.columns:
        if column not in intervals:
            intervals[column] = pd.NA
    intervals = intervals[retained.columns]
    return pd.concat([retained, intervals], ignore_index=True, sort=False)
