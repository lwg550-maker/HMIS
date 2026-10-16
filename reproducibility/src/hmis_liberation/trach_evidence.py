from __future__ import annotations

from collections.abc import Mapping

import pandas as pd


STANDARD_INVASIVE_MODES = {
    "(S) CMV", "APRV", "APRV/Biphasic+ApnPress", "APRV/Biphasic+ApnVol",
    "APV (cmv)", "Ambient", "Apnea Ventilation", "CMV", "CMV/ASSIST",
    "CMV/ASSIST/AutoFlow", "CMV/AutoFlow", "CPAP/PPS", "CPAP/PSV",
    "CPAP/PSV+Apn TCPL", "CPAP/PSV+ApnPres", "CPAP/PSV+ApnVol", "MMV",
    "MMV/AutoFlow", "MMV/PSV", "MMV/PSV/AutoFlow", "P-CMV", "PCV+",
    "PCV+/PSV", "PCV+Assist", "PRES/AC", "PRVC/AC", "PRVC/SIMV",
    "PSV/SBT", "SIMV", "SIMV/AutoFlow", "SIMV/PRES", "SIMV/PSV",
    "SIMV/PSV/AutoFlow", "SIMV/VOL", "SYNCHRON MASTER", "SYNCHRON SLAVE",
    "VOL/AC",
}
HAMILTON_INVASIVE_MODES = {
    "APRV", "APV (cmv)", "Ambient", "(S) CMV", "P-CMV", "SIMV",
    "APV (simv)", "P-SIMV", "VS", "ASV",
}
STANDBY_MODES = {"Standby"}
STRONG_PARAMETER_COLUMNS = {
    "respiratory_rate_set", "tidal_volume_set", "plateau_pressure", "peep"
}
SUPPORTIVE_PARAMETER_COLUMNS = {
    "respiratory_rate_total", "respiratory_rate_spontaneous", "minute_volume",
    "tidal_volume_observed", "tidal_volume_spontaneous", "fio2", "flow_rate",
}


def _text(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _positive_number(value: object) -> bool:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return bool(pd.notna(parsed) and float(parsed) > 0)


def ventilator_evidence_flags(row: Mapping[str, object]) -> dict[str, bool]:
    mode = _text(row.get("ventilator_mode"))
    hamilton = _text(row.get("ventilator_mode_hamilton"))
    ventilator_type = _text(row.get("ventilator_type"))
    explicit = mode in STANDARD_INVASIVE_MODES or hamilton in HAMILTON_INVASIVE_MODES
    standby = mode in STANDBY_MODES or hamilton in STANDBY_MODES
    any_mode = mode is not None or hamilton is not None
    strong = any(_positive_number(row.get(column)) for column in STRONG_PARAMETER_COLUMNS)
    supportive = bool(
        ventilator_type
        or any(_positive_number(row.get(column)) for column in SUPPORTIVE_PARAMETER_COLUMNS)
    )
    return {
        "explicit_invasive_mode": bool(explicit),
        "standby_mode": bool(standby),
        "unclassified_mode": bool(any_mode and not explicit and not standby),
        "strong_parameter_evidence": bool(strong),
        "supportive_parameter_evidence": bool(supportive),
    }


def classify_trach_interval(
    ventilator_rows: pd.DataFrame,
    oxygen_rows: pd.DataFrame,
    interval_start: pd.Timestamp,
    interval_end: pd.Timestamp,
) -> dict[str, object]:
    start = pd.Timestamp(interval_start)
    end = pd.Timestamp(interval_end)
    if end <= start:
        raise ValueError("tracheostomy interval must have positive duration")

    vent = ventilator_rows.copy()
    if len(vent):
        vent["charttime"] = pd.to_datetime(vent["charttime"], errors="coerce")
        vent = vent.loc[vent["charttime"].between(start, end, inclusive="left")]
    oxygen = oxygen_rows.copy()
    if len(oxygen):
        oxygen["charttime"] = pd.to_datetime(oxygen["charttime"], errors="coerce")
        oxygen = oxygen.loc[oxygen["charttime"].between(start, end, inclusive="left")]

    flags = [ventilator_evidence_flags(row) for row in vent.to_dict("records")]
    explicit_n = sum(item["explicit_invasive_mode"] for item in flags)
    strong_n = sum(item["strong_parameter_evidence"] for item in flags)
    supportive_n = sum(item["supportive_parameter_evidence"] for item in flags)
    standby_n = sum(item["standby_mode"] for item in flags)
    unclassified_mode_n = sum(item["unclassified_mode"] for item in flags)
    device_n = int(len(oxygen))

    if explicit_n and standby_n:
        evidence_class = "MIXED_ACTIVE_AND_STANDBY"
    elif explicit_n:
        evidence_class = "EXPLICIT_INVASIVE_MODE"
    elif strong_n and standby_n:
        evidence_class = "MIXED_PARAMETER_AND_STANDBY"
    elif strong_n:
        evidence_class = "STRONG_PARAMETER_ONLY"
    elif standby_n and device_n:
        evidence_class = "STANDBY_DEVICE_ONLY"
    elif standby_n:
        evidence_class = "STANDBY_ONLY"
    elif supportive_n or unclassified_mode_n:
        evidence_class = "SUPPORTIVE_OR_UNCLASSIFIED_ONLY"
    elif device_n:
        evidence_class = "DEVICE_ONLY"
    else:
        evidence_class = "NO_RAW_EVIDENCE"

    positive_times = []
    for row, item in zip(vent.to_dict("records"), flags):
        if item["explicit_invasive_mode"] or item["strong_parameter_evidence"]:
            positive_times.append(pd.Timestamp(row["charttime"]))
    positive_times = sorted(set(positive_times))
    positive_span_h = (
        (positive_times[-1] - positive_times[0]).total_seconds() / 3600
        if len(positive_times) >= 2
        else 0.0 if len(positive_times) == 1 else float("nan")
    )
    maximum_positive_gap_h = (
        max(
            (right - left).total_seconds() / 3600
            for left, right in zip(positive_times, positive_times[1:])
        )
        if len(positive_times) >= 2
        else float("nan")
    )
    return {
        "evidence_class": evidence_class,
        "strict_ppv_positive": bool(explicit_n),
        "broad_ppv_positive": bool(explicit_n or strong_n),
        "explicit_invasive_mode_point_n": int(explicit_n),
        "strong_parameter_point_n": int(strong_n),
        "supportive_parameter_point_n": int(supportive_n),
        "standby_point_n": int(standby_n),
        "unclassified_mode_point_n": int(unclassified_mode_n),
        "trach_device_point_n": int(device_n),
        "positive_point_n": int(len(positive_times)),
        "first_positive_time": positive_times[0] if positive_times else pd.NaT,
        "last_positive_time": positive_times[-1] if positive_times else pd.NaT,
        "positive_span_h": positive_span_h,
        "maximum_positive_gap_h": maximum_positive_gap_h,
        "interval_duration_h": (end - start).total_seconds() / 3600,
    }
