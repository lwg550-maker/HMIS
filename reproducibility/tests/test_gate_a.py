from __future__ import annotations

import numpy as np
import pandas as pd

from hmis_liberation.gate_a import (
    build_checkpoint_audit,
    classify_risk_set,
    derive_landmarks,
)


T0 = pd.Timestamp("2200-01-01 00:00:00")


def cohort() -> pd.DataFrame:
    return pd.DataFrame(
        [{
            "subject_id": 1,
            "hadm_id": 11,
            "stay_id": 111,
            "t0": T0,
            "outtime": T0 + pd.Timedelta(days=10),
            "deathtime": pd.NaT,
        }]
    )


def event(event_type: str, *, chart=None, start=None, end=None, value=np.nan,
          store=None, itemid=np.nan, rate=np.nan) -> dict[str, object]:
    return {
        "subject_id": 1,
        "event_type": event_type,
        "event_start": start,
        "event_end": end,
        "charttime": chart,
        "storetime": store,
        "itemid": itemid,
        "value_num": value,
        "rate": rate,
    }


def sentinel() -> dict[str, object]:
    return event("COHORT")


def test_left_open_right_closed_and_single_assignment() -> None:
    rows = [sentinel()]
    rows += [
        event("GCS_M", chart=T0 - pd.Timedelta(hours=4), value=1),
        event("GCS_M", chart=T0, value=2),
        event("GCS_M", chart=T0 + pd.Timedelta(hours=4), value=5),
        event("GCS_M", chart=T0 + pd.Timedelta(hours=20), value=3),
        event("GCS_M", chart=T0 + pd.Timedelta(hours=24), value=4),
        event("GCS_M", chart=T0 + pd.Timedelta(hours=44), value=5),
        event("GCS_M", chart=T0 + pd.Timedelta(hours=48), value=6),
    ]
    audit = build_checkpoint_audit(cohort(), pd.DataFrame(rows)).set_index("window_id")
    assert audit.loc["B0", "raw_gcsm"] == 2
    assert audit.loc["W1", "raw_gcsm"] == 5
    assert audit.loc["W4", "raw_gcsm"] == 4
    assert audit.loc["W5", "raw_gcsm"] == 6
    selected = audit["selected_source_row_number"].dropna()
    assert selected.nunique() == len(selected)


def test_later_storetime_wins_and_unresolved_conflict_is_unknown() -> None:
    rows = [sentinel()]
    at_b0 = T0
    rows += [
        event("GCS_M", chart=at_b0, value=2, store=T0 + pd.Timedelta(minutes=1)),
        event("GCS_M", chart=at_b0, value=4, store=None),
        event("GCS_M", chart=at_b0, value=3, store=T0 + pd.Timedelta(minutes=2)),
        event("GCS_M", chart=T0 + pd.Timedelta(hours=4), value=5, store=T0 + pd.Timedelta(hours=5)),
        event("GCS_M", chart=T0 + pd.Timedelta(hours=4), value=6, store=T0 + pd.Timedelta(hours=5)),
    ]
    audit = build_checkpoint_audit(cohort(), pd.DataFrame(rows)).set_index("window_id")
    assert audit.loc["B0", "raw_gcsm"] == 3
    assert audit.loc["B0", "motor_state"] == "LOW"
    assert audit.loc["W1", "same_timestamp_unresolved_conflict"]
    assert audit.loc["W1", "motor_state"] == "UNKNOWN"


def test_nmb_boundary_and_unknown_prevents_skipping() -> None:
    rows = [sentinel()]
    rows += [
        event("GCS_M", chart=T0, value=3),
        event("GCS_M", chart=T0 + pd.Timedelta(hours=4), value=5),
        event("GCS_M", chart=T0 + pd.Timedelta(hours=8), value=6),
        event(
            "NMB_INFUSION",
            start=T0 + pd.Timedelta(hours=4),
            end=T0 + pd.Timedelta(hours=8),
            itemid=221555,
            rate=1,
        ),
    ]
    audit = build_checkpoint_audit(cohort(), pd.DataFrame(rows)).set_index("window_id")
    assert audit.loc["W1", "nmb_active_at_checkpoint"]
    assert audit.loc["W1", "motor_state"] == "UNKNOWN"
    assert not audit.loc["W2", "nmb_active_at_checkpoint"]
    assert audit.loc["W2", "motor_state"] == "HIGH"
    assert derive_landmarks(audit.reset_index()).empty


def test_actual_record_time_and_vent_left_closed_right_open() -> None:
    rows = [sentinel()]
    rows += [
        event("GCS_M", chart=T0 - pd.Timedelta(minutes=20), value=3),
        event("GCS_M", chart=T0 + pd.Timedelta(hours=3), value=5),
        event(
            "INVASIVE_VENT",
            start=T0 + pd.Timedelta(hours=3),
            end=T0 + pd.Timedelta(hours=9),
        ),
    ]
    checkpoints = build_checkpoint_audit(cohort(), pd.DataFrame(rows))
    landmarks = derive_landmarks(checkpoints)
    assert len(landmarks) == 1
    assert landmarks.iloc[0]["t_HMIS"] == T0 + pd.Timedelta(hours=3)
    assert classify_risk_set(landmarks, pd.DataFrame(rows)).iloc[0]["risk_set"]

    rows[-1]["event_end"] = T0 + pd.Timedelta(hours=3)
    rows[-1]["event_start"] = T0 - pd.Timedelta(hours=1)
    excluded = classify_risk_set(landmarks, pd.DataFrame(rows)).iloc[0]
    assert not excluded["risk_set"]
    assert excluded["risk_set_reason"] == "AT_INVASIVE_INTERVAL_END_BOUNDARY"
