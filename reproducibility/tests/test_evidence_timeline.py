from __future__ import annotations

import pandas as pd

from evidence_timeline import (
    ALL_TRACH_UPPER,
    EVIDENCE_AWARE,
    HISTORICAL,
    build_definition_events,
    classify_charttime,
    positive_subintervals,
    restrict_events_to_risk_followup,
)


T0 = pd.Timestamp("2200-01-01 00:00:00")


def row(hour: float, **values):
    base = {
        "charttime": T0 + pd.Timedelta(hours=hour),
        "source_type": "VENTILATOR_SETTING_FULL",
        "ventilator_mode": None,
        "ventilator_mode_hamilton": None,
        "ventilator_type": None,
        "respiratory_rate_set": None,
        "respiratory_rate_total": None,
        "respiratory_rate_spontaneous": None,
        "minute_volume": None,
        "tidal_volume_set": None,
        "tidal_volume_observed": None,
        "tidal_volume_spontaneous": None,
        "plateau_pressure": None,
        "peep": None,
        "fio2": None,
        "flow_rate": None,
        "o2_delivery_device_1": None,
        "o2_delivery_device_2": None,
        "o2_delivery_device_3": None,
        "o2_delivery_device_4": None,
    }
    base.update(values)
    return base


def test_standby_overrides_mode_and_parameter_at_same_charttime() -> None:
    records = pd.DataFrame(
        [
            row(1, ventilator_mode="Standby", peep=5),
            row(1, ventilator_mode_hamilton="Ambient"),
        ]
    )
    assert classify_charttime(records) == "OFF"


def test_active_mode_overrides_mask_when_no_standby() -> None:
    records = pd.DataFrame(
        [
            row(1, ventilator_mode="CPAP/PSV"),
            row(
                1,
                source_type="OXYGEN_DELIVERY_FULL",
                o2_delivery_device_1="Trach mask",
            ),
        ]
    )
    assert classify_charttime(records) == "ACTIVE"


def test_mask_or_tpiece_is_off_but_tracheostomy_tube_alone_is_unknown() -> None:
    mask = pd.DataFrame(
        [row(1, source_type="OXYGEN_DELIVERY_FULL", o2_delivery_device_1="Trach mask")]
    )
    tube = pd.DataFrame(
        [
            row(
                1,
                source_type="OXYGEN_DELIVERY_FULL",
                o2_delivery_device_1="Tracheostomy tube",
            )
        ]
    )
    assert classify_charttime(mask) == "OFF"
    assert classify_charttime(tube) == "UNKNOWN"


def test_active_point_supports_14h_and_later_off_truncates() -> None:
    records = pd.DataFrame(
        [
            row(2, peep=5),
            row(
                8,
                source_type="OXYGEN_DELIVERY_FULL",
                o2_delivery_device_1="Trach mask",
            ),
        ]
    )
    intervals, audit = positive_subintervals(records, T0, T0 + pd.Timedelta(hours=30))
    assert intervals == [(T0 + pd.Timedelta(hours=2), T0 + pd.Timedelta(hours=8))]
    assert audit["active_point_n"] == 1
    assert audit["off_point_n"] == 1


def test_later_active_restarts_and_overlapping_14h_support_merges() -> None:
    records = pd.DataFrame(
        [row(1, peep=5), row(5, peep=5), row(10, ventilator_mode="Standby"), row(12, peep=5)]
    )
    intervals, _ = positive_subintervals(records, T0, T0 + pd.Timedelta(hours=40))
    assert intervals == [
        (T0 + pd.Timedelta(hours=1), T0 + pd.Timedelta(hours=10)),
        (T0 + pd.Timedelta(hours=12), T0 + pd.Timedelta(hours=26)),
    ]


def test_three_definitions_preserve_expected_interval_bounds() -> None:
    events = pd.DataFrame(
        [
            {"subject_id": 1, "hadm_id": 10, "source_stay_id": 100, "event_type": "INVASIVE_VENT", "event_start": T0 - pd.Timedelta(hours=2), "event_end": T0 + pd.Timedelta(hours=2)},
            {"subject_id": 1, "hadm_id": 10, "source_stay_id": 100, "event_type": "TRACHEOSTOMY_STATUS", "event_start": T0 + pd.Timedelta(hours=2), "event_end": T0 + pd.Timedelta(hours=30)},
        ]
    )
    raw = pd.DataFrame(
        [
            {"subject_id": 1, "hadm_id": 10, "stay_id": 100, **row(3, peep=5)},
            {
                "subject_id": 1,
                "hadm_id": 10,
                "stay_id": 100,
                **row(8, source_type="OXYGEN_DELIVERY_FULL", o2_delivery_device_1="Trach mask"),
            },
        ]
    )
    historical, _ = build_definition_events(events, raw, HISTORICAL)
    evidence, _ = build_definition_events(events, raw, EVIDENCE_AWARE)
    upper, _ = build_definition_events(events, raw, ALL_TRACH_UPPER)
    assert list(zip(historical.event_start, historical.event_end)) == [
        (T0 - pd.Timedelta(hours=2), T0 + pd.Timedelta(hours=2))
    ]
    assert list(zip(evidence.event_start, evidence.event_end)) == [
        (T0 - pd.Timedelta(hours=2), T0 + pd.Timedelta(hours=2)),
        (T0 + pd.Timedelta(hours=3), T0 + pd.Timedelta(hours=8)),
    ]
    assert list(zip(upper.event_start, upper.event_end)) == [
        (T0 - pd.Timedelta(hours=2), T0 + pd.Timedelta(hours=30))
    ]


def test_formal_events_are_limited_to_risk_patients_and_anchor_to_day35_trach() -> None:
    risk = pd.DataFrame(
        [{"subject_id": 1, "hadm_id": 10, "t_HMIS": T0, "risk_set": True}]
    )
    events = pd.DataFrame(
        [
            {"subject_id": 1, "hadm_id": 10, "event_type": "COHORT", "event_start": pd.NaT, "event_end": pd.NaT},
            {"subject_id": 1, "hadm_id": 10, "event_type": "TRACHEOSTOMY_STATUS", "event_start": T0 - pd.Timedelta(hours=8), "event_end": T0 - pd.Timedelta(hours=1)},
            {"subject_id": 1, "hadm_id": 10, "event_type": "TRACHEOSTOMY_STATUS", "event_start": T0 - pd.Timedelta(hours=1), "event_end": T0 + pd.Timedelta(hours=2)},
            {"subject_id": 1, "hadm_id": 10, "event_type": "TRACHEOSTOMY_STATUS", "event_start": T0 + pd.Timedelta(days=34), "event_end": T0 + pd.Timedelta(days=36)},
            {"subject_id": 2, "hadm_id": 20, "event_type": "TRACHEOSTOMY_STATUS", "event_start": T0, "event_end": T0 + pd.Timedelta(hours=2)},
        ]
    )
    result = restrict_events_to_risk_followup(events, risk)
    trach = result.loc[result.event_type.eq("TRACHEOSTOMY_STATUS")]
    assert len(trach) == 2
    assert list(trach.event_start) == [T0, T0 + pd.Timedelta(days=34)]
    assert list(trach.event_end) == [T0 + pd.Timedelta(hours=2), T0 + pd.Timedelta(days=35)]
    assert set(result.subject_id) == {1}
