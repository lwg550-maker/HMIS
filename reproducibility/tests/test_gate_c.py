from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_gate_c_runtime_sql import build_values  # noqa: E402
from gate_c import (  # noqa: E402
    PRIMARY_COLUMNS,
    audit_discharge_surrogates,
    covariate_audits,
    prepare_analysis,
    strict_destination_class,
)


def test_destination_rules_are_conservative():
    assert strict_destination_class("HOME") == "EXPLICIT_RETAIN_DESTINATION"
    assert strict_destination_class("REHAB") == "EXPLICIT_RETAIN_DESTINATION"
    assert strict_destination_class("CHRONIC/LONG TERM ACUTE CARE") == "TRANSFER_CENSOR_DESTINATION"
    assert strict_destination_class("ACUTE HOSPITAL") == "TRANSFER_CENSOR_DESTINATION"
    assert strict_destination_class("HOSPICE") == "CONSERVATIVE_CENSOR_DESTINATION"
    assert strict_destination_class("SKILLED NURSING FACILITY") == "CONSERVATIVE_CENSOR_DESTINATION"


def test_discharge_audit_requires_clear_destination_and_no_active_interval():
    base_time = pd.Timestamp("2020-01-01")
    locations = ["HOME"] * 29 + ["CHRONIC/LONG TERM ACUTE CARE"]
    outcome = pd.DataFrame(
        {
            "subject_id": np.arange(1, 31),
            "confirmation_source": "ALIVE_DISCHARGE_SURROGATE",
            "event_time": base_time + pd.Timedelta(hours=1),
            "hospital_discharge": base_time + pd.Timedelta(hours=24),
            "hospital_death": pd.NaT,
            "t_HMIS": base_time,
            "discharge_location": locations,
        }
    )
    candidates = pd.DataFrame(
        {
            "subject_id": np.arange(1, 31),
            "candidate_stop": base_time + pd.Timedelta(hours=1),
            "confirmation_end": base_time + pd.Timedelta(hours=169),
            "candidate_result": "QUALIFIED_SUCCESS",
            "confirmation_source": "ALIVE_DISCHARGE_SURROGATE",
            "next_invasive_start": pd.NaT,
            "hospital_discharge": base_time + pd.Timedelta(hours=24),
        }
    )
    intervals = pd.DataFrame(
        {
            "subject_id": np.arange(1, 31),
            "event_start": base_time - pd.Timedelta(hours=2),
            "event_end": base_time + pd.Timedelta(hours=1),
        }
    )
    audit, _ = audit_discharge_surrogates(outcome, candidates, intervals)
    assert audit.strict_action.value_counts().to_dict() == {
        "RETAIN_SURROGATE_EVENT": 29,
        "CENSOR_AT_DISCHARGE": 1,
    }


def synthetic_raw(n: int = 176) -> pd.DataFrame:
    idx = np.arange(n)
    phenotypes = np.array(["SAH", "ICH", "SDH_EDH", "TBI"])[idx % 4]
    years = np.array(["2008 - 2010", "2011 - 2013", "2014 - 2016", "2017 - 2019", "2020 - 2022"])[idx % 5]
    frame = pd.DataFrame(
        {
            "subject_id": idx + 1,
            "hadm_id": idx + 101,
            "stay_id": idx + 201,
            "age": 18 + idx % 70,
            "recorded_sex": np.where(idx % 2, "M", "F"),
            "diagnosis_phenotype": phenotypes,
            "current_invasive_segment_h": 1 + idx,
            "landmark_gcsm": np.where(idx % 3, 5, 6),
            "pupil_bilateral_reactive": np.where(idx % 4, 1, 0),
            "major_sedative_active": idx % 2,
            "any_vasopressor_exposed_6h": (idx // 2) % 2,
            "pf_ratio": 80 + idx,
            "peep": 5 + idx % 8,
            "anchor_year_group": years,
            "pupil_staleness_h": 1.0,
            "pf_pair_staleness_h": 2.0,
            "peep_staleness_h": 1.0,
            "pf_pair_gap_min": 30.0,
            "auxiliary_fio2": 0.4,
            "auxiliary_resp_rate": 18,
            "auxiliary_rass": -2,
            "auxiliary_creatinine": 1.0,
            "auxiliary_wbc": 10.0,
            "propofol_active": idx % 2,
            "dexmedetomidine_active": 0,
            "midazolam_active": 0,
            "lorazepam_active": 0,
            "ketamine_active": 0,
            "active_sedative_drug_n": idx % 2,
        }
    )
    return frame


def test_13_df_matrix_is_constructed_without_outcome_information():
    raw = synthetic_raw()
    outcome = pd.DataFrame(
        {
            "subject_id": np.arange(1, 177),
            "outcome": ["SUSTAINED_LIBERATION"] * 176,
            "event_time": "2020-01-02",
            "followup_h": 24,
            "confirmation_source": "FULL_168H_OBSERVATION",
        }
    )
    data = prepare_analysis(raw, outcome)
    assert len(PRIMARY_COLUMNS) == 13
    assert data[PRIMARY_COLUMNS].shape == (176, 13)
    _, _, _, report = covariate_audits(data)
    assert report["primary_predictor_df"] == 13
    assert report["expected_rank_with_intercept"] == 14


def test_seed_values_keep_actual_hmis_and_active_segment():
    risk = pd.DataFrame(
        {
            "subject_id": [1], "hadm_id": [2], "stay_id": [3],
            "t_HMIS": ["2020-01-01 04:00:00"], "landmark_gcsm": [5],
            "target_window": ["W1"],
            "active_vent_start": ["2019-12-31 20:00:00"],
            "active_vent_end": ["2020-01-02 08:00:00"],
        }
    )
    text = build_values(risk)
    assert "2020-01-01 04:00:00" in text
    assert "2019-12-31 20:00:00" in text
    assert "W1" in text


def test_sql_contract_is_read_only_and_uses_frozen_windows():
    sql = (ROOT / "sql" / "03_extract_gate_c_covariates_template.sql").read_text(encoding="utf-8")
    required = [
        "default_transaction_read_only = on",
        "t_hmis-interval '4 hours'",
        "t_hmis-interval '6 hours'",
        "t_hmis-interval '24 hours'",
        "<=7200",
        "active_vent_start",
        "anchor_year_group",
    ]
    for token in required:
        assert token in sql.lower()
    forbidden = ["create table", "insert into", "update ", "delete from", "copy "]
    lowered = sql.lower()
    for token in forbidden:
        assert token not in lowered
