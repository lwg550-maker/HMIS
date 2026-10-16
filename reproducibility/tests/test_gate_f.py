from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import gate_f


def ts(hours: float) -> pd.Timestamp:
    return pd.Timestamp("2020-01-01") + pd.Timedelta(hours=hours)


def test_day21_cutoff_reclassifies_only_late_successes():
    primary = pd.DataFrame([
        {"subject_id": 1, "outcome": "SUSTAINED_LIBERATION", "event_time": ts(20 * 24), "t_HMIS": ts(0), "hospital_death": pd.NaT, "hospital_discharge": ts(30 * 24), "confirmation_source": "FULL_168H_OBSERVATION"},
        {"subject_id": 2, "outcome": "SUSTAINED_LIBERATION", "event_time": ts(22 * 24), "t_HMIS": ts(0), "hospital_death": pd.NaT, "hospital_discharge": ts(25 * 24), "confirmation_source": "ALIVE_DISCHARGE_SURROGATE"},
        {"subject_id": 3, "outcome": "SUSTAINED_LIBERATION", "event_time": ts(23 * 24), "t_HMIS": ts(0), "hospital_death": ts(26 * 24), "hospital_discharge": ts(26 * 24), "confirmation_source": "FULL_168H_OBSERVATION"},
    ])
    result = gate_f.apply_day21_candidate_cutoff(primary)
    assert result.loc[result.subject_id.eq(1), "outcome"].item() == "SUSTAINED_LIBERATION"
    assert result.loc[result.subject_id.eq(2), "outcome"].item() == "CENSORED_WITHOUT_CONFIRMED_LIBERATION"
    assert result.loc[result.subject_id.eq(2), "followup_h"].item() == 25 * 24
    assert result.loc[result.subject_id.eq(3), "outcome"].item() == "DEATH_BEFORE_LIBERATION"
    assert result.loc[result.subject_id.eq(3), "followup_h"].item() == 26 * 24


def test_merge_short_gaps_only_when_gap_begins_after_landmark():
    intervals = pd.DataFrame([
        {"subject_id": 1, "event_start": ts(-20), "event_end": ts(-10)},
        {"subject_id": 1, "event_start": ts(-7), "event_end": ts(2)},
        {"subject_id": 1, "event_start": ts(5), "event_end": ts(10)},
        {"subject_id": 1, "event_start": ts(17), "event_end": ts(20)},
    ])
    anchors = pd.DataFrame({"subject_id": [1], "t_HMIS": [ts(0)]})
    merged, audit = gate_f.merge_post_landmark_gaps(intervals, anchors, maximum_gap_hours=6)
    observed = list(zip(merged.event_start, merged.event_end))
    assert observed == [(ts(-20), ts(-10)), (ts(-7), ts(10)), (ts(17), ts(20))]
    assert audit["merged_gap_n"] == 1


def test_strict_discharge_rule_preserves_risk_set_and_censors_selected_surrogate():
    outcome = pd.DataFrame([
        {"subject_id": 1, "outcome": "SUSTAINED_LIBERATION", "event_time": ts(10), "t_HMIS": ts(0), "hospital_discharge": ts(20), "confirmation_source": "ALIVE_DISCHARGE_SURROGATE"},
        {"subject_id": 2, "outcome": "SUSTAINED_LIBERATION", "event_time": ts(12), "t_HMIS": ts(0), "hospital_discharge": ts(22), "confirmation_source": "ALIVE_DISCHARGE_SURROGATE"},
    ])
    audit = pd.DataFrame({"subject_id": [1, 2], "strict_action": ["RETAIN_SURROGATE_EVENT", "CENSOR_AT_DISCHARGE"]})
    result = gate_f.apply_discharge_policy(outcome, audit, policy="STRICT_DESTINATION")
    assert len(result) == 2
    assert result.loc[result.subject_id.eq(1), "outcome"].item() == "SUSTAINED_LIBERATION"
    assert result.loc[result.subject_id.eq(2), "outcome"].item() == "CENSORED_WITHOUT_CONFIRMED_LIBERATION"
    assert result.loc[result.subject_id.eq(2), "followup_h"].item() == 22


def test_exclude_all_surrogates_censors_all_eligible_at_discharge():
    outcome = pd.DataFrame([
        {"subject_id": 1, "outcome": "SUSTAINED_LIBERATION", "event_time": ts(10), "t_HMIS": ts(0), "hospital_discharge": ts(20), "confirmation_source": "ALIVE_DISCHARGE_SURROGATE"},
        {"subject_id": 2, "outcome": "SUSTAINED_LIBERATION", "event_time": ts(12), "t_HMIS": ts(0), "hospital_discharge": ts(22), "confirmation_source": "FULL_168H_OBSERVATION"},
    ])
    audit = pd.DataFrame({"subject_id": [1], "strict_action": ["RETAIN_SURROGATE_EVENT"]})
    result = gate_f.apply_discharge_policy(outcome, audit, policy="NO_SURROGATE")
    assert result.loc[result.subject_id.eq(1), "outcome"].item() == "CENSORED_WITHOUT_CONFIRMED_LIBERATION"
    assert result.loc[result.subject_id.eq(2), "outcome"].item() == "SUSTAINED_LIBERATION"


def test_endpoint_rederivation_distinguishes_48h_from_7d_confirmation():
    base = pd.DataFrame([{
        "subject_id": 1, "t_HMIS": ts(0), "landmark_gcsm": 5, "target_window": "W1",
        "hospital_death": pd.NaT, "hospital_discharge": ts(20 * 24),
        "discharge_location": "HOME", "first_tracheostomy": pd.NaT,
    }])
    intervals = pd.DataFrame([
        {"subject_id": 1, "event_start": ts(-2), "event_end": ts(10)},
        {"subject_id": 1, "event_start": ts(82), "event_end": ts(100)},
    ])
    outcome48, _ = gate_f.rederive_endpoint(base, intervals, confirm_hours=48)
    outcome7d, _ = gate_f.rederive_endpoint(base, intervals, confirm_hours=168)
    assert outcome48.outcome.item() == "SUSTAINED_LIBERATION"
    assert outcome48.event_time.item() == ts(10)
    assert outcome7d.outcome.item() == "SUSTAINED_LIBERATION"
    assert outcome7d.event_time.item() == ts(100)


def test_outcome_arrays_treat_early_censoring_as_code_zero():
    outcome = pd.DataFrame([
        {"outcome": "SUSTAINED_LIBERATION", "followup_h": 10},
        {"outcome": "DEATH_BEFORE_LIBERATION", "followup_h": 20},
        {"outcome": "CENSORED_WITHOUT_CONFIRMED_LIBERATION", "followup_h": 30},
    ])
    durations, codes = gate_f.outcome_arrays(outcome)
    assert np.array_equal(durations, [10.0, 20.0, 30.0])
    assert np.array_equal(codes, [1, 2, 0])


def test_fio2_replacement_keeps_13_df_and_reports_per_point_one():
    data = pd.DataFrame({column: [1.0, 2.0] for column in gate_f.PRIMARY_COLUMNS})
    data["auxiliary_fio2"] = [0.4, 0.8]
    matrix, columns = gate_f.fio2_replacement_matrix(data)
    assert len(columns) == 13
    assert "pf_per_50" not in columns
    assert "fio2_per_0_10" in columns
    assert np.array_equal(matrix["fio2_per_0_10"], [4.0, 8.0])


def test_descriptive_strata_are_gcs5_gcs6_and_w1_later():
    data = pd.DataFrame({"landmark_gcsm": [5, 6, 5], "target_window": ["W1", "W2", "W5"]})
    strata = gate_f.descriptive_strata(data)
    assert strata["GCS-M 5"].tolist() == [True, False, True]
    assert strata["GCS-M 6"].tolist() == [False, True, False]
    assert strata["W1"].tolist() == [True, False, False]
    assert strata["W2-W5"].tolist() == [False, True, True]


def test_imputation_frame_rebuilds_outcome_specific_auxiliaries():
    data = pd.DataFrame({
        "subject_id": [1, 2, 3], "x": [1.0, np.nan, 3.0],
        "auxiliary_fio2": [0.4, 0.5, 0.6], "auxiliary_resp_rate": [20, 21, 22],
        "auxiliary_rass": [-2, -1, 0], "auxiliary_creatinine": [1, 1, 1],
        "auxiliary_wbc": [10, 11, 12],
    })
    outcome = pd.DataFrame({
        "subject_id": [1, 2, 3],
        "outcome": ["SUSTAINED_LIBERATION", "DEATH_BEFORE_LIBERATION", "CENSORED_WITHOUT_CONFIRMED_LIBERATION"],
        "followup_h": [10, 20, 30],
    })
    frame, durations, codes = gate_f.build_imputation_frame(data, outcome, ["x"])
    assert frame.event_liberation.tolist() == [1.0, 0.0, 0.0]
    assert frame.event_death.tolist() == [0.0, 1.0, 0.0]
    assert np.array_equal(durations, [10.0, 20.0, 30.0])
    assert np.array_equal(codes, [1, 2, 0])
    assert np.isfinite(frame.na_liberation).all()
    assert np.isfinite(frame.na_death).all()


def test_generic_mi_cox_reports_every_prespecified_parameter():
    rng = np.random.default_rng(12)
    n = 100
    data = pd.DataFrame({
        "subject_id": np.arange(n),
        "x1": rng.normal(size=n),
        "x2": rng.normal(size=n),
        "auxiliary_fio2": rng.uniform(0.3, 0.8, size=n),
        "auxiliary_resp_rate": rng.normal(20, 3, size=n),
        "auxiliary_rass": rng.integers(-4, 2, size=n),
        "auxiliary_creatinine": rng.lognormal(0, 0.2, size=n),
        "auxiliary_wbc": rng.lognormal(2.3, 0.2, size=n),
    })
    data.loc[[2, 5, 7], "x1"] = np.nan
    followup = rng.uniform(1, 100, size=n)
    outcome = pd.DataFrame({
        "subject_id": np.arange(n),
        "outcome": np.where(np.arange(n) < 60, "SUSTAINED_LIBERATION", np.where(np.arange(n) < 80, "DEATH_BEFORE_LIBERATION", "NO_LIBERATION_BY_28D")),
        "followup_h": followup,
    })
    result = gate_f.fit_mi_cause_specific(data, outcome, ["x1", "x2"], imputations=2, iterations=2, seed=91)
    assert result["estimates"].parameter.tolist() == ["x1", "x2"]
    assert np.isfinite(result["estimates"].hazard_ratio).all()
    assert result["diagnostics"]["all_models_converged"]


def test_stratified_aj_has_no_hypothesis_test_columns():
    data = pd.DataFrame({
        "subject_id": [1, 2, 3, 4],
        "landmark_gcsm": [5, 5, 6, 6],
        "target_window": ["W1", "W2", "W1", "W3"],
    })
    outcome = pd.DataFrame({
        "subject_id": [1, 2, 3, 4],
        "outcome": ["SUSTAINED_LIBERATION", "DEATH_BEFORE_LIBERATION", "SUSTAINED_LIBERATION", "NO_LIBERATION_BY_28D"],
        "followup_h": [24, 48, 72, 672],
    })
    result = gate_f.stratified_aj(data, outcome, evaluation_days=[3, 28], replicates=10, seed=4)
    assert set(result.stratum) == {"GCS-M 5", "GCS-M 6", "W1", "W2-W5"}
    assert not any("p" == column.lower() or "p_value" in column.lower() for column in result.columns)


def test_actual_gate_f_outcome_variants_preserve_risk_set_and_frozen_boundaries():
    execution = gate_f.EXECUTION_ROOT
    primary_path = execution / "hmis_liberation_gate_b_evidence_20261012_161516/restricted/evidence_aware_outcome_7d_internal.csv"
    if not primary_path.exists():
        pytest.skip("Credentialed patient-level inputs are intentionally excluded from the public package")
    primary = pd.read_csv(primary_path)
    intervals = pd.read_csv(execution / "hmis_liberation_gate_b_evidence_20261012_161516/restricted/evidence_aware_invasive_intervals_internal.csv")
    audit = pd.read_csv(execution / "hmis_liberation_gate_c_20261012_165736/restricted/gate_c_discharge_surrogate_audit_internal.csv")
    variants, gap_audit = gate_f.assemble_sensitivity_outcomes(primary, intervals, audit)
    assert all(len(frame) == 176 and frame.subject_id.nunique() == 176 for frame in variants.values())
    assert variants["DAY21_CANDIDATE_CUTOFF"].outcome.value_counts().to_dict() == {
        "SUSTAINED_LIBERATION": 125,
        "DEATH_BEFORE_LIBERATION": 40,
        "CENSORED_WITHOUT_CONFIRMED_LIBERATION": 6,
        "NO_LIBERATION_BY_28D": 5,
    }
    assert int(variants["STRICT_DISCHARGE"].outcome.eq("SUSTAINED_LIBERATION").sum()) == 118
    assert int(variants["NO_DISCHARGE_SURROGATE"].outcome.eq("SUSTAINED_LIBERATION").sum()) == 101
    assert int(variants["SUSTAINED_48H"].outcome.eq("SUSTAINED_LIBERATION").sum()) == 145
    assert gap_audit["maximum_gap_hours"] == 6.0


def test_complete_case_cox_uses_only_complete_rows_and_reports_all_parameters():
    rng = np.random.default_rng(13)
    n = 80
    data = pd.DataFrame({"subject_id": np.arange(n), "x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    data.loc[[1, 2], "x1"] = np.nan
    outcome = pd.DataFrame({
        "subject_id": np.arange(n),
        "outcome": np.where(np.arange(n) < 50, "SUSTAINED_LIBERATION", np.where(np.arange(n) < 65, "DEATH_BEFORE_LIBERATION", "NO_LIBERATION_BY_28D")),
        "followup_h": rng.uniform(1, 100, size=n),
    })
    result = gate_f.fit_complete_case_cox(data, outcome, ["x1", "x2"])
    assert result["diagnostics"]["n"] == 78
    assert result["estimates"].parameter.tolist() == ["x1", "x2"]
    assert result["diagnostics"]["all_models_converged"]


def test_markdown_writer_has_no_optional_tabulate_dependency(tmp_path, monkeypatch):
    table = pd.DataFrame({"analysis": ["A"], "hazard_ratio": [1.23456]})
    monkeypatch.setattr(
        pd.DataFrame,
        "to_markdown",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("must not call")),
    )
    destination = tmp_path / "table.md"
    gate_f._write_markdown_table(table, destination)
    text = destination.read_text(encoding="utf-8")
    assert "analysis" in text and "1.235" in text
