import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


MODULE = Path(__file__).parents[1] / "src" / "gate_e.py"
spec = importlib.util.spec_from_file_location("gate_e", MODULE)
gate_e = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(gate_e)


def test_nelson_aalen_uses_full_risk_set_at_tied_times():
    t = np.array([1.0, 2.0, 2.0, 3.0])
    e = np.array([1, 1, 0, 0])
    observed = gate_e.nelson_aalen_at_observed_times(t, e)
    expected = np.array([1 / 4, 1 / 4 + 1 / 3, 1 / 4 + 1 / 3, 1 / 4 + 1 / 3])
    assert np.allclose(observed, expected)


def test_model_matrix_has_exactly_13_frozen_columns():
    row = {name: 0.0 for name in gate_e.MODEL_COLUMNS}
    data = pd.DataFrame([row, row])
    matrix = gate_e.build_model_matrix(data)
    assert matrix.columns.tolist() == gate_e.MODEL_COLUMNS
    assert matrix.shape == (2, 13)


def test_rubin_pool_adds_within_and_between_imputation_variance():
    q = np.array([[0.0, 1.0], [2.0, 3.0]])
    u = np.stack([np.eye(2), np.eye(2)])
    pooled = gate_e.rubin_pool(q, u)
    assert np.allclose(pooled["estimate"], [1.0, 2.0])
    assert np.allclose(pooled["within_variance"], [1.0, 1.0])
    assert np.allclose(pooled["between_variance"], [2.0, 2.0])
    assert np.allclose(np.diag(pooled["total_covariance"]), [4.0, 4.0])


def test_fine_gray_pseudo_time_retains_competing_death_until_horizon():
    duration = np.array([2.0, 4.0, 28.0])
    event = np.array([1, 2, 0])
    pseudo_t, status = gate_e.fine_gray_pseudo_time(duration, event, horizon=28.0)
    assert np.array_equal(pseudo_t, [2.0, 28.0, 28.0])
    assert np.array_equal(status, [1, 0, 0])


def test_fine_gray_pseudo_time_rejects_early_ordinary_censoring():
    with pytest.raises(AssertionError, match="early ordinary censoring"):
        gate_e.fine_gray_pseudo_time(np.array([2.0, 10.0]), np.array([1, 0]), horizon=28.0)


def test_three_knot_rcs_is_zero_below_first_knot_and_rejects_duplicate_knots():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    basis = gate_e.three_knot_rcs_nonlinear(x, np.array([1.0, 2.0, 3.0]))
    assert basis[0] == 0.0
    assert np.isfinite(basis).all()
    with pytest.raises(ValueError, match="strictly increasing"):
        gate_e.three_knot_rcs_nonlinear(x, np.array([1.0, 1.0, 3.0]))


def test_mice_pmm_is_deterministic_preserves_observed_and_donor_values():
    data = pd.DataFrame({
        "x": [1.0, 2.0, np.nan, 4.0, 5.0, np.nan],
        "binary": [0.0, 1.0, 1.0, np.nan, 0.0, 1.0],
        "aux": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    })
    a = gate_e.mice_pmm_impute(data, m=3, iterations=3, seed=77, k_pmm=2)
    b = gate_e.mice_pmm_impute(data, m=3, iterations=3, seed=77, k_pmm=2)
    assert len(a) == 3
    for xa, xb in zip(a, b):
        pd.testing.assert_frame_equal(xa, xb)
        assert not xa.isna().any().any()
        assert set(xa.loc[data.x.isna(), "x"]).issubset(set(data.x.dropna()))
        assert set(xa.loc[data.binary.isna(), "binary"]).issubset({0.0, 1.0})
        assert np.array_equal(xa.loc[data.x.notna(), "x"], data.loc[data.x.notna(), "x"])


def test_cox_and_fine_gray_fits_are_finite_on_synthetic_data():
    rng = np.random.default_rng(4)
    n = 120
    x = rng.normal(size=(n, 2))
    event_time = rng.exponential(np.exp(-0.5 * x[:, 0]))
    death_time = rng.exponential(3.0, size=n)
    horizon = 4.0
    target = (event_time < death_time) & (event_time < horizon)
    death = (death_time < event_time) & (death_time < horizon)
    duration = np.minimum(np.minimum(event_time, death_time), horizon)
    code = np.where(target, 1, np.where(death, 2, 0))
    cox = gate_e.fit_phreg(duration, (code == 1).astype(int), x, robust=False)
    pseudo_t, pseudo_status = gate_e.fine_gray_pseudo_time(duration, code, horizon)
    fg = gate_e.fit_phreg(pseudo_t, pseudo_status, x, robust=True)
    assert np.isfinite(cox["params"]).all() and np.isfinite(cox["covariance"]).all()
    assert np.isfinite(fg["params"]).all() and np.isfinite(fg["covariance"]).all()
    assert cox["gradient_max_abs"] < 1e-4
    assert fg["gradient_max_abs"] < 1e-4


def test_schoenfeld_global_test_returns_valid_probability():
    times = np.arange(1.0, 11.0)
    residuals = np.column_stack([np.linspace(-1, 1, 10), np.sin(times)])
    result = gate_e.schoenfeld_time_test(times, residuals)
    assert 0.0 <= result["global_p"] <= 1.0
    assert result["per_variable_p"].shape == (2,)


def test_influence_summary_counts_patient_level_means_not_dataframe_columns():
    values = pd.DataFrame({
        "subject_id": [1, 1, 2, 2],
        "dfbeta_x": [0.10, 0.20, 0.01, 0.03],
    })
    summary, patient = gate_e.summarize_influence(values, threshold=0.10, pooled_se=np.array([1.0]), parameters=["x"])
    assert summary["patients_over_threshold_n"] == 1
    assert np.isclose(summary["maximum"], 0.15)
    assert "max_abs_pooled_dfbeta" in patient.columns


def test_extended_cox_analytic_score_matches_finite_difference():
    rng = np.random.default_rng(8)
    n = 30
    duration = rng.integers(1, 8, size=n).astype(float)
    status = (rng.random(n) < 0.55).astype(int)
    x = rng.normal(size=(n, 3))
    beta = np.array([0.1, -0.2, 0.05, 0.03])
    ll, score, hessian = gate_e.extended_cox_loglik_score_hessian(beta, duration, status, x, time_varying_index=2)
    eps = 1e-6
    numeric = []
    for j in range(len(beta)):
        step = np.zeros_like(beta); step[j] = eps
        lp = gate_e.extended_cox_loglik_score_hessian(beta + step, duration, status, x, 2)[0]
        lm = gate_e.extended_cox_loglik_score_hessian(beta - step, duration, status, x, 2)[0]
        numeric.append((lp - lm) / (2 * eps))
    assert np.allclose(score, numeric, atol=1e-5)
    assert np.allclose(hessian, hessian.T, atol=1e-10)


def test_extended_cox_time_interaction_converges():
    rng = np.random.default_rng(9)
    n = 100
    x = rng.normal(size=(n, 2))
    duration = rng.exponential(3.0, size=n)
    status = (rng.random(n) < 0.7).astype(int)
    result = gate_e.fit_extended_cox(duration, status, x, time_varying_index=1)
    assert result["converged"]
    assert result["gradient_max_abs"] < 1e-6
    assert result["params"].shape == (3,)


def test_linear_contrast_covariances_have_rubin_pool_shape():
    contrast = np.array([1.0, 2.0])
    covariances = np.stack([np.eye(2), np.eye(2) * 2.0])
    result = gate_e.linear_contrast_covariances(covariances, contrast)
    assert result.shape == (2, 1, 1)
    assert np.allclose(result[:, 0, 0], [5.0, 10.0])
