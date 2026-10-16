from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import warnings
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.stats import chi2, rankdata, t as student_t
import statsmodels
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.imputation.mice import MICEData
from statsmodels.nonparametric.smoothers_lowess import lowess


MODEL_COLUMNS = [
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

VARIABLE_LABELS = {
    "age_10": ("Age", "per 10 years"),
    "recorded_sex_male": ("Recorded sex", "male vs female"),
    "phenotype_sah": ("Diagnostic phenotype: SAH", "vs TBI"),
    "phenotype_ich": ("Diagnostic phenotype: ICH", "vs TBI"),
    "phenotype_sdh_edh": ("Diagnostic phenotype: SDH/EDH", "vs TBI"),
    "log2_1_current_invasive_h": ("Current invasive-ventilation duration", "per doubling of 1 + hours"),
    "landmark_gcsm_6": ("Landmark GCS-M", "6 vs 5"),
    "pupil_bilateral_reactive": ("Pupil response", "bilaterally reactive vs any abnormal"),
    "major_sedative_active": ("Major continuous sedative active", "yes vs no"),
    "any_vasopressor_exposed_6h": ("Vasopressor exposure in prior 6 h", "yes vs no"),
    "pf_per_50": ("P/F ratio", "per 50 units"),
    "peep_per_5": ("PEEP", "per 5 cmH2O"),
    "anchor_year_group_ordinal": ("Anchor year group", "per one-category increase"),
}

IMPUTATION_AUXILIARIES = [
    "event_liberation",
    "event_death",
    "log_followup_h",
    "na_liberation",
    "na_death",
    "auxiliary_fio2",
    "auxiliary_resp_rate",
    "auxiliary_rass",
    "auxiliary_creatinine",
    "auxiliary_wbc",
]

INCOMPLETE_EXPECTED = {
    "pupil_bilateral_reactive": 6,
    "pf_per_50": 38,
    "peep_per_5": 1,
    "auxiliary_rass": 29,
    "auxiliary_creatinine": 3,
    "auxiliary_wbc": 3,
}

BLUE = "#2F6B9A"
ORANGE = "#D9782D"
DARK = "#3F4B55"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def nelson_aalen_at_observed_times(durations: np.ndarray, status: np.ndarray) -> np.ndarray:
    durations = np.asarray(durations, dtype=float)
    status = np.asarray(status, dtype=int)
    if durations.ndim != 1 or status.ndim != 1 or len(durations) != len(status):
        raise ValueError("durations and status must be equal-length vectors")
    if np.any(durations < 0) or not set(np.unique(status)).issubset({0, 1}):
        raise ValueError("invalid duration or status")
    event_times = np.sort(np.unique(durations[status == 1]))
    increments = []
    cumulative = 0.0
    for time in event_times:
        at_risk = int(np.sum(durations >= time))
        events = int(np.sum((durations == time) & (status == 1)))
        cumulative += events / at_risk
        increments.append(cumulative)
    if len(event_times) == 0:
        return np.zeros(len(durations))
    idx = np.searchsorted(event_times, durations, side="right") - 1
    out = np.zeros(len(durations))
    valid = idx >= 0
    out[valid] = np.asarray(increments)[idx[valid]]
    return out


def build_model_matrix(data: pd.DataFrame) -> pd.DataFrame:
    missing = [name for name in MODEL_COLUMNS if name not in data.columns]
    if missing:
        raise KeyError(f"missing frozen model columns: {missing}")
    matrix = data.loc[:, MODEL_COLUMNS].astype(float).copy()
    if matrix.isna().any().any():
        raise AssertionError("model matrix contains missing values")
    return matrix


def rubin_pool(estimates: np.ndarray, covariances: np.ndarray) -> dict[str, np.ndarray]:
    q = np.asarray(estimates, dtype=float)
    u = np.asarray(covariances, dtype=float)
    if q.ndim != 2 or u.shape != (q.shape[0], q.shape[1], q.shape[1]) or q.shape[0] < 2:
        raise ValueError("invalid Rubin pooling arrays")
    m = q.shape[0]
    qbar = q.mean(axis=0)
    within = u.mean(axis=0)
    between = np.cov(q, rowvar=False, ddof=1)
    if q.shape[1] == 1:
        between = np.asarray([[float(between)]])
    total = within + (1.0 + 1.0 / m) * between
    wdiag = np.diag(within)
    bdiag = np.diag(between)
    with np.errstate(divide="ignore", invalid="ignore"):
        relative_increase = (1.0 + 1.0 / m) * bdiag / wdiag
        df = (m - 1.0) * (1.0 + 1.0 / relative_increase) ** 2
    df = np.where((bdiag <= 1e-15) | ~np.isfinite(df), np.inf, df)
    fraction_missing = ((1.0 + 1.0 / m) * bdiag) / np.diag(total)
    return {
        "estimate": qbar,
        "within_covariance": within,
        "within_variance": wdiag,
        "between_covariance": between,
        "between_variance": bdiag,
        "total_covariance": total,
        "degrees_freedom": df,
        "fraction_missing_information": fraction_missing,
    }


def linear_contrast_covariances(covariances: np.ndarray, contrast: np.ndarray) -> np.ndarray:
    covariances = np.asarray(covariances, dtype=float)
    contrast = np.asarray(contrast, dtype=float)
    if covariances.ndim != 3 or covariances.shape[1:] != (len(contrast), len(contrast)):
        raise ValueError("invalid covariance stack or contrast")
    values = np.einsum("j,mjk,k->m", contrast, covariances, contrast)
    return values[:, None, None]


def fine_gray_pseudo_time(durations: np.ndarray, event_codes: np.ndarray, horizon: float) -> tuple[np.ndarray, np.ndarray]:
    durations = np.asarray(durations, dtype=float)
    events = np.asarray(event_codes, dtype=int)
    if not set(np.unique(events)).issubset({0, 1, 2}):
        raise ValueError("event code must be 0, 1, or 2")
    if np.any(durations > horizon + 1e-10):
        raise AssertionError("follow-up exceeds the Fine-Gray horizon")
    if np.any((events == 0) & (np.abs(durations - horizon) > 1e-10)):
        raise AssertionError("early ordinary censoring requires IPCW; not present in the frozen primary dataset")
    pseudo = durations.copy()
    pseudo[events != 1] = horizon
    return pseudo, (events == 1).astype(int)


def three_knot_rcs_nonlinear(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    k = np.asarray(knots, dtype=float)
    if k.shape != (3,) or not np.all(np.diff(k) > 0):
        raise ValueError("three knots must be strictly increasing")
    k1, k2, k3 = k
    positive = lambda z: np.maximum(z, 0.0) ** 3
    basis = positive(x - k1)
    basis -= positive(x - k2) * (k3 - k1) / (k3 - k2)
    basis += positive(x - k3) * (k2 - k1) / (k3 - k2)
    return basis / ((k3 - k1) ** 2)


def mice_pmm_impute(data: pd.DataFrame, m: int, iterations: int, seed: int, k_pmm: int = 5) -> list[pd.DataFrame]:
    if m < 1 or iterations < 1:
        raise ValueError("m and iterations must be positive")
    original = data.copy()
    missing_masks = original.isna()
    outputs: list[pd.DataFrame] = []
    for imputation_id in range(1, m + 1):
        np.random.seed(seed + 1009 * imputation_id)
        mice = MICEData(original.copy(), perturbation_method="gaussian", k_pmm=k_pmm)
        mice.update_all(n_iter=iterations)
        completed = mice.data.loc[:, original.columns].copy()
        completed.index = original.index
        for column in original.columns:
            completed.loc[~missing_masks[column], column] = original.loc[~missing_masks[column], column]
            if missing_masks[column].any():
                donor_values = set(original.loc[~missing_masks[column], column].astype(float))
                imputed_values = set(completed.loc[missing_masks[column], column].astype(float))
                if not imputed_values.issubset(donor_values):
                    raise AssertionError(f"PMM imputation for {column} used a non-donor value")
        if completed.isna().any().any():
            raise AssertionError("MICE left missing values")
        outputs.append(completed)
    return outputs


def fit_phreg(durations: np.ndarray, status: np.ndarray, exog: np.ndarray, robust: bool) -> dict:
    durations = np.asarray(durations, dtype=float)
    status = np.asarray(status, dtype=int)
    exog = np.asarray(exog, dtype=float)
    if exog.ndim != 2 or len(durations) != exog.shape[0]:
        raise ValueError("invalid PHReg dimensions")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = PHReg(durations, exog, status=status, ties="efron")
        groups = np.arange(len(durations)) if robust else None
        result = model.fit(groups=groups, method="newton", maxiter=1000, disp=False)
    params = np.asarray(result.params, dtype=float)
    covariance = np.asarray(result.cov_params(), dtype=float)
    gradient = np.asarray(model.score(params), dtype=float)
    warning_messages = [str(x.message) for x in caught]
    if not np.isfinite(params).all() or not np.isfinite(covariance).all():
        raise AssertionError("non-finite PHReg estimate")
    return {
        "params": params,
        "covariance": covariance,
        "gradient_max_abs": float(np.max(np.abs(gradient))),
        "log_likelihood": float(result.llf),
        "warnings": warning_messages,
        "result": result,
        "model": model,
    }


def extended_cox_loglik_score_hessian(
    beta: np.ndarray,
    durations: np.ndarray,
    status: np.ndarray,
    exog: np.ndarray,
    time_varying_index: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Efron partial likelihood for one covariate by log(1+t/24) interaction."""
    beta = np.asarray(beta, dtype=float)
    durations = np.asarray(durations, dtype=float)
    status = np.asarray(status, dtype=int)
    exog = np.asarray(exog, dtype=float)
    if exog.ndim != 2 or len(durations) != exog.shape[0]:
        raise ValueError("invalid extended Cox dimensions")
    if not 0 <= time_varying_index < exog.shape[1] or beta.shape != (exog.shape[1] + 1,):
        raise ValueError("invalid time-varying covariate specification")
    loglik = 0.0
    score = np.zeros_like(beta)
    hessian = np.zeros((len(beta), len(beta)))
    for event_time in np.unique(durations[status == 1]):
        event_mask = (durations == event_time) & (status == 1)
        risk_mask = durations >= event_time
        event_n = int(event_mask.sum())
        time_basis = math.log1p(event_time / 24.0)
        z = np.column_stack([exog, exog[:, time_varying_index] * time_basis])
        z_risk = z[risk_mask]
        z_event = z[event_mask]
        xb_risk = z_risk @ beta
        max_xb = float(np.max(xb_risk))
        risk_weight = np.exp(xb_risk - max_xb)
        event_weight = np.exp((z_event @ beta) - max_xb)
        s0 = float(risk_weight.sum())
        s1 = (risk_weight[:, None] * z_risk).sum(axis=0)
        s2 = np.einsum("i,ij,ik->jk", risk_weight, z_risk, z_risk)
        e0 = float(event_weight.sum())
        e1 = (event_weight[:, None] * z_event).sum(axis=0)
        e2 = np.einsum("i,ij,ik->jk", event_weight, z_event, z_event)
        loglik += float((z_event @ beta).sum())
        score += z_event.sum(axis=0)
        for tie_index in range(event_n):
            fraction = tie_index / event_n
            denominator = s0 - fraction * e0
            if denominator <= 0 or not np.isfinite(denominator):
                raise FloatingPointError("invalid extended Cox risk-set denominator")
            numerator_1 = s1 - fraction * e1
            numerator_2 = s2 - fraction * e2
            mean_z = numerator_1 / denominator
            loglik -= max_xb + math.log(denominator)
            score -= mean_z
            hessian -= numerator_2 / denominator - np.outer(mean_z, mean_z)
    return float(loglik), score, hessian


def fit_extended_cox(
    durations: np.ndarray,
    status: np.ndarray,
    exog: np.ndarray,
    time_varying_index: int,
    initial: np.ndarray | None = None,
) -> dict:
    """Fit the single-interaction extended Cox model with safeguarded Newton steps."""
    exog = np.asarray(exog, dtype=float)
    beta = np.zeros(exog.shape[1] + 1) if initial is None else np.asarray(initial, dtype=float).copy()
    converged = False
    iterations = 0
    for iterations in range(1, 101):
        loglik, score, hessian = extended_cox_loglik_score_hessian(beta, durations, status, exog, time_varying_index)
        if float(np.max(np.abs(score))) < 1e-8:
            converged = True
            break
        information = -hessian
        step = np.linalg.solve(information, score)
        scale = 1.0
        accepted = False
        while scale >= 2 ** -20:
            candidate = beta + scale * step
            candidate_ll = extended_cox_loglik_score_hessian(candidate, durations, status, exog, time_varying_index)[0]
            if np.isfinite(candidate_ll) and candidate_ll >= loglik:
                beta = candidate
                accepted = True
                break
            scale *= 0.5
        if not accepted:
            break
    loglik, score, hessian = extended_cox_loglik_score_hessian(beta, durations, status, exog, time_varying_index)
    if float(np.max(np.abs(score))) < 1e-6:
        converged = True
    covariance = np.linalg.inv(-hessian)
    return {
        "params": beta,
        "covariance": covariance,
        "gradient_max_abs": float(np.max(np.abs(score))),
        "log_likelihood": loglik,
        "converged": converged,
        "iterations": iterations,
    }


def schoenfeld_time_test(event_times: np.ndarray, scaled_residuals: np.ndarray) -> dict:
    times = np.asarray(event_times, dtype=float)
    residuals = np.asarray(scaled_residuals, dtype=float)
    if residuals.ndim != 2 or len(times) != residuals.shape[0] or len(times) < 4:
        raise ValueError("invalid Schoenfeld residual data")
    g = rankdata(times, method="average") / (len(times) + 1.0)
    g = g - g.mean()
    gg = float(g @ g)
    slopes = (g[:, None] * residuals).sum(axis=0) / gg
    fitted = g[:, None] * slopes[None, :]
    errors = residuals - fitted
    sigma = np.cov(errors, rowvar=False, ddof=1)
    if residuals.shape[1] == 1:
        sigma = np.asarray([[float(sigma)]])
    slope_cov = sigma / gg
    se = np.sqrt(np.maximum(np.diag(slope_cov), 0.0))
    z = np.divide(slopes, se, out=np.zeros_like(slopes), where=se > 0)
    per_p = 2.0 * scipy.stats.norm.sf(np.abs(z))
    rank = int(np.linalg.matrix_rank(slope_cov))
    global_stat = float(slopes @ np.linalg.pinv(slope_cov) @ slopes)
    global_p = float(chi2.sf(global_stat, max(rank, 1)))
    return {
        "slopes": slopes,
        "standard_errors": se,
        "per_variable_p": per_p,
        "global_statistic": global_stat,
        "global_df": rank,
        "global_p": global_p,
    }


def standardized_design_diagnostics(matrix: pd.DataFrame) -> dict:
    x = matrix.to_numpy(float)
    rank = int(np.linalg.matrix_rank(x))
    rank_intercept = int(np.linalg.matrix_rank(np.column_stack([np.ones(len(x)), x])))
    mean = x.mean(axis=0)
    sd = x.std(axis=0, ddof=1)
    if np.any(sd <= 0):
        raise AssertionError("constant predictor in model matrix")
    z = (x - mean) / sd
    correlation = np.corrcoef(z, rowvar=False)
    condition = float(np.linalg.cond(z))
    vif = np.diag(np.linalg.pinv(correlation))
    return {"rank": rank, "rank_with_intercept": rank_intercept, "condition_number": condition, "vif": vif}


def summarize_influence(
    influence: pd.DataFrame,
    threshold: float,
    pooled_se: np.ndarray,
    parameters: list[str],
) -> tuple[dict, pd.DataFrame]:
    columns = [f"dfbeta_{name}" for name in parameters]
    patient = influence.groupby("subject_id", as_index=False)[columns].mean()
    standardized = patient.loc[:, columns].to_numpy(float) / np.asarray(pooled_se, dtype=float)[None, :]
    standardized_columns = [f"pooled_standardized_dfbeta_{name}" for name in parameters]
    patient.loc[:, standardized_columns] = standardized
    patient["max_abs_pooled_dfbeta"] = np.max(np.abs(standardized), axis=1)
    patient_maximum = patient["max_abs_pooled_dfbeta"]
    return {
        "metric": "maximum absolute Rubin-pooled standardized approximate DFBETA after signed averaging across imputations",
        "screening_threshold": float(threshold),
        "patients_over_threshold_n": int((patient_maximum > threshold).sum()),
        "median": float(patient_maximum.median()),
        "p95": float(patient_maximum.quantile(0.95)),
        "maximum": float(patient_maximum.max()),
    }, patient


def pooled_table(cox_pool: dict, fg_pool: dict) -> pd.DataFrame:
    rows = []
    for j, column in enumerate(MODEL_COLUMNS):
        row = {"parameter": column, "variable": VARIABLE_LABELS[column][0], "contrast_or_unit": VARIABLE_LABELS[column][1]}
        for prefix, pool in [("cs", cox_pool), ("sd", fg_pool)]:
            estimate = float(pool["estimate"][j])
            se = math.sqrt(float(pool["total_covariance"][j, j]))
            df = float(pool["degrees_freedom"][j])
            critical = float(student_t.ppf(0.975, df)) if np.isfinite(df) else 1.959963984540054
            stat = estimate / se
            p = float(2 * student_t.sf(abs(stat), df)) if np.isfinite(df) else float(2 * scipy.stats.norm.sf(abs(stat)))
            row[f"{prefix}HR"] = math.exp(estimate)
            row[f"{prefix}HR_ci_low"] = math.exp(estimate - critical * se)
            row[f"{prefix}HR_ci_high"] = math.exp(estimate + critical * se)
            row[f"{prefix}_p_value"] = p
            row[f"{prefix}_fmi"] = float(pool["fraction_missing_information"][j])
        rows.append(row)
    return pd.DataFrame(rows)


def write_markdown_table(df: pd.DataFrame, path: Path, title: str, footnote: str) -> None:
    def cell(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            return f"{value:.4g}"
        return str(value).replace("|", "\\|").replace("\n", " ")
    header = "| " + " | ".join(map(str, df.columns)) + " |"
    divider = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False, name=None)]
    path.write_text(f"# {title}\n\n" + "\n".join([header, divider, *body]) + f"\n\n{footnote}\n", encoding="utf-8")


def plot_forest(table: pd.DataFrame, output_stem: Path) -> None:
    plot = table.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(plot))
    fig, ax = plt.subplots(figsize=(9.2, 7.5))
    hr = plot.csHR.to_numpy(); lo = plot.csHR_ci_low.to_numpy(); hi = plot.csHR_ci_high.to_numpy()
    ax.errorbar(hr, y, xerr=[hr - lo, hi - hr], fmt="o", color=BLUE, ecolor=BLUE, capsize=3, markersize=5)
    ax.axvline(1.0, color="#888888", lw=1, ls="--")
    labels = [f"{v}\n{u}" for v, u in zip(plot.variable, plot.contrast_or_unit)]
    ax.set_yticks(y, labels); ax.set_xscale("log"); ax.set_xlabel("Adjusted cause-specific hazard ratio (95% CI)")
    ax.grid(axis="x", color="#E0E4E8", lw=0.8); ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, labelsize=9)
    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".png"), dpi=400, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_ph(event_times: np.ndarray, mean_residuals: np.ndarray, output_stem: Path) -> None:
    fig, axes = plt.subplots(5, 3, figsize=(11, 15), sharex=True)
    axes = axes.ravel()
    x = np.log1p(event_times / 24.0)
    for j, column in enumerate(MODEL_COLUMNS):
        ax = axes[j]
        y = mean_residuals[:, j]
        ax.scatter(x, y, s=9, alpha=0.35, color=BLUE, edgecolor="none")
        smooth = lowess(y, x, frac=0.5, return_sorted=True)
        ax.plot(smooth[:, 0], smooth[:, 1], color=ORANGE, lw=1.6)
        ax.axhline(0, color="#888888", lw=0.8, ls="--")
        ax.set_title(VARIABLE_LABELS[column][0], fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[len(MODEL_COLUMNS):]:
        ax.axis("off")
    fig.supxlabel("log(1 + days since t_HMIS)")
    fig.supylabel("Mean scaled Schoenfeld residual across imputations")
    fig.tight_layout(rect=(0.03, 0.03, 1, 1))
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_time_varying_hr(table: pd.DataFrame, output_stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    ax.fill_between(table.day, table.hr_ci_low, table.hr_ci_high, color=BLUE, alpha=0.18, linewidth=0)
    ax.plot(table.day, table.hr, color=BLUE, lw=2.0)
    ax.axhline(1.0, color="#777777", lw=1.0, ls="--")
    ax.set(xlabel="Days since t_HMIS", ylabel="Cause-specific hazard ratio per 5 cmH2O PEEP")
    ax.set_xlim(float(table.day.min()), float(table.day.max()))
    ax.grid(axis="y", color="#E0E4E8", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".png"), dpi=400, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--gate-d-lock", required=True, type=Path)
    parser.add_argument("--gate-d-qc", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--imputations", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261012)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for subdir in ["aggregate", "restricted", "tables", "figures"]:
        (args.output_dir / subdir).mkdir(parents=True, exist_ok=True)
    lock = json.loads(args.gate_d_lock.read_text(encoding="utf-8-sig"))
    gate_d_qc = json.loads(args.gate_d_qc.read_text(encoding="utf-8-sig"))
    if lock["analysis_population_n"] != 176 or lock["primary_predictor_df_hard_cap"] != 13:
        raise AssertionError("Gate D model lock drift")
    if lock["multiple_imputation"]["datasets"] != args.imputations or lock["multiple_imputation"]["seed"] != args.seed:
        raise AssertionError("imputation count or seed differs from Gate D lock")
    if gate_d_qc.get("status") != "PASS" or not gate_d_qc.get("gate_e_model_lock_created"):
        raise AssertionError("Gate D is not passed and locked")

    source = pd.read_csv(args.input)
    if len(source) != 176 or source.subject_id.nunique() != 176:
        raise AssertionError("Gate E requires 176 unique patients")
    counts = source.outcome.value_counts().to_dict()
    if counts != {"SUSTAINED_LIBERATION": 131, "DEATH_BEFORE_LIBERATION": 40, "NO_LIBERATION_BY_28D": 5}:
        raise AssertionError(f"outcome count drift: {counts}")
    event_codes = source.outcome.map({"NO_LIBERATION_BY_28D": 0, "SUSTAINED_LIBERATION": 1, "DEATH_BEFORE_LIBERATION": 2}).to_numpy(int)
    durations_h = source.followup_h.to_numpy(float)
    if np.any((event_codes == 0) & (np.abs(durations_h - 672.0) > 1e-10)):
        raise AssertionError("ordinary censoring before day 28 detected")

    imputation = source.loc[:, MODEL_COLUMNS + ["auxiliary_fio2", "auxiliary_resp_rate", "auxiliary_rass", "auxiliary_creatinine", "auxiliary_wbc"]].astype(float).copy()
    imputation["event_liberation"] = (event_codes == 1).astype(float)
    imputation["event_death"] = (event_codes == 2).astype(float)
    imputation["log_followup_h"] = np.log(durations_h)
    imputation["na_liberation"] = nelson_aalen_at_observed_times(durations_h, (event_codes == 1).astype(int))
    imputation["na_death"] = nelson_aalen_at_observed_times(durations_h, (event_codes == 2).astype(int))
    observed_missing = {column: int(imputation[column].isna().sum()) for column in imputation.columns if imputation[column].isna().any()}
    if observed_missing != INCOMPLETE_EXPECTED:
        raise AssertionError(f"missingness drift: {observed_missing}")

    completed = mice_pmm_impute(imputation, args.imputations, args.iterations, args.seed, k_pmm=5)
    imputation_rows = []
    for column, missing_n in observed_missing.items():
        missing_mask = imputation[column].isna()
        means = np.array([frame.loc[missing_mask, column].mean() for frame in completed])
        pooled_imputed = pd.concat([frame.loc[missing_mask, column] for frame in completed], ignore_index=True)
        observed = imputation.loc[~missing_mask, column]
        imputation_rows.append({
            "variable": column,
            "observed_n": int(observed.notna().sum()),
            "missing_n": missing_n,
            "missing_pct": 100 * missing_n / len(imputation),
            "observed_mean": float(observed.mean()),
            "observed_sd": float(observed.std(ddof=1)),
            "imputed_mean_all": float(pooled_imputed.mean()),
            "imputed_sd_all": float(pooled_imputed.std(ddof=1)),
            "between_imputation_mean_min": float(means.min()),
            "between_imputation_mean_max": float(means.max()),
            "donor_range_preserved": bool(pooled_imputed.between(observed.min(), observed.max()).all()),
        })
    pd.DataFrame(imputation_rows).to_csv(args.output_dir / "aggregate" / "mi_diagnostics.csv", index=False, encoding="utf-8-sig")

    cox_estimates, cox_covariances = [], []
    fg_estimates, fg_covariances = [], []
    model_diagnostics = []
    scaled_schoenfeld = []
    ph_global = []
    influence_frames = []
    extended_estimates, extended_covariances = [], []
    extended_diagnostics = []
    mi_long = []
    fine_time, fine_status = fine_gray_pseudo_time(durations_h / 24.0, event_codes, 28.0)

    for imputation_id, frame in enumerate(completed, start=1):
        xdf = build_model_matrix(frame)
        diagnostics = standardized_design_diagnostics(xdf)
        cox = fit_phreg(durations_h, (event_codes == 1).astype(int), xdf.to_numpy(), robust=False)
        fg = fit_phreg(fine_time, fine_status, xdf.to_numpy(), robust=True)
        cox_estimates.append(cox["params"]); cox_covariances.append(cox["covariance"])
        fg_estimates.append(fg["params"]); fg_covariances.append(fg["covariance"])
        model_diagnostics.append({
            "imputation": imputation_id,
            "n": len(frame), "target_events": int((event_codes == 1).sum()), "competing_deaths": int((event_codes == 2).sum()),
            "rank": diagnostics["rank"], "rank_with_intercept": diagnostics["rank_with_intercept"],
            "condition_number": diagnostics["condition_number"], "max_vif": float(np.max(diagnostics["vif"])),
            "cox_gradient_max_abs": cox["gradient_max_abs"], "finegray_gradient_max_abs": fg["gradient_max_abs"],
            "cox_warning_n": len(cox["warnings"]), "finegray_warning_n": len(fg["warnings"]),
            "cox_log_likelihood": cox["log_likelihood"], "finegray_log_likelihood": fg["log_likelihood"],
        })
        residuals = np.asarray(cox["result"].schoenfeld_residuals)
        event_mask = event_codes == 1
        scaled = residuals[event_mask] @ cox["covariance"] * int(event_mask.sum())
        scaled_schoenfeld.append(scaled)
        ph = schoenfeld_time_test(durations_h[event_mask], scaled)
        ph_global.append(ph["global_p"])
        score = np.asarray(cox["result"].score_residuals)
        dfbeta = score @ cox["covariance"]
        influence_frame = pd.DataFrame({
            "subject_id": source.subject_id,
            "imputation": imputation_id,
        })
        for parameter_index, parameter in enumerate(MODEL_COLUMNS):
            influence_frame[f"dfbeta_{parameter}"] = dfbeta[:, parameter_index]
        influence_frames.append(influence_frame)
        peep_index = MODEL_COLUMNS.index("peep_per_5")
        extended = fit_extended_cox(
            durations_h,
            (event_codes == 1).astype(int),
            xdf.to_numpy(),
            time_varying_index=peep_index,
            initial=np.r_[cox["params"], 0.0],
        )
        extended_estimates.append(extended["params"])
        extended_covariances.append(extended["covariance"])
        extended_diagnostics.append({
            "imputation": imputation_id,
            "converged": extended["converged"],
            "gradient_max_abs": extended["gradient_max_abs"],
            "iterations": extended["iterations"],
            "log_likelihood": extended["log_likelihood"],
        })
        internal = frame.loc[:, MODEL_COLUMNS].copy()
        internal.insert(0, "subject_id", source.subject_id.to_numpy())
        internal.insert(0, "imputation", imputation_id)
        internal["followup_h"] = durations_h
        internal["event_code"] = event_codes
        mi_long.append(internal)

    cox_pool = rubin_pool(np.asarray(cox_estimates), np.asarray(cox_covariances))
    fg_pool = rubin_pool(np.asarray(fg_estimates), np.asarray(fg_covariances))
    extended_pool = rubin_pool(np.asarray(extended_estimates), np.asarray(extended_covariances))
    table3 = pooled_table(cox_pool, fg_pool)
    table3.to_csv(args.output_dir / "tables" / "Table_3_primary_models.csv", index=False, encoding="utf-8-sig")
    write_markdown_table(table3, args.output_dir / "tables" / "Table_3_primary_models.md", "Table 3. Prespecified cause-specific Cox and complementary Fine–Gray associations", "All 13 prespecified degrees of freedom are shown. Hazard ratios are conditional associations, not causal effects or individual predictions. Fine–Gray estimates address a different competing-risk estimand and are not a validation of the Cox model.")
    pd.concat(mi_long, ignore_index=True).to_csv(args.output_dir / "restricted" / "mi_long_internal.csv", index=False, encoding="utf-8-sig")

    diagnostics_df = pd.DataFrame(model_diagnostics)
    diagnostics_df.to_csv(args.output_dir / "aggregate" / "model_diagnostics_by_imputation.csv", index=False, encoding="utf-8-sig")
    extended_diagnostics_df = pd.DataFrame(extended_diagnostics)
    extended_diagnostics_df.to_csv(args.output_dir / "aggregate" / "peep_time_interaction_diagnostics_by_imputation.csv", index=False, encoding="utf-8-sig")
    influence = pd.concat(influence_frames, ignore_index=True)
    influence.to_csv(args.output_dir / "restricted" / "cox_influence_by_imputation_internal.csv", index=False, encoding="utf-8-sig")
    threshold = 2 / math.sqrt(len(source))
    influence_summary, influence_patient = summarize_influence(
        influence,
        threshold,
        np.sqrt(np.diag(cox_pool["total_covariance"])),
        MODEL_COLUMNS,
    )
    influence_patient.to_csv(args.output_dir / "restricted" / "cox_influence_pooled_internal.csv", index=False, encoding="utf-8-sig")
    influence_parameter_rows = []
    for parameter in MODEL_COLUMNS:
        values = influence_patient[f"pooled_standardized_dfbeta_{parameter}"].abs()
        influence_parameter_rows.append({
            "parameter": parameter,
            "screening_threshold": threshold,
            "patients_over_threshold_n": int((values > threshold).sum()),
            "maximum_absolute_pooled_standardized_dfbeta": float(values.max()),
        })
    pd.DataFrame(influence_parameter_rows).to_csv(
        args.output_dir / "aggregate" / "influence_by_parameter.csv", index=False, encoding="utf-8-sig"
    )
    (args.output_dir / "aggregate" / "influence_summary.json").write_text(json.dumps(influence_summary, indent=2), encoding="utf-8")

    mean_schoenfeld = np.mean(np.stack(scaled_schoenfeld), axis=0)
    pooled_ph = schoenfeld_time_test(durations_h[event_codes == 1], mean_schoenfeld)
    ph_rows = [{"parameter": name, "p_value": float(p)} for name, p in zip(MODEL_COLUMNS, pooled_ph["per_variable_p"])]
    pd.DataFrame(ph_rows).to_csv(args.output_dir / "aggregate" / "ph_variable_tests.csv", index=False, encoding="utf-8-sig")
    ph_summary = {
        "method": "rank-time test of mean scaled Schoenfeld residuals across 50 imputations",
        "pooled_residual_global_statistic": pooled_ph["global_statistic"],
        "pooled_residual_global_df": pooled_ph["global_df"],
        "pooled_residual_global_p": pooled_ph["global_p"],
        "per_imputation_global_p_min": float(np.min(ph_global)),
        "per_imputation_global_p_median": float(np.median(ph_global)),
        "per_imputation_global_p_max": float(np.max(ph_global)),
    }
    (args.output_dir / "aggregate" / "ph_global_summary.json").write_text(json.dumps(ph_summary, indent=2), encoding="utf-8")

    interaction_index = len(MODEL_COLUMNS)
    interaction_estimate = float(extended_pool["estimate"][interaction_index])
    interaction_se = math.sqrt(float(extended_pool["total_covariance"][interaction_index, interaction_index]))
    interaction_df = float(extended_pool["degrees_freedom"][interaction_index])
    interaction_critical = float(student_t.ppf(0.975, interaction_df)) if np.isfinite(interaction_df) else 1.959963984540054
    interaction_p = (
        float(2 * student_t.sf(abs(interaction_estimate / interaction_se), interaction_df))
        if np.isfinite(interaction_df)
        else float(2 * scipy.stats.norm.sf(abs(interaction_estimate / interaction_se)))
    )
    time_rows = []
    extended_estimates_array = np.asarray(extended_estimates)
    extended_covariances_array = np.asarray(extended_covariances)
    peep_index = MODEL_COLUMNS.index("peep_per_5")
    for day in [0.0, 1.0, 3.0, 7.0, 14.0, 28.0]:
        contrast = np.zeros(len(MODEL_COLUMNS) + 1)
        contrast[peep_index] = 1.0
        contrast[interaction_index] = math.log1p(day)
        q = (extended_estimates_array @ contrast)[:, None]
        u = linear_contrast_covariances(extended_covariances_array, contrast)
        pool = rubin_pool(q, u)
        estimate = float(pool["estimate"][0])
        se = math.sqrt(float(pool["total_covariance"][0, 0]))
        df = float(pool["degrees_freedom"][0])
        critical = float(student_t.ppf(0.975, df)) if np.isfinite(df) else 1.959963984540054
        p = float(2 * student_t.sf(abs(estimate / se), df)) if np.isfinite(df) else float(2 * scipy.stats.norm.sf(abs(estimate / se)))
        time_rows.append({
            "day": day,
            "n_at_risk": int(np.sum(durations_h >= day * 24.0)),
            "cumulative_target_events": int(np.sum((event_codes == 1) & (durations_h <= day * 24.0))),
            "hr": math.exp(estimate),
            "hr_ci_low": math.exp(estimate - critical * se),
            "hr_ci_high": math.exp(estimate + critical * se),
            "p_value": p,
        })
    peep_time_table = pd.DataFrame(time_rows)
    peep_time_table.to_csv(args.output_dir / "aggregate" / "peep_time_varying_support.csv", index=False, encoding="utf-8-sig")
    interaction_summary = {
        "purpose": "supportive response to the prespecified proportional-hazards diagnostic; the frozen 13-df average-effect Cox model remains primary",
        "interaction": "PEEP per 5 cmH2O multiplied by log(1 + days since t_HMIS)",
        "log_hazard_interaction_estimate": interaction_estimate,
        "standard_error": interaction_se,
        "degrees_freedom": interaction_df,
        "ci_low": interaction_estimate - interaction_critical * interaction_se,
        "ci_high": interaction_estimate + interaction_critical * interaction_se,
        "p_value": interaction_p,
        "all_50_imputations_converged": bool(extended_diagnostics_df.converged.all()),
        "maximum_gradient_abs": float(extended_diagnostics_df.gradient_max_abs.max()),
    }
    (args.output_dir / "aggregate" / "peep_time_interaction_summary.json").write_text(json.dumps(interaction_summary, indent=2), encoding="utf-8")

    nonlinear_rows = []
    for variable in ["age_10", "log2_1_current_invasive_h", "pf_per_50", "peep_per_5"]:
        observed = imputation[variable].dropna().to_numpy(float)
        knots = np.quantile(observed, [0.10, 0.50, 0.90])
        if not np.all(np.diff(knots) > 0):
            nonlinear_rows.append({"variable": variable, "status": "NOT_ESTIMABLE_DUPLICATE_10_50_90_KNOTS", "knot_1": knots[0], "knot_2": knots[1], "knot_3": knots[2], "nonlinear_p": np.nan})
            continue
        q, u = [], []
        for frame in completed:
            xdf = build_model_matrix(frame)
            nonlinear = three_knot_rcs_nonlinear(xdf[variable].to_numpy(), knots)
            augmented = np.column_stack([xdf.to_numpy(), nonlinear])
            fit = fit_phreg(durations_h, (event_codes == 1).astype(int), augmented, robust=False)
            q.append(fit["params"]); u.append(fit["covariance"])
        pool = rubin_pool(np.asarray(q), np.asarray(u))
        j = len(MODEL_COLUMNS)
        estimate = pool["estimate"][j]; se = math.sqrt(pool["total_covariance"][j, j]); df = pool["degrees_freedom"][j]
        p = float(2 * student_t.sf(abs(estimate / se), df)) if np.isfinite(df) else float(2 * scipy.stats.norm.sf(abs(estimate / se)))
        nonlinear_rows.append({"variable": variable, "status": "ESTIMATED_DIAGNOSTIC_ONLY", "knot_1": knots[0], "knot_2": knots[1], "knot_3": knots[2], "nonlinear_p": p})
    pd.DataFrame(nonlinear_rows).to_csv(args.output_dir / "aggregate" / "nonlinearity_diagnostics.csv", index=False, encoding="utf-8-sig")

    plot_forest(table3, args.output_dir / "figures" / "Figure_4_cause_specific_cox_forest")
    plot_ph(durations_h[event_codes == 1], mean_schoenfeld, args.output_dir / "figures" / "Supplementary_PH_diagnostics")
    plot_time_varying_hr(peep_time_table, args.output_dir / "figures" / "Supplementary_PEEP_time_varying_HR")

    convergence_ok = bool(
        (diagnostics_df[["rank", "rank_with_intercept"]] == [13, 14]).all().all()
        and diagnostics_df.cox_gradient_max_abs.max() < 1e-4
        and diagnostics_df.finegray_gradient_max_abs.max() < 1e-4
        and diagnostics_df.cox_warning_n.sum() == 0
        and diagnostics_df.finegray_warning_n.sum() == 0
        and extended_diagnostics_df.converged.all()
        and extended_diagnostics_df.gradient_max_abs.max() < 1e-4
    )
    qc = {
        "gate": "E",
        "status": "PASS" if convergence_ok else "STOPPED_DIAGNOSTIC_FAILURE",
        "n": len(source), "target_events": int((event_codes == 1).sum()), "competing_deaths": int((event_codes == 2).sum()), "administratively_censored": int((event_codes == 0).sum()),
        "imputations": args.imputations, "iterations_per_imputation": args.iterations, "seed": args.seed, "pmm_donors": 5,
        "model_df": len(MODEL_COLUMNS), "all_13_parameters_reported": len(table3) == 13,
        "cox_ties": "Efron", "fine_gray_implementation": "subdistribution risk-set equivalence under day-28-only ordinary censoring; robust sandwich covariance",
        "all_design_matrices_full_rank": bool((diagnostics_df["rank"] == 13).all() and (diagnostics_df["rank_with_intercept"] == 14).all()),
        "max_condition_number": float(diagnostics_df.condition_number.max()), "max_vif": float(diagnostics_df.max_vif.max()),
        "max_cox_gradient_abs": float(diagnostics_df.cox_gradient_max_abs.max()), "max_finegray_gradient_abs": float(diagnostics_df.finegray_gradient_max_abs.max()),
        "cox_warning_total": int(diagnostics_df.cox_warning_n.sum()), "finegray_warning_total": int(diagnostics_df.finegray_warning_n.sum()),
        "ph_global_p": pooled_ph["global_p"], "influence_screen_positive_n": influence_summary["patients_over_threshold_n"],
        "ph_diagnostic_interpretation": "global PH violation; PEEP was the only variable assigned one supportive time interaction",
        "peep_time_interaction_support_converged": bool(extended_diagnostics_df.converged.all()),
        "peep_time_interaction_p": interaction_p,
        "max_extended_cox_gradient_abs": float(extended_diagnostics_df.gradient_max_abs.max()),
        "no_variable_selection": True, "gate_f_not_run": True,
    }
    (args.output_dir / "aggregate" / "gate_e_qc.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    receipt = {
        "gate": "E", "status": qc["status"], "seed": args.seed,
        "inputs": [{"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size} for path in [args.input, args.gate_d_lock, args.gate_d_qc]],
        "code": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "software": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "statsmodels": statsmodels.__version__, "matplotlib": matplotlib.__version__},
    }
    (args.output_dir / "aggregate" / "run_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    if not convergence_ok:
        raise RuntimeError("Gate E stopped: rank, convergence, or warning diagnostic failed")
    print(f"Gate E PASS: n={len(source)}, events=131, deaths=40, imputations={args.imputations}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
