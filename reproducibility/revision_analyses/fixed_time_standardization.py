from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats
import statsmodels.api as sm


MODEL_COLUMNS = [
    "age_10", "recorded_sex_male", "phenotype_sah", "phenotype_ich",
    "phenotype_sdh_edh", "log2_1_current_invasive_h", "landmark_gcsm_6",
    "pupil_bilateral_reactive", "major_sedative_active",
    "any_vasopressor_exposed_6h", "pf_per_50", "peep_per_5",
    "anchor_year_group_ordinal",
]
EXPOSURE = "any_vasopressor_exposed_6h"


def rubin_scalar(estimates: list[float], within_variances: list[float]) -> tuple[float, float, float, float]:
    q = np.asarray(estimates, float)
    u = np.asarray(within_variances, float)
    m = len(q)
    qbar = float(q.mean())
    ubar = float(u.mean())
    between = float(q.var(ddof=1)) if m > 1 else 0.0
    total = ubar + (1 + 1 / m) * between
    if between <= 0:
        df = math.inf
    else:
        df = (m - 1) * (1 + ubar / ((1 + 1 / m) * between)) ** 2
    critical = scipy.stats.t.ppf(.975, df) if np.isfinite(df) else 1.959963984540054
    se = math.sqrt(max(total, 0))
    return qbar, qbar - critical * se, qbar + critical * se, df


def standardized_risk(beta: np.ndarray, covariance: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    linear = x @ beta
    p = scipy.special.expit(linear)
    gradient = (p * (1 - p))[:, None] * x
    gradient = gradient.mean(axis=0)
    return float(p.mean()), float(gradient @ covariance @ gradient)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mi-long", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = pd.read_csv(args.mi_long)
    if data.imputation.nunique() != 50 or len(data) != 176 * 50:
        raise AssertionError("expected the frozen 50 by 176 completed datasets")
    rows = []
    for day in [7, 14, 28]:
        estimates = {"risk_unexposed": [], "risk_exposed": [], "risk_difference": []}
        variances = {"risk_unexposed": [], "risk_exposed": [], "risk_difference": []}
        for _, g in data.groupby("imputation", sort=True):
            y = ((g.event_code.eq(1)) & (g.followup_h.le(day * 24))).astype(int).to_numpy()
            base = sm.add_constant(g[MODEL_COLUMNS].to_numpy(float), has_constant="add")
            fit = sm.GLM(y, base, family=sm.families.Binomial()).fit()
            beta = np.asarray(fit.params)
            cov = np.asarray(fit.cov_params())
            x0 = base.copy(); x1 = base.copy()
            exposure_col = 1 + MODEL_COLUMNS.index(EXPOSURE)
            x0[:, exposure_col] = 0
            x1[:, exposure_col] = 1
            r0, v0 = standardized_risk(beta, cov, x0)
            r1, v1 = standardized_risk(beta, cov, x1)
            p0 = scipy.special.expit(x0 @ beta); p1 = scipy.special.expit(x1 @ beta)
            grad0 = (((p0 * (1 - p0))[:, None] * x0).mean(axis=0))
            grad1 = (((p1 * (1 - p1))[:, None] * x1).mean(axis=0))
            grad_d = grad1 - grad0
            vd = float(grad_d @ cov @ grad_d)
            estimates["risk_unexposed"].append(r0); variances["risk_unexposed"].append(v0)
            estimates["risk_exposed"].append(r1); variances["risk_exposed"].append(v1)
            estimates["risk_difference"].append(r1 - r0); variances["risk_difference"].append(vd)
        pooled = {key: rubin_scalar(estimates[key], variances[key]) for key in estimates}
        rows.append({
            "day": day,
            "adjusted_cumulative_incidence_unexposed": pooled["risk_unexposed"][0],
            "unexposed_ci_low": max(0.0, pooled["risk_unexposed"][1]),
            "unexposed_ci_high": min(1.0, pooled["risk_unexposed"][2]),
            "adjusted_cumulative_incidence_exposed": pooled["risk_exposed"][0],
            "exposed_ci_low": max(0.0, pooled["risk_exposed"][1]),
            "exposed_ci_high": min(1.0, pooled["risk_exposed"][2]),
            "adjusted_risk_difference_exposed_minus_unexposed": pooled["risk_difference"][0],
            "difference_ci_low": pooled["risk_difference"][1],
            "difference_ci_high": pooled["risk_difference"][2],
            "method": "MI-pooled logistic standardization for the binary cumulative-incidence status at the fixed day; competing deaths remain non-events",
        })
    output = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
