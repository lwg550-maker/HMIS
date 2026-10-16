from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FROZEN = {
    "parent_n": 474,
    "landmark_candidate_n": 221,
    "risk_set_n": 176,
    "not_invasive_at_landmark_n": 45,
    "sustained_7d_n": 131,
    "death_before_success_n": 40,
    "no_success_n": 5,
    "full_168h_confirmation_n": 101,
    "alive_discharge_surrogate_n": 30,
    "late_start_after_day21_n": 6,
    "sustained_48h_n": 145,
    "reinvasion_within_48h_n": 44,
    "reinvasion_within_7d_n": 61,
}

OUTCOME_CODE = {
    "SUSTAINED_LIBERATION": 1,
    "DEATH_BEFORE_LIBERATION": 2,
    "NO_LIBERATION_BY_28D": 0,
}

BLUE = "#2F6B9A"
ORANGE = "#D9782D"
NAVY = "#183B56"
LIGHT_BLUE = "#DCEAF3"
LIGHT_ORANGE = "#F7E4D5"
LIGHT_GREY = "#EEF1F4"
DARK_GREY = "#404B55"


@dataclass(frozen=True)
class AJResult:
    times: np.ndarray
    survival: np.ndarray
    cif_liberation: np.ndarray
    cif_death: np.ndarray


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate_accepted_frozen_counts(accepted: dict) -> None:
    excluded = {"parent_n", "landmark_candidate_n", "not_invasive_at_landmark_n"}
    observed = accepted.get("frozen_counts", {})
    for key, expected in FROZEN.items():
        if key in excluded:
            continue
        if observed.get(key) != expected:
            raise AssertionError(f"author-acceptance count drift: {key} observed={observed.get(key)} expected={expected}")


def qstats(values: Iterable[float]) -> tuple[float, float, float, int]:
    x = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if x.empty:
        return np.nan, np.nan, np.nan, 0
    q = x.quantile([0.25, 0.5, 0.75])
    return float(q.loc[0.5]), float(q.loc[0.25]), float(q.loc[0.75]), int(x.size)


def fmt_median_iqr(values: Iterable[float], digits: int = 1) -> str:
    med, q1, q3, n = qstats(values)
    if n == 0:
        return "NA"
    return f"{med:.{digits}f} ({q1:.{digits}f}–{q3:.{digits}f})"


def fmt_n_pct(n: int, denominator: int) -> str:
    return f"{int(n)} ({100.0 * n / denominator:.1f}%)"


def aalen_johansen(durations_days: np.ndarray, event_codes: np.ndarray) -> AJResult:
    """Aalen–Johansen CIFs for two competing events (1 and 2).

    Events and censoring at the same time share the same risk set. Events are
    applied before censoring, as required for right-continuous counting-process
    estimates.
    """
    durations = np.asarray(durations_days, dtype=float)
    events = np.asarray(event_codes, dtype=int)
    if durations.ndim != 1 or events.ndim != 1 or len(durations) != len(events):
        raise ValueError("durations and event_codes must be equal-length vectors")
    if len(durations) == 0 or np.any(~np.isfinite(durations)) or np.any(durations < 0):
        raise ValueError("durations must be finite, nonnegative, and nonempty")
    if not set(np.unique(events)).issubset({0, 1, 2}):
        raise ValueError("event_codes must be 0, 1, or 2")

    unique_times = np.sort(np.unique(durations))
    s, f1, f2 = 1.0, 0.0, 0.0
    out_t, out_s, out_f1, out_f2 = [0.0], [s], [f1], [f2]
    for t in unique_times:
        y = int(np.sum(durations >= t))
        d1 = int(np.sum((durations == t) & (events == 1)))
        d2 = int(np.sum((durations == t) & (events == 2)))
        if y <= 0 or d1 + d2 > y:
            raise AssertionError("invalid risk-set arithmetic")
        f1 += s * d1 / y
        f2 += s * d2 / y
        s *= 1.0 - (d1 + d2) / y
        out_t.append(float(t))
        out_s.append(float(s))
        out_f1.append(float(f1))
        out_f2.append(float(f2))
    result = AJResult(
        np.asarray(out_t), np.asarray(out_s), np.asarray(out_f1), np.asarray(out_f2)
    )
    if np.any(result.survival < -1e-12) or np.any(result.cif_liberation < -1e-12):
        raise AssertionError("negative Aalen–Johansen probability")
    if np.max(np.abs(result.survival + result.cif_liberation + result.cif_death - 1.0)) > 1e-10:
        raise AssertionError("Aalen–Johansen probabilities do not sum to one")
    return result


def step_at(result: AJResult, evaluation_times: np.ndarray, cause: int) -> np.ndarray:
    values = result.cif_liberation if cause == 1 else result.cif_death
    idx = np.searchsorted(result.times, np.asarray(evaluation_times), side="right") - 1
    idx = np.clip(idx, 0, len(values) - 1)
    return values[idx]


def bootstrap_aj(
    durations_days: np.ndarray,
    event_codes: np.ndarray,
    evaluation_times: np.ndarray,
    replicates: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if replicates < 2:
        raise ValueError("replicates must be at least 2")
    rng = np.random.default_rng(seed)
    n = len(durations_days)
    cause1 = np.empty((replicates, len(evaluation_times)), dtype=float)
    cause2 = np.empty_like(cause1)
    for b in range(replicates):
        idx = rng.integers(0, n, size=n)
        aj = aalen_johansen(durations_days[idx], event_codes[idx])
        cause1[b] = step_at(aj, evaluation_times, 1)
        cause2[b] = step_at(aj, evaluation_times, 2)
    return (
        np.quantile(cause1, 0.025, axis=0),
        np.quantile(cause1, 0.975, axis=0),
        np.quantile(cause2, 0.025, axis=0),
        np.quantile(cause2, 0.975, axis=0),
    )


def add_cat(rows: list[dict], section: str, label: str, series: pd.Series, levels: list[tuple[str, object]], n: int) -> None:
    nonmissing = int(series.notna().sum())
    missing = n - nonmissing
    for display, value in levels:
        count = int((series == value).sum())
        rows.append({
            "section": section,
            "characteristic": label,
            "level": display,
            "overall_n_pct_or_median_iqr": fmt_n_pct(count, n),
            "nonmissing_n": nonmissing,
            "missing_n": missing,
            "missing_pct": round(100 * missing / n, 1),
            "measurement_staleness_h_median_iqr": "NA",
        })


def add_cont(rows: list[dict], section: str, label: str, series: pd.Series, n: int, digits: int = 1, staleness: pd.Series | None = None) -> None:
    nonmissing = int(series.notna().sum())
    missing = n - nonmissing
    rows.append({
        "section": section,
        "characteristic": label,
        "level": "",
        "overall_n_pct_or_median_iqr": fmt_median_iqr(series, digits),
        "nonmissing_n": nonmissing,
        "missing_n": missing,
        "missing_pct": round(100 * missing / n, 1),
        "measurement_staleness_h_median_iqr": fmt_median_iqr(staleness, 1) if staleness is not None else "NA",
    })


def sedative_overlap_flag(active_drug_count: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(active_drug_count, errors="coerce")
    result = (numeric >= 2).astype(float)
    result[numeric.isna()] = np.nan
    return result


def make_table1(data: pd.DataFrame) -> pd.DataFrame:
    n = len(data)
    rows: list[dict] = []
    add_cont(rows, "Demographics", "Age, years", data.age, n, 1)
    add_cat(rows, "Demographics", "Recorded sex", data.recorded_sex, [("Female", "F"), ("Male", "M")], n)
    add_cat(rows, "Case composition", "Mutually exclusive diagnostic phenotype", data.diagnosis_phenotype,
            [("SAH", "SAH"), ("ICH", "ICH"), ("SDH/EDH", "SDH_EDH"), ("TBI", "TBI")], n)
    for display, column in [("SAH", "dx_sah"), ("ICH", "dx_ich"), ("SDH/EDH", "dx_sdh_edh"), ("TBI", "dx_tbi")]:
        add_cat(rows, "Case composition", "Raw diagnosis flags (overlap allowed)", data[column], [(display, 1)], n)
    year_order = ["2008 - 2010", "2011 - 2013", "2014 - 2016", "2017 - 2019", "2020 - 2022"]
    add_cat(rows, "Landmark", "Anchor year group", data.anchor_year_group, [(x, x) for x in year_order], n)
    add_cat(rows, "Landmark", "Target checkpoint", data.target_window, [(x, x) for x in ["W1", "W2", "W3", "W4", "W5"]], n)
    add_cat(rows, "Landmark", "Landmark GCS motor score", data.landmark_gcsm, [("5", 5), ("6", 6)], n)
    add_cont(rows, "Landmark", "Current invasive ventilation segment before t_HMIS, h", data.current_invasive_segment_h, n, 1)
    pupil = data.pupil_bilateral_reactive.map({1.0: "Bilateral reactive", 0.0: "Any abnormal"})
    add_cat(rows, "Neurologic and treatment state", "Pupil response within 4 h", pupil,
            [("Bilateral reactive", "Bilateral reactive"), ("Any abnormal", "Any abnormal")], n)
    # Attach pupil staleness to a separate audit row to avoid repeating it across categories.
    add_cont(rows, "Neurologic and treatment state", "Pupil assessment staleness, h", data.pupil_staleness_h, n, 1)
    add_cat(rows, "Neurologic and treatment state", "Major continuous sedative active at t_HMIS", data.major_sedative_active, [("Yes", 1)], n)
    for display, column in [
        ("Propofol", "propofol_active"), ("Dexmedetomidine", "dexmedetomidine_active"),
        ("Midazolam", "midazolam_active"), ("Lorazepam", "lorazepam_active"), ("Ketamine", "ketamine_active")
    ]:
        add_cat(rows, "Neurologic and treatment state", "Specific active sedative (overlap allowed)", data[column], [(display, 1)], n)
    add_cat(rows, "Neurologic and treatment state", "Two or more major continuous sedatives active", sedative_overlap_flag(data.active_sedative_drug_n), [("Yes", 1.0)], n)
    add_cat(rows, "Systemic and respiratory state", "Vasopressor exposure in prior 6 h", data.any_vasopressor_exposed_6h, [("Yes", 1)], n)
    add_cont(rows, "Systemic and respiratory state", "P/F ratio", data.pf_ratio, n, 1, data.pf_pair_staleness_h)
    add_cont(rows, "Systemic and respiratory state", "PEEP, cmH2O", data.peep, n, 1, data.peep_staleness_h)
    out = pd.DataFrame(rows)
    if any("p" == c.lower() or "p_value" in c.lower() for c in out.columns):
        raise AssertionError("Table 1 must not contain P values")
    return out


def make_table2(outcome: pd.DataFrame, accepted: dict) -> pd.DataFrame:
    n = len(outcome)
    sustained = outcome.outcome.eq("SUSTAINED_LIBERATION")
    death = outcome.outcome.eq("DEATH_BEFORE_LIBERATION")
    rows = []
    def row(section: str, label: str, value: str, note: str = "") -> None:
        rows.append({"section": section, "characteristic": label, "value": value, "note": note})
    row("Primary outcome state", "7-day-confirmed sustained liberation", fmt_n_pct(int(sustained.sum()), n), "Event time is the qualifying candidate stop time")
    row("Primary outcome state", "Death before sustained liberation", fmt_n_pct(int(death.sum()), n), "Competing event")
    row("Primary outcome state", "No sustained liberation by day 28", fmt_n_pct(int(outcome.outcome.eq("NO_LIBERATION_BY_28D").sum()), n), "Administrative censoring at day 28")
    row("Confirmation evidence", "Full 168-hour observation", fmt_n_pct(int((outcome.confirmation_source == "FULL_168H_OBSERVATION").sum()), n))
    row("Confirmation evidence", "Alive-discharge surrogate confirmation", fmt_n_pct(int((outcome.confirmation_source == "ALIVE_DISCHARGE_SURROGATE").sum()), n), "Evidence-boundary sensitivity analysis is prespecified")
    row("Candidate process", "Candidate stops per patient", fmt_median_iqr(outcome.candidate_n, 0), "Median (IQR)")
    row("Candidate process", "More than one candidate stop", fmt_n_pct(int((outcome.candidate_n > 1).sum()), n))
    row("Candidate process", "At least one failed candidate", fmt_n_pct(int((outcome.failed_candidate_n > 0).sum()), n))
    row("Candidate process", "First candidate followed by reinvasion within 48 h", fmt_n_pct(int(outcome.reinvasion_within_48h_of_first_stop.sum()), n), "Path event; may overlap final sustained liberation")
    row("Candidate process", "First candidate followed by reinvasion within 7 d", fmt_n_pct(int(outcome.reinvasion_within_7d_of_first_stop.sum()), n), "Path event; may overlap final sustained liberation")
    row("Candidate process", "Qualifying event started after day 21", fmt_n_pct(int(outcome.late_start_after_day21.sum()), n))
    row("Airway path", "Tracheostomy by day 28", fmt_n_pct(int(outcome.tracheostomy_by_28d.sum()), n), "Airway path; not an outcome state")
    row("Supporting outcome", "48-hour-confirmed sustained liberation", fmt_n_pct(int(accepted["frozen_counts"]["sustained_48h_n"]), n), "Prespecified sensitivity outcome")
    row("Event timing", "t_HMIS to sustained liberation, days", fmt_median_iqr(outcome.loc[sustained, "followup_h"] / 24, 1), "Median (IQR)")
    row("Event timing", "t_HMIS to death before liberation, days", fmt_median_iqr(outcome.loc[death, "followup_h"] / 24, 1), "Median (IQR)")
    return pd.DataFrame(rows)


def make_selection_table(gate_a: pd.DataFrame, parent: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    merge_cols = ["subject_id", "hadm_id", "stay_id", "age", "sex", "phenotype", "anchor_year_group", "t0"]
    merged = gate_a.merge(parent[merge_cols], on=["subject_id", "hadm_id", "stay_id"], how="left", validate="one_to_one")
    if merged.age.isna().any():
        raise AssertionError("Gate A candidates did not merge completely to the 474-person cohort")
    merged["t0"] = pd.to_datetime(merged["t0"], format="%d/%m/%Y %H:%M:%S")
    for c in ["t_HMIS", "low_gcsm_charttime"]:
        merged[c] = pd.to_datetime(merged[c], format="%Y-%m-%d %H:%M:%S")
    merged["hours_t0_to_landmark"] = (merged.t_HMIS - merged.t0).dt.total_seconds() / 3600
    merged["hours_low_to_high"] = (merged.t_HMIS - merged.low_gcsm_charttime).dt.total_seconds() / 3600
    groups = [(True, "Entered risk set (n=176)"), (False, "Not invasively ventilated at t_HMIS (n=45)")]
    characteristics: list[tuple[str, str, object]] = []
    characteristics.append(("Age, years", "continuous", "age"))
    characteristics.append(("Male recorded sex", "binary", ("sex", "M")))
    for lev in ["SAH", "ICH", "SDH_EDH", "TBI"]:
        characteristics.append((f"Diagnostic phenotype: {lev.replace('_', '/')}", "binary", ("phenotype", lev)))
    for lev in ["W1", "W2", "W3", "W4", "W5"]:
        characteristics.append((f"Target checkpoint: {lev}", "binary", ("target_window", lev)))
    characteristics += [
        ("Landmark GCS-M=6", "binary", ("landmark_gcsm", 6)),
        ("Pre-landmark low GCS-M", "continuous", "low_gcsm"),
        ("t0 to t_HMIS, h", "continuous", "hours_t0_to_landmark"),
        ("Low-to-high record interval, h", "continuous", "hours_low_to_high"),
    ]
    rows = []
    for label, kind, spec in characteristics:
        row = {"characteristic": label}
        for flag, column in groups:
            g = merged.loc[merged.risk_set == flag]
            if kind == "continuous":
                row[column] = fmt_median_iqr(g[str(spec)], 1)
            else:
                col, value = spec
                row[column] = fmt_n_pct(int((g[col] == value).sum()), len(g))
        rows.append(row)
    return pd.DataFrame(rows), merged


def write_markdown_table(df: pd.DataFrame, path: Path, title: str, footnote: str) -> None:
    def cell(value: object) -> str:
        if pd.isna(value):
            return ""
        return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")

    header = "| " + " | ".join(cell(c) for c in df.columns) + " |"
    divider = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False, name=None)]
    text = f"# {title}\n\n" + "\n".join([header, divider, *body]) + f"\n\n{footnote}\n"
    path.write_text(text, encoding="utf-8")


def format_exclusion_reasons(reasons: object) -> str:
    mapping = {
        "BETWEEN_INVASIVE_INTERVALS": "Between invasive intervals",
        "AT_OR_AFTER_FINAL_INVASIVE_INTERVAL": "At/after final invasive interval",
        "AT_INVASIVE_INTERVAL_END_BOUNDARY": "At invasive-interval end boundary",
    }
    items = reasons.items() if hasattr(reasons, "items") else reasons
    ordered = sorted(items, key=lambda x: (-int(x[1]), str(x[0])))
    return "\n".join(f"{mapping.get(str(key), str(key).replace('_', ' ').title())}: {int(value)}" for key, value in ordered)


def plot_flow(flow: pd.DataFrame, gate_a: pd.DataFrame, output_stem: Path) -> None:
    reasons = gate_a.loc[~gate_a.risk_set, "risk_set_reason"].value_counts()
    fig, ax = plt.subplots(figsize=(9.0, 6.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    def box(x, y, w, h, text, face, edge=NAVY, fs=11):
        p = plt.Rectangle((x, y), w, h, facecolor=face, edgecolor=edge, linewidth=1.5)
        ax.add_patch(p); ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=fs, color=DARK_GREY, wrap=True)
    box(2.6, 8.2, 4.8, 1.0, "Acute brain injury enriched cohort\nN=474", LIGHT_GREY)
    box(2.6, 5.9, 4.8, 1.15, "First valid adjacent low-to-high GCS-M transition\nN=221", LIGHT_BLUE)
    box(0.25, 2.3, 4.25, 1.25, "Invasive ventilation active at t_HMIS\nFormal landmark risk set, N=176", "#DDEFE3", "#39734C")
    reason_text = format_exclusion_reasons(reasons)
    box(5.15, 1.9, 4.6, 2.05, f"Not invasively ventilated at t_HMIS\nN=45\n{reason_text}", LIGHT_ORANGE, ORANGE, 8.8)
    arrow = dict(arrowstyle="-|>", lw=1.5, color=DARK_GREY, mutation_scale=14)
    ax.annotate("", xy=(5, 7.05), xytext=(5, 8.2), arrowprops=arrow)
    ax.annotate("", xy=(2.38, 3.55), xytext=(4.45, 5.9), arrowprops=arrow)
    ax.annotate("", xy=(7.45, 3.95), xytext=(5.55, 5.9), arrowprops=arrow)
    ax.text(5, 0.7, "t_HMIS is the actual chart time of the target GCS-M 5–6 record.", ha="center", fontsize=9.5, color=DARK_GREY)
    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".png"), dpi=400, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_state_machine(outcome: pd.DataFrame, accepted: dict, output_stem: Path) -> None:
    n = len(outcome)
    fig, ax = plt.subplots(figsize=(12.0, 7.2))
    ax.set_xlim(0, 14); ax.set_ylim(0, 10); ax.axis("off")
    def rounded(x, y, w, h, text, face, edge, fs=10.5):
        p = matplotlib.patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.08", facecolor=face, edgecolor=edge, linewidth=1.5)
        ax.add_patch(p); ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=fs, color=DARK_GREY, wrap=True)
    arrow = dict(arrowstyle="-|>", lw=1.4, color=DARK_GREY, mutation_scale=13)
    rounded(0.4, 4.2, 2.5, 1.35, f"Landmark risk set\nN={n}\n(t_HMIS)", LIGHT_BLUE, BLUE)
    rounded(3.7, 4.2, 2.7, 1.35, "Candidate stop\nstrictly after t_HMIS", LIGHT_GREY, DARK_GREY)
    rounded(7.1, 7.4, 3.1, 1.35, "No reinvasion or death\nthrough 168 h", "#DDEFE3", "#39734C")
    rounded(10.7, 7.2, 2.7, 1.7, "7-day-confirmed\nsustained liberation\nN=131", "#CBE7D4", "#39734C")
    rounded(7.1, 4.0, 3.1, 1.7, "Reinvasion within 168 h\nCandidate fails; remains at risk", LIGHT_ORANGE, ORANGE)
    rounded(10.7, 3.9, 2.7, 1.8, "Later qualifying stop\ncan still become event\n44 within 48 h; 61 within 7 d*", "#FBEEDD", ORANGE, 9.3)
    rounded(7.1, 0.6, 3.1, 1.65, "Death before sustained liberation\nCompeting event, N=40", "#F2D9D7", "#A84D49")
    rounded(10.7, 0.6, 2.7, 1.65, "No sustained liberation\nby day 28, N=5", LIGHT_GREY, DARK_GREY)
    for start, end in [((2.9,4.88),(3.7,4.88)), ((6.4,5.05),(7.1,8.05)), ((10.2,8.05),(10.7,8.05)), ((6.4,4.65),(7.1,4.85)), ((10.2,4.85),(10.7,4.85)), ((6.0,4.2),(7.4,2.25)), ((6.4,4.35),(10.7,1.45))]:
        ax.annotate("", xy=end, xytext=start, arrowprops=arrow)
    ax.text(11.98, 6.75, "101 full 168-h observation\n30 alive-discharge surrogate", ha="center", va="top", fontsize=9.2, color="#39734C")
    ax.text(7.0, 9.55, "Event time is backdated to the qualifying candidate stop.", ha="center", fontsize=10, color=DARK_GREY)
    ax.text(7.0, 0.05, "* Reinvasion counts are non-mutually-exclusive path events and may overlap the final 131 sustained-liberation events. Tracheostomy is not an outcome state.", ha="center", fontsize=8.8, color=DARK_GREY)
    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".png"), dpi=400, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_aj(durations: np.ndarray, events: np.ndarray, grid: np.ndarray, ci: tuple[np.ndarray, ...], output_stem: Path) -> None:
    aj = aalen_johansen(durations, events)
    point1 = step_at(aj, grid, 1); point2 = step_at(aj, grid, 2)
    lo1, hi1, lo2, hi2 = ci
    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    ax.fill_between(grid, lo1, hi1, step="post", color=BLUE, alpha=0.16, linewidth=0)
    ax.fill_between(grid, lo2, hi2, step="post", color=ORANGE, alpha=0.18, linewidth=0)
    ax.step(grid, point1, where="post", color=BLUE, lw=2.2, label="7-day-confirmed sustained liberation")
    ax.step(grid, point2, where="post", color=ORANGE, lw=2.2, label="Death before sustained liberation")
    ax.set(xlim=(0, 28), ylim=(0, 1), xlabel="Days since t_HMIS", ylabel="Cumulative incidence")
    ax.set_xticks([0, 3, 7, 14, 21, 28]); ax.set_yticks(np.linspace(0, 1, 6))
    ax.grid(axis="y", color="#D9DEE3", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="center right", frameon=False, fontsize=9.5)
    ax.text(0.0, -0.19, "Shading: 95% patient-level nonparametric bootstrap CI (2,000 replicates).\nLiberation event time is the qualifying candidate-stop time, after future 7-day confirmation.", transform=ax.transAxes, fontsize=8.7, color=DARK_GREY)
    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".png"), dpi=400, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--parent", required=True, type=Path)
    p.add_argument("--gate-a", required=True, type=Path)
    p.add_argument("--gate-a-flow", required=True, type=Path)
    p.add_argument("--gate-b-outcome", required=True, type=Path)
    p.add_argument("--gate-b-acceptance", required=True, type=Path)
    p.add_argument("--gate-c", required=True, type=Path)
    p.add_argument("--gate-c-report", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--bootstrap-replicates", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20261012)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    for d in ["aggregate", "restricted", "figures", "tables"]:
        (args.output_dir / d).mkdir(parents=True, exist_ok=True)
    inputs = [args.parent, args.gate_a, args.gate_a_flow, args.gate_b_outcome, args.gate_b_acceptance, args.gate_c, args.gate_c_report]
    for path in inputs:
        if not path.exists():
            raise FileNotFoundError(path)

    parent = pd.read_csv(args.parent)
    gate_a = pd.read_csv(args.gate_a)
    gate_a_flow = pd.read_csv(args.gate_a_flow)
    outcome = pd.read_csv(args.gate_b_outcome)
    gate_c = pd.read_csv(args.gate_c)
    accepted = json.loads(args.gate_b_acceptance.read_text(encoding="utf-8-sig"))

    # Hard gates: stop before producing inferential outputs if any frozen count drifts.
    assertions = {
        "parent_n": len(parent),
        "parent_unique_subjects": parent.subject_id.nunique(),
        "landmark_candidate_n": len(gate_a),
        "landmark_candidate_unique_subjects": gate_a.subject_id.nunique(),
        "risk_set_n": int(gate_a.risk_set.sum()),
        "not_invasive_at_landmark_n": int((~gate_a.risk_set).sum()),
        "gate_b_n": len(outcome),
        "gate_b_unique_subjects": outcome.subject_id.nunique(),
        "gate_c_n": len(gate_c),
        "gate_c_unique_subjects": gate_c.subject_id.nunique(),
        "sustained_7d_n": int(outcome.outcome.eq("SUSTAINED_LIBERATION").sum()),
        "death_before_success_n": int(outcome.outcome.eq("DEATH_BEFORE_LIBERATION").sum()),
        "no_success_n": int(outcome.outcome.eq("NO_LIBERATION_BY_28D").sum()),
        "full_168h_confirmation_n": int(outcome.confirmation_source.eq("FULL_168H_OBSERVATION").sum()),
        "alive_discharge_surrogate_n": int(outcome.confirmation_source.eq("ALIVE_DISCHARGE_SURROGATE").sum()),
        "late_start_after_day21_n": int(outcome.late_start_after_day21.sum()),
        "reinvasion_within_48h_n": int(outcome.reinvasion_within_48h_of_first_stop.sum()),
        "reinvasion_within_7d_n": int(outcome.reinvasion_within_7d_of_first_stop.sum()),
        "sustained_48h_n": int(accepted["frozen_counts"]["sustained_48h_n"]),
    }
    expected = {
        **FROZEN,
        "parent_unique_subjects": 474,
        "landmark_candidate_unique_subjects": 221,
        "gate_b_n": 176,
        "gate_b_unique_subjects": 176,
        "gate_c_n": 176,
        "gate_c_unique_subjects": 176,
    }
    for key, value in assertions.items():
        if key in expected and value != expected[key]:
            raise AssertionError(f"frozen count drift: {key} observed={value} expected={expected[key]}")
    if not accepted.get("gate_b_passed") or accepted.get("status") != "AUTHOR_ACCEPTED_GATE_B_PASS":
        raise AssertionError("author-accepted Gate B receipt is missing")
    validate_accepted_frozen_counts(accepted)
    if "Gate C通过" not in args.gate_c_report.read_text(encoding="utf-8-sig"):
        raise AssertionError("Gate C pass statement not found")
    if set(outcome.subject_id) != set(gate_c.subject_id) or set(outcome.subject_id) != set(gate_a.loc[gate_a.risk_set, "subject_id"]):
        raise AssertionError("Gate A, B, and C risk-set identities differ")
    if int(outcome.reinvasion_within_48h_of_first_stop.sum()) > int(outcome.reinvasion_within_7d_of_first_stop.sum()):
        raise AssertionError("48-hour reinvasion must be a subset of 7-day reinvasion")

    table1 = make_table1(gate_c)
    table2 = make_table2(outcome, accepted)
    selection, selection_internal = make_selection_table(gate_a, parent)
    table1.to_csv(args.output_dir / "tables" / "Table_1_landmark_characteristics.csv", index=False, encoding="utf-8-sig")
    table2.to_csv(args.output_dir / "tables" / "Table_2_outcomes_and_paths.csv", index=False, encoding="utf-8-sig")
    selection.to_csv(args.output_dir / "tables" / "Supplementary_selection_176_vs_45.csv", index=False, encoding="utf-8-sig")
    selection_internal.to_csv(args.output_dir / "restricted" / "selection_process_internal.csv", index=False, encoding="utf-8-sig")
    write_markdown_table(table1, args.output_dir / "tables" / "Table_1_landmark_characteristics.md", "Table 1. Characteristics of the 176-patient landmark risk set", "Values are n (%) or median (IQR). Missingness and measurement staleness are reported explicitly. No P values were calculated.")
    write_markdown_table(table2, args.output_dir / "tables" / "Table_2_outcomes_and_paths.md", "Table 2. Outcome and post-landmark ventilatory-path composition", "Reinvasion and tracheostomy rows describe non-mutually-exclusive paths and must not be added to the three primary outcome states.")
    write_markdown_table(selection, args.output_dir / "tables" / "Supplementary_selection_176_vs_45.md", "Selection-process description: 176 risk-set entrants and 45 non-entrants", "Descriptive statistics only; no P values or inverse-probability weighting. The table characterizes conditioning on invasive ventilation at t_HMIS and is not a mechanistic comparison.")

    durations = pd.to_numeric(outcome.followup_h, errors="raise").to_numpy(float) / 24.0
    events = outcome.outcome.map(OUTCOME_CODE)
    if events.isna().any() or np.any(durations > 28 + 1e-9):
        raise AssertionError("invalid outcome coding or follow-up beyond day 28")
    events_np = events.to_numpy(int)
    grid = np.linspace(0, 28, 281)
    ci = bootstrap_aj(durations, events_np, grid, args.bootstrap_replicates, args.seed)
    aj = aalen_johansen(durations, events_np)
    eval_days = np.asarray([3.0, 7.0, 14.0, 28.0])
    eval_ci = bootstrap_aj(durations, events_np, eval_days, args.bootstrap_replicates, args.seed)
    estimates = pd.DataFrame({
        "day": eval_days.astype(int),
        "liberation_cif": step_at(aj, eval_days, 1),
        "liberation_ci_low": eval_ci[0],
        "liberation_ci_high": eval_ci[1],
        "death_cif": step_at(aj, eval_days, 2),
        "death_ci_low": eval_ci[2],
        "death_ci_high": eval_ci[3],
        "at_risk_immediately_before": [int(np.sum(durations >= d)) for d in eval_days],
    })
    estimates.to_csv(args.output_dir / "aggregate" / "aalen_johansen_estimates.csv", index=False, encoding="utf-8-sig")
    curve = pd.DataFrame({
        "day": grid,
        "liberation_cif": step_at(aj, grid, 1), "liberation_ci_low": ci[0], "liberation_ci_high": ci[1],
        "death_cif": step_at(aj, grid, 2), "death_ci_low": ci[2], "death_ci_high": ci[3],
    })
    curve.to_csv(args.output_dir / "aggregate" / "aalen_johansen_curve.csv", index=False, encoding="utf-8-sig")

    plot_flow(gate_a_flow, gate_a, args.output_dir / "figures" / "Figure_1_cohort_flow")
    plot_state_machine(outcome, accepted, args.output_dir / "figures" / "Figure_2_outcome_state_machine")
    plot_aj(durations, events_np, grid, ci, args.output_dir / "figures" / "Figure_3_aalen_johansen")

    model_lock = {
        "lock_stage": "Gate D before any adjusted association result",
        "analysis_population_n": 176,
        "primary_estimand": "conditional cause-specific association with time to 7-day-confirmed sustained invasive-ventilation liberation",
        "primary_model": "cause-specific Cox proportional hazards; Efron ties",
        "competing_event": "death before sustained liberation",
        "complementary_model": "Fine-Gray subdistribution hazards; complementary competing-risk estimand",
        "primary_predictor_df_hard_cap": 13,
        "predictors": [
            "age per 10 years (1 df)", "recorded sex (1 df)",
            "mutually exclusive diagnostic phenotype SAH/ICH/SDH-EDH/TBI (3 df)",
            "log2(1 + current invasive ventilation segment hours) (1 df)",
            "landmark GCS-M 6 vs 5 (1 df)", "bilaterally reactive pupils vs any abnormal (1 df)",
            "major continuous sedative active at t_HMIS (1 df)",
            "vasopressor exposure during prior 6 h (1 df)", "P/F per 50 units (1 df)",
            "PEEP per 5 cmH2O (1 df)", "anchor_year_group ordinal trend 0-4 (1 df)"
        ],
        "multiple_imputation": {
            "datasets": 50, "seed": args.seed,
            "required": ["all primary predictors", "liberation indicator", "death indicator", "log follow-up time", "cause-specific Nelson-Aalen auxiliaries"],
            "prespecified_auxiliaries": ["FiO2", "respiratory rate", "RASS", "creatinine", "white blood cell count"]
        },
        "sensitivity_order": [
            "candidate stop must begin on or before day 21",
            "fixed 176-person risk set; merge only post-t_HMIS ventilation gaps <=6 h",
            "strict discharge/transfer evidence rule",
            "complete-case analysis",
            "replace P/F ratio with FiO2",
            "Fine-Gray complementary estimate",
            "48-hour-confirmed sustained liberation",
            "GCS-M 5 vs 6 and W1 vs W2-W5 descriptive comparisons only"
        ],
        "prohibited": ["stepwise regression", "univariable P-value screening", "best-subset selection", "machine learning", "AUC", "score development", "train-test split"],
    }
    (args.output_dir / "aggregate" / "gate_e_model_lock.json").write_text(json.dumps(model_lock, indent=2, ensure_ascii=False), encoding="utf-8")

    qc = {
        "gate": "D",
        "status": "PASS",
        "no_adjusted_model_run": True,
        "frozen_count_assertions": assertions,
        "frozen_identity_sets_match": True,
        "table_1_has_p_values": False,
        "selection_table_has_p_values": False,
        "aj_method": "Aalen-Johansen; patient-level nonparametric percentile bootstrap 95% CI",
        "bootstrap_replicates": args.bootstrap_replicates,
        "seed": args.seed,
        "patient_level_files_only_in_restricted": True,
        "gate_e_model_lock_created": True,
    }
    (args.output_dir / "aggregate" / "gate_d_qc.json").write_text(json.dumps(qc, indent=2, ensure_ascii=False), encoding="utf-8")

    receipt = {
        "gate": "D", "status": "PASS", "seed": args.seed,
        "bootstrap_replicates": args.bootstrap_replicates,
        "inputs": [{"path": str(p), "sha256": sha256(p), "bytes": p.stat().st_size} for p in inputs],
        "code": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "software": {"python": sys.version, "platform": platform.platform(), "pandas": pd.__version__, "numpy": np.__version__, "matplotlib": matplotlib.__version__},
    }
    (args.output_dir / "aggregate" / "run_receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Gate D PASS: n={len(outcome)}, liberation={int((events_np==1).sum())}, death={int((events_np==2).sum())}, censored={int((events_np==0).sum())}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
