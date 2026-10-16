from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle

from audit_panel_alignment import require_matplotlib_panel_alignment


MM_PER_INCH = 25.4
width_mm = 183.0

NAVY = "#173B5E"
GREEN = "#2F6B4F"
CORAL = "#C96552"
GREY = "#66717C"
MID_GREY = "#98A2AB"
LIGHT_GREY = "#D8DEE3"
VERY_LIGHT_GREY = "#F3F5F6"
BLACK = "#1B1F23"
WHITE = "#FFFFFF"

PARAMETER_ORDER = [
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

DISPLAY_LABELS = {
    "age_10": ("Age", "per 10 years"),
    "recorded_sex_male": ("Recorded sex", "male vs female"),
    "phenotype_sah": ("Diagnostic phenotype: SAH", "vs TBI"),
    "phenotype_ich": ("Diagnostic phenotype: ICH", "vs TBI"),
    "phenotype_sdh_edh": ("Diagnostic phenotype: SDH/EDH", "vs TBI"),
    "log2_1_current_invasive_h": ("Current IMV duration", "per doubling of 1 + hours"),
    "landmark_gcsm_6": ("Landmark GCS-M", "6 vs 5"),
    "pupil_bilateral_reactive": ("Pupil response", "bilaterally reactive vs any abnormal"),
    "major_sedative_active": ("Major continuous sedative active", "yes vs no"),
    "any_vasopressor_exposed_6h": ("Vasopressor exposure in prior 6 h", "yes vs no"),
    "pf_per_50": ("P/F ratio", "per 50 units"),
    "peep_per_5": ("PEEP †", "per 5 cmH2O"),
    "anchor_year_group_ordinal": ("Anchor year group", "per one-category increase"),
}

SENSITIVITY_ORDER = [
    "PRIMARY_7D_MI_REFERENCE",
    "CONFIRMATION_TIME",
    "PEEP_TIME_INTERACTION",
    "UNaugmented_TIMELINE",
    "ALL_TRACH_PPV_BOUNDARY",
    "DAY21_CANDIDATE_CUTOFF",
    "GAP_LE_6H",
    "STRICT_DISCHARGE",
    "NO_DISCHARGE_SURROGATE",
    "COMPLETE_CASE",
    "FIO2_REPLACES_PF",
    "SUSTAINED_48H",
]

SENSITIVITY_LABELS = {
    "PRIMARY_7D_MI_REFERENCE": "Primary rule",
    "CONFIRMATION_TIME": "Confirmation-time analysis",
    "PEEP_TIME_INTERACTION": "PEEP time-interaction model",
    "UNaugmented_TIMELINE": "Unaugmented IMV timeline",
    "ALL_TRACH_PPV_BOUNDARY": "All tracheostomy time as PPV",
    "DAY21_CANDIDATE_CUTOFF": "Candidates by day 21",
    "GAP_LE_6H": "Merge IMV gaps ≤6 h",
    "STRICT_DISCHARGE": "Stricter discharge evidence",
    "NO_DISCHARGE_SURROGATE": "No discharge surrogate",
    "COMPLETE_CASE": "Complete cases",
    "FIO2_REPLACES_PF": "FiO2 replaces P/F ratio",
    "SUSTAINED_48H": "48-h confirmation",
}


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7.0,
        "axes.titlesize": 8.5,
        "axes.labelsize": 7.0,
        "xtick.labelsize": 6.0,
        "ytick.labelsize": 6.0,
        "legend.fontsize": 6.0,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": WHITE,
        "savefig.facecolor": WHITE,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
    }
)


def mm_to_in(mm: float) -> float:
    return mm / MM_PER_INCH


def panel_label(ax, label: str, x: float = -0.045, y: float = 1.02) -> None:
    ax.text(x, y, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=9, fontweight="bold", color=BLACK)


def clean_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BLACK)
    ax.spines["bottom"].set_color(BLACK)
    ax.tick_params(colors=BLACK, width=0.8, length=3)


def save_figure(
    fig,
    axes: list,
    panel_ids: list[str],
    stem: Path,
    *,
    exemptions: list[dict] | None = None,
) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig,
        axes=axes,
        panel_ids=panel_ids,
        json_out=stem.with_suffix(".alignment.json"),
        overlay_svg=stem.with_suffix(".alignment.svg"),
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=False,
        strict=True,
        exemptions=exemptions or [],
    )
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    fig.savefig(stem.with_suffix(".tiff"), dpi=600)
    plt.close(fig)


def first_integer(value: str) -> int:
    match = re.search(r"\d+", str(value))
    if match is None:
        raise ValueError(f"No integer found in {value!r}")
    return int(match.group())


def require_positive(values, label: str) -> None:
    arr = np.asarray(values, dtype=float)
    if np.any(~np.isfinite(arr)) or np.any(arr <= 0):
        raise ValueError(f"{label} must contain finite positive values for logarithmic plotting")


def load_inputs(source_dir: Path) -> dict[str, pd.DataFrame]:
    names = {
        "curve": "Figure_3_curve.csv",
        "fixed": "Figure_3_fixed_time_estimates.csv",
        "models": "Figure_4_primary_models.csv",
        "ph": "Figure_S1_PH_variable_tests.csv",
        "peep": "Figure_S2_PEEP_time_varying.csv",
        "sensitivity": "Figure_S3_revision_sensitivity_models.csv",
        "stratified": "Figure_S4_descriptive_stratified_CIF.csv",
        "paths": "Table_2_outcomes_paths.csv",
        "diagnostics": "Figure_3_model_diagnostics.csv",
    }
    return {key: pd.read_csv(source_dir / filename) for key, filename in names.items()}


def assert_contract(data: dict[str, pd.DataFrame]) -> dict[str, int]:
    curve = data["curve"]
    fixed = data["fixed"]
    models = data["models"]
    paths = data["paths"]

    assert list(models["parameter"]) == PARAMETER_ORDER
    assert len(models) == 13
    assert set(fixed["day"].astype(int)) == {3, 7, 14, 28}
    assert fixed.set_index("day").loc[3, "at_risk_immediately_before"] == 113
    assert fixed.set_index("day").loc[7, "at_risk_immediately_before"] == 70
    assert fixed.set_index("day").loc[14, "at_risk_immediately_before"] == 30
    assert fixed.set_index("day").loc[28, "at_risk_immediately_before"] == 5

    day28 = curve.loc[np.isclose(curve["day"], 28.0)].iloc[-1]
    assert np.isclose(day28["liberation_cif"], 131 / 176)
    assert np.isclose(day28["death_cif"], 40 / 176)

    lookup = {row["characteristic"]: first_integer(row["value"]) for _, row in paths.iterrows()}
    counts = {
        "confirmed": lookup["7-day-confirmed sustained liberation"],
        "death": lookup["Death before sustained liberation"],
        "unconfirmed": lookup["No sustained liberation by day 28"],
        "full_168h": lookup["Full 168-hour observation"],
        "discharge": lookup["Alive-discharge surrogate confirmation"],
        "multiple": lookup["More than one candidate stop"],
        "failed": lookup["At least one failed candidate"],
        "reinit_48h": lookup.get("First candidate followed by IMV re-initiation within 48 h", lookup.get("First candidate followed by reinvasion within 48 h")),
        "reinit_7d": lookup.get("First candidate followed by IMV re-initiation within 7 d", lookup.get("First candidate followed by reinvasion within 7 d")),
        "late": lookup["Qualifying event started after day 21"],
        "trach": lookup["Tracheostomy by day 28"],
        "support_48h": lookup["48-hour-confirmed sustained liberation"],
    }
    assert counts["confirmed"] == 131
    assert counts["death"] == 40
    assert counts["unconfirmed"] == 5
    assert counts["full_168h"] == 101
    assert counts["discharge"] == 30
    assert counts["multiple"] == 59
    assert counts["failed"] == 86
    assert counts["reinit_48h"] == 44
    assert counts["reinit_7d"] == 61
    assert counts["late"] == 6
    assert counts["trach"] == 31
    assert counts["support_48h"] == 145
    assert 176 == counts["confirmed"] + counts["death"] + counts["unconfirmed"]
    assert counts["confirmed"] == counts["full_168h"] + counts["discharge"]
    return counts


def draw_cumulative_incidence(ax, curve: pd.DataFrame, fixed: pd.DataFrame) -> None:
    panel_label(ax, "A")
    ax.set_title("Competing-risk cumulative incidence", loc="left", fontweight="bold", pad=7)
    x = curve["day"].to_numpy(float)
    ax.fill_between(x, curve["liberation_ci_low"], curve["liberation_ci_high"], step="post", color=GREEN, alpha=0.14, linewidth=0)
    ax.fill_between(x, curve["death_ci_low"], curve["death_ci_high"], step="post", color=CORAL, alpha=0.14, linewidth=0)
    ax.step(x, curve["liberation_cif"], where="post", color=GREEN, lw=1.7, label="Primary-rule-confirmed cessation start")
    ax.step(x, curve["death_cif"], where="post", color=CORAL, lw=1.7, ls=(0, (4, 2)), label="Death before a confirmed event")
    ax.set_xlim(0, 28)
    ax.set_ylim(0, 0.86)
    ax.set_xticks([0, 3, 7, 14, 21, 28])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_xlabel("Days since t_HMIS")
    ax.set_ylabel("Cumulative incidence")
    ax.grid(axis="y", color=LIGHT_GREY, linewidth=0.6)
    clean_axis(ax)
    ax.legend(loc="lower right", frameon=False, handlelength=2.4)

    risk_days = [0, 3, 7, 14, 28]
    fixed_map = fixed.set_index(fixed["day"].astype(int))["at_risk_immediately_before"].to_dict()
    risk_n = [176] + [int(fixed_map[d]) for d in risk_days[1:]]
    ax.text(-0.06, -0.22, "At risk", transform=ax.transAxes, ha="right", va="center", fontsize=6.2, fontweight="bold", clip_on=False)
    for day, n in zip(risk_days, risk_n):
        ax.text(day / 28.0, -0.22, str(n), transform=ax.transAxes, ha="center", va="center", fontsize=6.2, clip_on=False)
    ax.text(
        0.0,
        -0.34,
        "95% CIs from 2,000 patient-level bootstrap samples; event time is backdated after future confirmation.",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=6.0,
        color=GREY,
        clip_on=False,
    )


def draw_day28_states(ax, counts: dict[str, int]) -> None:
    panel_label(ax, "B")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.0, 0.99, "Day-28 primary states", ha="left", va="top", fontsize=8.5, fontweight="bold", color=BLACK)
    ax.text(0.0, 0.93, "Mutually exclusive; N=176", ha="left", va="top", fontsize=6.0, color=GREY)

    total = 176
    segments = [
        ("confirmed", GREEN),
        ("death", CORAL),
        ("unconfirmed", GREY),
    ]
    left = 0.0
    y, h = 0.73, 0.13
    for key, color in segments:
        width = counts[key] / total
        ax.add_patch(Rectangle((left, y), width, h, fc=color, ec=WHITE, lw=0.8))
        left += width
    ax.add_patch(Rectangle((0, y), 1.0, h, fc="none", ec=LIGHT_GREY, lw=0.8))

    legend_rows = [
        (0.62, GREEN, f"Confirmed event  {counts['confirmed']}/176 (74.4%)"),
        (0.54, CORAL, f"Death before event  {counts['death']}/176 (22.7%)"),
        (0.46, GREY, f"No confirmed event or death  {counts['unconfirmed']}/176 (2.8%)"),
    ]
    for yy, color, text in legend_rows:
        ax.add_patch(Rectangle((0.0, yy - 0.012), 0.024, 0.024, fc=color, ec=color))
        ax.text(0.038, yy, text, ha="left", va="center", fontsize=6.1, color=BLACK)

    ax.add_line(Line2D([0.02, 0.02], [0.18, 0.34], color=GREEN, lw=1.0))
    ax.add_line(Line2D([0.02, 0.16], [0.34, 0.34], color=GREEN, lw=1.0))
    ax.add_line(Line2D([0.02, 0.16], [0.18, 0.18], color=GREEN, lw=1.0))
    ax.text(0.0, 0.38, "Evidence supporting 131 confirmed events", ha="left", va="bottom", fontsize=6.2, fontweight="bold", color=GREEN)
    ax.add_patch(Circle((0.18, 0.34), 0.012, fc=GREEN, ec=GREEN))
    ax.text(0.21, 0.34, f"Full 168-h observation  {counts['full_168h']}", ha="left", va="center", fontsize=6.0, color=GREEN)
    ax.add_patch(Circle((0.18, 0.18), 0.012, fc=WHITE, ec=GREEN, lw=1.2))
    ax.text(0.21, 0.18, f"Alive-discharge surrogate  {counts['discharge']}", ha="left", va="center", fontsize=6.0, color=GREEN)


def draw_pathway_features(ax, counts: dict[str, int]) -> None:
    panel_label(ax, "C", x=-0.024, y=1.09)
    labels = [
        "More than one\ncandidate cessation",
        "At least one\nfailed candidate",
        "IMV re-initiation ≤48 h\nafter first cessation",
        "IMV re-initiation ≤7 d\nafter first cessation",
        "Qualifying event started\nafter day 21",
        "Tracheostomy\nby day 28",
        "Supportive 48-h\nconfirmed event",
    ]
    values = [
        counts["multiple"],
        counts["failed"],
        counts["reinit_48h"],
        counts["reinit_7d"],
        counts["late"],
        counts["trach"],
        counts["support_48h"],
    ]
    colors = [CORAL, CORAL, CORAL, CORAL, GREY, GREY, GREEN]
    y = np.arange(len(labels))[::-1]
    ax.text(0.0, 1.09, "Non-exclusive pathway features", transform=ax.transAxes, ha="left", va="bottom", fontsize=8.5, fontweight="bold", color=BLACK)
    ax.text(0.0, 1.015, "These counts may overlap and must not be summed.", transform=ax.transAxes, ha="left", va="bottom", fontsize=6.0, color=GREY)
    for yy, value, color in zip(y, values, colors):
        ax.hlines(yy, 0, value, color=color, lw=1.5)
        ax.scatter(value, yy, s=28, facecolor=WHITE, edgecolor=color, linewidth=1.3, zorder=3)
        ax.text(value + 4, yy, f"{value}/176 ({100 * value / 176:.1f}%)", ha="left", va="center", fontsize=6.0, color=color)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 176)
    ax.set_xticks([0, 40, 80, 120, 160])
    ax.set_xlabel("Patients")
    ax.grid(False)
    clean_axis(ax)


def build_figure_2(data: dict[str, pd.DataFrame], counts: dict[str, int], out_dir: Path) -> None:
    fig = plt.figure(figsize=(mm_to_in(183), mm_to_in(150)), facecolor=WHITE)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0], height_ratios=[1.18, 0.82], left=0.16, right=0.985, top=0.95, bottom=0.08, wspace=0.22, hspace=0.66)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])
    draw_cumulative_incidence(ax_a, data["curve"], data["fixed"])
    draw_day28_states(ax_b, counts)
    draw_pathway_features(ax_c, counts)
    save_figure(
        fig,
        [ax_a, ax_b, ax_c],
        ["A", "B", "C"],
        out_dir / "Figure_2_course_and_outcomes",
        exemptions=[
            {
                "panels": ["A", "B", "C"],
                "checks": ["panel-width"],
                "reason": "Intentional two-row evidence architecture: cumulative incidence and day-28 states above a full-width pathway panel.",
            }
        ],
    )


def format_effect(hr: float, lo: float, hi: float) -> str:
    return f"{hr:.3f} ({lo:.3f}–{hi:.3f})"


def setup_forest_axis(ax, y: np.ndarray, values: np.ndarray, low: np.ndarray, high: np.ndarray, color: str) -> None:
    require_positive(values, "hazard-ratio estimates")
    require_positive(low, "hazard-ratio lower confidence limits")
    require_positive(high, "hazard-ratio upper confidence limits")
    ax.axvline(1.0, color=MID_GREY, ls=(0, (3, 2)), lw=0.9, zorder=1)
    for yy, value, lo, hi in zip(y, values, low, high):
        ax.errorbar(value, yy, xerr=[[value - lo], [hi - value]], fmt="o", ms=3.4, color=color, ecolor=color, elinewidth=1.1, capsize=2.0, zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(0.25, 2.7)
    ax.set_xticks([0.25, 0.5, 1.0, 2.0])
    ax.set_xticklabels(["0.25", "0.5", "1", "2"])
    ax.minorticks_off()
    ax.set_ylim(-0.8, len(y) - 0.2)
    ax.set_yticks([])
    ax.grid(axis="x", color=LIGHT_GREY, lw=0.5)
    clean_axis(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Hazard ratio")


def build_figure_3(models: pd.DataFrame, diagnostics: pd.DataFrame, out_dir: Path) -> None:
    models = models.set_index("parameter").loc[PARAMETER_ORDER].reset_index()
    y = np.arange(len(models))[::-1]
    fig = plt.figure(figsize=(mm_to_in(183), mm_to_in(158)), facecolor=WHITE)
    gs = fig.add_gridspec(1, 5, width_ratios=[1.72, 1.20, 1.15, 1.20, 1.15], left=0.025, right=0.985, top=0.82, bottom=0.13, wspace=0.07)
    ax_label = fig.add_subplot(gs[0, 0])
    ax_ctext = fig.add_subplot(gs[0, 1])
    ax_cforest = fig.add_subplot(gs[0, 2])
    ax_ftext = fig.add_subplot(gs[0, 3])
    ax_fforest = fig.add_subplot(gs[0, 4])

    for ax in [ax_label, ax_ctext, ax_ftext]:
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.8, len(y) - 0.2)
        ax.axis("off")

    for yy, parameter in zip(y, models["parameter"]):
        variable, unit = DISPLAY_LABELS[parameter]
        ax_label.text(0.0, yy + 0.12, variable, ha="left", va="bottom", fontsize=6.25, color=BLACK, fontweight="bold" if parameter == "peep_per_5" else "normal")
        ax_label.text(0.0, yy - 0.14, unit, ha="left", va="top", fontsize=6.0, color=GREY)

    for yy, (_, row) in zip(y, models.iterrows()):
        ax_ctext.text(0.5, yy, format_effect(row["csHR"], row["csHR_ci_low"], row["csHR_ci_high"]), ha="center", va="center", fontsize=6.0, color=NAVY)
        ax_ftext.text(0.5, yy, format_effect(row["sdHR"], row["sdHR_ci_low"], row["sdHR_ci_high"]), ha="center", va="center", fontsize=6.0, color=GREEN)

    setup_forest_axis(
        ax_cforest,
        y,
        models["csHR"].to_numpy(float),
        models["csHR_ci_low"].to_numpy(float),
        models["csHR_ci_high"].to_numpy(float),
        NAVY,
    )
    setup_forest_axis(
        ax_fforest,
        y,
        models["sdHR"].to_numpy(float),
        models["sdHR_ci_low"].to_numpy(float),
        models["sdHR_ci_high"].to_numpy(float),
        GREEN,
    )

    fig.text(0.025, 0.975, "Prespecified landmark associations with confirmed cessation start", ha="left", va="top", fontsize=9.0, fontweight="bold", color=BLACK)
    fig.text(0.025, 0.908, "Prespecified variable and comparison", ha="left", va="bottom", fontsize=6.5, fontweight="bold", color=BLACK)
    fig.text(0.43, 0.925, "Cause-specific Cox", ha="center", va="bottom", fontsize=8.0, fontweight="bold", color=NAVY)
    fig.text(0.43, 0.892, "Instantaneous event rate among patients still at risk", ha="center", va="bottom", fontsize=6.0, color=GREY)
    fig.text(0.43, 0.858, "HR (95% CI)", ha="center", va="bottom", fontsize=6.3, fontweight="bold", color=NAVY)
    fig.text(0.80, 0.925, "Fine–Gray", ha="center", va="bottom", fontsize=8.0, fontweight="bold", color=GREEN)
    fig.text(0.80, 0.892, "Association with the cumulative-incidence process", ha="center", va="bottom", fontsize=6.0, color=GREY)
    fig.text(0.80, 0.858, "sdHR (95% CI)", ha="center", va="bottom", fontsize=6.3, fontweight="bold", color=GREEN)
    fig.text(0.80, 0.835, "Supplementary estimand; not mutual validation", ha="center", va="bottom", fontsize=6.0, color=GREY)

    diag = dict(zip(diagnostics["metric"], diagnostics["value"]))
    assert float(diag["cox_global_ph_df"]) == 13
    assert np.isclose(float(diag["cox_global_ph_chisq"]), 30.04)
    assert np.isclose(float(diag["cox_global_ph_p"]), 0.00465)
    assert float(diag["fine_gray_proportionality_assessed"]) == 0
    fig.text(
        0.025,
        0.045,
        "† PEEP showed the clearest variable-level proportional-hazards departure. Global Cox PH test: χ²=30.04, df=13, P=0.00465.\n"
        "Fine–Gray pseudo-time residual global test: P=0.00066; constant sdHRs are supplementary summaries.",
        ha="left",
        va="bottom",
        fontsize=6.0,
        color=GREY,
        linespacing=1.15,
    )

    save_figure(
        fig,
        [ax_label, ax_ctext, ax_cforest, ax_ftext, ax_fforest],
        ["labels", "cox-text", "cox-forest", "fg-text", "fg-forest"],
        out_dir / "Figure_3_primary_models_integrated",
        exemptions=[
            {
                "panels": ["labels", "cox-text", "cox-forest", "fg-text", "fg-forest"],
                "checks": ["panel-width"],
                "reason": "Intentional evidence columns for variable labels, numerical estimates, and two forests.",
            }
        ],
    )


def build_figure_s1(ph: pd.DataFrame, diagnostics: pd.DataFrame, out_dir: Path) -> None:
    d = ph.set_index("parameter").loc[PARAMETER_ORDER].reset_index()
    require_positive(d["p_value"], "proportional-hazards P values")
    d["minus_log10_p"] = -np.log10(d["p_value"].astype(float))
    y = np.arange(len(d))[::-1]
    fig, ax = plt.subplots(figsize=(mm_to_in(183), mm_to_in(116)))
    for yy, (_, row) in zip(y, d.iterrows()):
        is_peep = row["parameter"] == "peep_per_5"
        marker = "s" if is_peep else "o"
        face = WHITE if is_peep else NAVY
        edge = GREY if is_peep else NAVY
        ax.hlines(yy, 0, row["minus_log10_p"], color=NAVY, lw=1.2)
        ax.scatter(row["minus_log10_p"], yy, marker=marker, s=28, facecolor=face, edgecolor=edge, linewidth=1.2, zorder=3)
    threshold = -np.log10(0.05)
    ax.axvline(threshold, color=MID_GREY, ls=(0, (3, 2)), lw=0.9)
    ax.text(threshold + 0.04, len(d) - 1.0, "P = 0.05", fontsize=6.0, color=GREY, va="top")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{DISPLAY_LABELS[p][0]} — {DISPLAY_LABELS[p][1]}" for p in d["parameter"]])
    ax.set_xlabel("−log10(variable-level proportional-hazards P value)")
    ax.set_xlim(0, max(4.6, float(d["minus_log10_p"].max()) + 0.35))
    ax.grid(axis="x", color=LIGHT_GREY, lw=0.5)
    clean_axis(ax)
    diag = dict(zip(diagnostics["metric"], diagnostics["value"]))
    ax.set_title(
        f"Variable-level proportional-hazards diagnostics\nGlobal test: χ²={float(diag['cox_global_ph_chisq']):.2f}, df={int(float(diag['cox_global_ph_df']))}, P={float(diag['cox_global_ph_p']):.5f}",
        loc="left",
        fontweight="bold",
        pad=7,
    )
    ax.text(0.0, -0.12, "Open square denotes PEEP, the clearest variable-level departure.", transform=ax.transAxes, ha="left", va="top", fontsize=6.0, color=GREY)
    fig.subplots_adjust(left=0.37, right=0.985, top=0.86, bottom=0.18)
    save_figure(fig, [ax], ["S1"], out_dir / "Figure_S1_PH_diagnostics")


def build_figure_s2(peep: pd.DataFrame, out_dir: Path) -> None:
    require_positive(peep[["hr", "hr_ci_low", "hr_ci_high"]].to_numpy(), "time-varying PEEP hazard ratios")
    fig, ax = plt.subplots(figsize=(mm_to_in(110), mm_to_in(84)))
    ax.fill_between(peep["day"], peep["hr_ci_low"], peep["hr_ci_high"], color=NAVY, alpha=0.14, linewidth=0)
    ax.plot(peep["day"], peep["hr"], color=NAVY, marker="o", ms=3.8, lw=1.5)
    ax.axhline(1.0, color=MID_GREY, ls=(0, (3, 2)), lw=0.9)
    ax.set_yscale("log")
    ax.set_yticks([0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10])
    ax.set_yticklabels(["0.05", "0.1", "0.2", "0.5", "1", "2", "5", "10"])
    ax.minorticks_off()
    ax.set_xticks([0, 1, 3, 7, 14, 28])
    ax.set_xlabel("Days since t_HMIS")
    ax.set_ylabel("Cause-specific HR per 5 cmH2O PEEP")
    ax.grid(axis="y", color=LIGHT_GREY, lw=0.5)
    clean_axis(ax)
    ax.set_title("Time-varying PEEP association", loc="left", fontweight="bold", pad=7)
    risk_text = "/".join(str(int(v)) for v in peep["n_at_risk"])
    day_text = "/".join(str(int(v)) for v in peep["day"])
    ax.text(0.0, -0.20, f"Risk set at days {day_text}: {risk_text}", transform=ax.transAxes, ha="left", va="top", fontsize=6.0, color=GREY)
    fig.subplots_adjust(left=0.18, right=0.97, top=0.90, bottom=0.25)
    save_figure(fig, [ax], ["S2"], out_dir / "Figure_S2_PEEP_time_varying")


def build_figure_s3(sensitivity: pd.DataFrame, out_dir: Path) -> None:
    d = sensitivity[
        (sensitivity["parameter"] == "any_vasopressor_exposed_6h")
        & (sensitivity["estimand"] == "cause-specific hazard")
        & (sensitivity["analysis"].isin(SENSITIVITY_ORDER))
    ].copy()
    d = d.set_index("analysis").loc[SENSITIVITY_ORDER].reset_index()
    require_positive(d[["hazard_ratio", "ci_low", "ci_high"]].to_numpy(), "sensitivity-analysis hazard ratios")
    y = np.arange(len(d))[::-1]
    fig, ax = plt.subplots(figsize=(mm_to_in(150), mm_to_in(135)))
    for yy, (_, row) in zip(y, d.iterrows()):
        primary = row["analysis"] == "PRIMARY_7D_MI_REFERENCE"
        ax.errorbar(
            row["hazard_ratio"],
            yy,
            xerr=[[row["hazard_ratio"] - row["ci_low"]], [row["ci_high"] - row["hazard_ratio"]]],
            fmt="o",
            ms=4.0,
            mfc=NAVY if primary else WHITE,
            mec=NAVY,
            mew=1.1,
            ecolor=NAVY,
            elinewidth=1.1,
            capsize=2,
        )
    ax.axvline(1.0, color=MID_GREY, ls=(0, (3, 2)), lw=0.9)
    ax.set_xscale("log")
    ax.set_xlim(0.18, 1.35)
    ax.set_xticks([0.2, 0.3, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0.2", "0.3", "0.5", "0.75", "1"])
    ax.minorticks_off()
    ax.set_yticks(y)
    ax.set_yticklabels([SENSITIVITY_LABELS[x] for x in d["analysis"]])
    ax.set_xlabel("Vasopressor cause-specific HR (95% CI)")
    ax.grid(axis="x", color=LIGHT_GREY, lw=0.5)
    clean_axis(ax)
    ax.set_title("Vasopressor association across analysis specifications", loc="left", fontweight="bold", pad=7)
    ax.text(0.0, -0.13, "Filled marker denotes the primary rule; open markers denote sensitivity analyses.", transform=ax.transAxes, ha="left", va="top", fontsize=6.0, color=GREY)
    fig.subplots_adjust(left=0.38, right=0.97, top=0.90, bottom=0.19)
    save_figure(fig, [ax], ["S3"], out_dir / "Figure_S3_vasopressor_sensitivity")


def build_figure_s4(stratified: pd.DataFrame, out_dir: Path) -> None:
    day28 = stratified[stratified["day"] == 28].copy()
    groups = [(["GCS-M 5", "GCS-M 6"], "Landmark GCS-M"), (["W1", "W2-W5"], "Target checkpoint")]
    fig, axes = plt.subplots(1, 2, figsize=(mm_to_in(183), mm_to_in(82)), sharex=True)
    for idx, (levels, title) in enumerate(groups):
        ax = axes[idx]
        d = day28.set_index("stratum").loc[levels].reset_index()
        y = np.arange(len(levels))[::-1]
        ax.errorbar(d["liberation_cif"], y + 0.10, xerr=[d["liberation_cif"] - d["liberation_ci_low"], d["liberation_ci_high"] - d["liberation_cif"]], fmt="o", color=GREEN, ecolor=GREEN, capsize=2, label="Confirmed cessation start")
        ax.errorbar(d["death_cif"], y - 0.10, xerr=[d["death_cif"] - d["death_ci_low"], d["death_ci_high"] - d["death_cif"]], fmt="s", color=CORAL, ecolor=CORAL, capsize=2, label="Death before event")
        ax.set_yticks(y)
        ax.set_yticklabels(levels)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Day-28 cumulative incidence")
        ax.set_title(title, loc="left", fontweight="bold", pad=7)
        ax.grid(axis="x", color=LIGHT_GREY, lw=0.5)
        clean_axis(ax)
        panel_label(ax, chr(ord("A") + idx))
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.11), ncol=2, frameon=False, fontsize=6.0)
    fig.text(0.5, 0.035, "Descriptive estimates only; no interaction or subgroup-effect test was performed.", ha="center", va="bottom", fontsize=6.0, color=GREY)
    fig.subplots_adjust(left=0.12, right=0.985, top=0.88, bottom=0.29, wspace=0.28)
    save_figure(fig, list(axes), ["A", "B"], out_dir / "Figure_S4_descriptive_day28_strata")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build redesigned HMIS liberation main and supplementary figures from aggregate-only source data.")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--main-output-dir", type=Path, required=True)
    parser.add_argument("--supp-output-dir", type=Path, required=True)
    args = parser.parse_args()

    data = load_inputs(args.source_dir)
    counts = assert_contract(data)
    build_figure_2(data, counts, args.main_output_dir)
    build_figure_3(data["models"], data["diagnostics"], args.main_output_dir)
    build_figure_s1(data["ph"], data["diagnostics"], args.supp_output_dir)
    build_figure_s2(data["peep"], args.supp_output_dir)
    build_figure_s3(data["sensitivity"], args.supp_output_dir)
    build_figure_s4(data["stratified"], args.supp_output_dir)
    print("Redesigned remaining figures completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
