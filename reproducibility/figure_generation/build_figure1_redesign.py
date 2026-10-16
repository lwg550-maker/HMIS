from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Circle, Ellipse, FancyArrowPatch, Polygon, Rectangle

from audit_panel_alignment import require_matplotlib_panel_alignment


MM_PER_INCH = 25.4
width_mm = 183.0
FIG_W_MM = 183.0
FIG_H_MM = 135.0

NAVY = "#173B5E"
GREEN = "#2F6B4F"
CORAL = "#C96552"
GREY = "#66717C"
LIGHT_GREY = "#D8DEE3"
VERY_LIGHT_GREY = "#F3F5F6"
BLACK = "#1B1F23"
WHITE = "#FFFFFF"


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


def read_keyed_counts(path: Path) -> dict[str, int]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["key"]: int(row["n"]) for row in rows}


def read_windows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def assert_contract(cohort: dict[str, int], outcome: dict[str, int], windows: list[dict[str, str]]) -> None:
    assert cohort["source_cohort"] == 474
    assert cohort["qualifying_transition"] == 221
    assert cohort["landmark_risk_set"] == 176
    assert cohort["not_concurrent_imv"] == 45
    assert cohort["no_qualifying_transition"] == 253
    assert 474 == 221 + 253
    assert 221 == 176 + 45
    assert 45 == cohort["between_intervals"] + cohort["at_or_after_final_interval"] + cohort["at_interval_end_boundary"]
    assert outcome["confirmed_event"] == 131
    assert outcome["death_before_event"] == 40
    assert outcome["no_event_by_day28"] == 5
    assert 176 == 131 + 40 + 5
    assert 131 == outcome["full_168h"] + outcome["alive_discharge_surrogate"]
    assert [row["checkpoint"] for row in windows] == ["B0", "W1", "W2", "W3", "W4", "W5"]
    for row in windows:
        assert float(row["search_end_h"]) - float(row["search_start_h"]) == 4.0
        assert row["left_open"].lower() == "true"
        assert row["right_closed"].lower() == "true"


def panel_label(ax, label: str) -> None:
    ax.text(-0.035, 1.015, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=9, fontweight="bold", color=BLACK)


def panel_title(ax, title: str, subtitle: str | None = None) -> None:
    ax.text(0.01, 0.985, title, transform=ax.transAxes, ha="left", va="top", fontsize=8.5, fontweight="bold", color=BLACK)
    if subtitle:
        ax.text(0.01, 0.945, subtitle, transform=ax.transAxes, ha="left", va="top", fontsize=6.0, color=GREY)


def orthogonal(ax, start: tuple[float, float], end: tuple[float, float], color: str, lw: float = 1.3, arrow: bool = False) -> None:
    x0, y0 = start
    x1, y1 = end
    mid_y = y1
    ax.add_line(Line2D([x0, x0, x1], [y0, mid_y, mid_y], color=color, lw=lw, zorder=1))
    if arrow:
        ax.add_patch(FancyArrowPatch((x1 - 0.001, y1), (x1, y1), arrowstyle="-|>", mutation_scale=7, lw=lw, color=color, zorder=2))


def database_icon(ax, x: float, y: float, scale: float, color: str) -> None:
    w, h = 0.12 * scale, 0.10 * scale
    ax.add_patch(Ellipse((x, y + h / 2), w, h * 0.35, fill=False, ec=color, lw=1.0))
    ax.add_patch(Arc((x, y), w, h * 0.35, theta1=180, theta2=360, ec=color, lw=1.0))
    ax.add_line(Line2D([x - w / 2, x - w / 2], [y, y + h / 2], color=color, lw=1.0))
    ax.add_line(Line2D([x + w / 2, x + w / 2], [y, y + h / 2], color=color, lw=1.0))
    ax.add_patch(Arc((x, y + h * 0.23), w, h * 0.35, theta1=180, theta2=360, ec=color, lw=0.8))


def brain_icon(ax, x: float, y: float, scale: float, color: str) -> None:
    r = 0.027 * scale
    centers = [(-1.1, 0.15), (-0.55, 0.70), (0.05, 0.82), (0.65, 0.55), (1.0, 0.0), (0.55, -0.55), (-0.2, -0.70), (-0.85, -0.45)]
    for dx, dy in centers:
        ax.add_patch(Circle((x + dx * r, y + dy * r), r * 0.78, fill=False, ec=color, lw=0.8))
    ax.add_line(Line2D([x, x], [y - r * 1.2, y + r * 1.2], color=color, lw=0.7))


def ventilator_icon(ax, x: float, y: float, scale: float, color: str) -> None:
    w, h = 0.11 * scale, 0.08 * scale
    ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, fill=False, ec=color, lw=1.0))
    ax.add_line(Line2D([x - w * 0.35, x - w * 0.05, x + w * 0.05, x + w * 0.35], [y, y, y + h * 0.22, y + h * 0.22], color=color, lw=0.9))
    ax.add_line(Line2D([x + w / 2, x + w * 0.72], [y, y], color=color, lw=1.0))
    ax.add_line(Line2D([x - w * 0.25, x - w * 0.25], [y - h / 2, y - h * 0.72], color=color, lw=0.8))
    ax.add_line(Line2D([x + w * 0.25, x + w * 0.25], [y - h / 2, y - h * 0.72], color=color, lw=0.8))


def clock_icon(ax, x: float, y: float, scale: float, color: str) -> None:
    r = 0.024 * scale
    ax.add_patch(Circle((x, y), r, fill=False, ec=color, lw=1.0))
    ax.add_line(Line2D([x, x], [y, y + r * 0.58], color=color, lw=0.9))
    ax.add_line(Line2D([x, x + r * 0.48], [y, y - r * 0.20], color=color, lw=0.9))


def hospital_icon(ax, x: float, y: float, scale: float, color: str) -> None:
    w, h = 0.055 * scale, 0.065 * scale
    ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, fill=False, ec=color, lw=0.9))
    ax.add_line(Line2D([x - w * 0.18, x + w * 0.18], [y + h * 0.14, y + h * 0.14], color=color, lw=0.9))
    ax.add_line(Line2D([x, x], [y - h * 0.03, y + h * 0.31], color=color, lw=0.9))


def outcome_marker(ax, x: float, y: float, color: str, shape: str) -> None:
    if shape == "diamond":
        ax.add_patch(Polygon([[x, y + 0.013], [x + 0.013, y], [x, y - 0.013], [x - 0.013, y]], closed=True, fc=color, ec=color))
    elif shape == "square":
        ax.add_patch(Rectangle((x - 0.011, y - 0.011), 0.022, 0.022, fc=color, ec=color))
    else:
        ax.add_patch(Circle((x, y), 0.011, fc=color, ec=color))


def draw_panel_a(ax, c: dict[str, int]) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel_label(ax, "A")
    panel_title(ax, "Cohort conditioning", "The primary path is 474 → 221 → 176")

    spine_x = 0.22
    ys = [0.82, 0.57, 0.28]
    ax.add_line(Line2D([spine_x, spine_x], [ys[-1], ys[0]], color=NAVY, lw=2.2, zorder=1))
    for y in ys:
        ax.add_patch(Circle((spine_x, y), 0.015, fc=WHITE, ec=NAVY if y != ys[-1] else GREEN, lw=1.5, zorder=3))

    database_icon(ax, 0.09, 0.82, 1.0, NAVY)
    ax.text(0.29, 0.845, f"{c['source_cohort']}", ha="left", va="center", fontsize=15, fontweight="bold", color=NAVY)
    ax.text(0.29, 0.795, "Acute brain injury\nenriched cohort", ha="left", va="top", fontsize=6.5, color=BLACK, linespacing=1.15)

    ax.text(0.29, 0.595, f"{c['qualifying_transition']}", ha="left", va="center", fontsize=13, fontweight="bold", color=NAVY)
    ax.text(0.29, 0.548, "First qualifying sampled\nGCS-M 1–4 → 5–6 transition", ha="left", va="top", fontsize=6.2, color=BLACK, linespacing=1.15)

    ax.text(0.29, 0.305, f"{c['landmark_risk_set']}", ha="left", va="center", fontsize=15, fontweight="bold", color=GREEN)
    ax.text(0.29, 0.258, "IMV active at t_HMIS\nconditional landmark risk set", ha="left", va="top", fontsize=6.2, color=BLACK, linespacing=1.15)

    orthogonal(ax, (spine_x, 0.735), (0.66, 0.735), GREY, lw=0.9)
    ax.text(0.69, 0.752, f"{c['no_qualifying_transition']}", ha="left", va="center", fontsize=10, fontweight="bold", color=GREY)
    ax.text(0.69, 0.700, "No qualifying adjacent\ncheckpoint transition", ha="left", va="top", fontsize=6.0, color=GREY, linespacing=1.1)

    orthogonal(ax, (spine_x, 0.43), (0.66, 0.43), GREY, lw=0.9)
    ax.text(0.69, 0.452, f"{c['not_concurrent_imv']}", ha="left", va="center", fontsize=10, fontweight="bold", color=GREY)
    ax.text(0.69, 0.410, "Not concurrently\ninvasively ventilated", ha="left", va="top", fontsize=6.0, color=GREY, linespacing=1.1)
    ax.text(0.69, 0.350, f"{c['between_intervals']} between intervals", ha="left", va="top", fontsize=6.0, color=GREY)
    ax.text(0.69, 0.315, f"{c['at_or_after_final_interval']} at/after final interval", ha="left", va="top", fontsize=6.0, color=GREY)
    ax.text(0.69, 0.280, f"{c['at_interval_end_boundary']} at interval-end boundary", ha="left", va="top", fontsize=6.0, color=GREY)

    ax.text(0.02, 0.055, "Main path", ha="left", va="center", fontsize=6.0, fontweight="bold", color=NAVY)
    ax.add_line(Line2D([0.02, 0.18], [0.022, 0.022], color=NAVY, lw=2.2))
    ax.add_line(Line2D([0.22, 0.30], [0.022, 0.022], color=GREY, lw=1.0))
    ax.text(0.32, 0.022, "Conditioned out", ha="left", va="center", fontsize=6.0, color=GREY)


def draw_open_closed_window(ax, x_end: float, y: float, width: float = 0.115) -> None:
    x_start = x_end - width
    ax.add_line(Line2D([x_start, x_end], [y, y], color=NAVY, lw=2.0, zorder=2))
    ax.add_patch(Circle((x_start, y), 0.008, fc=WHITE, ec=NAVY, lw=1.0, zorder=3))
    ax.add_patch(Circle((x_end, y), 0.008, fc=NAVY, ec=NAVY, lw=1.0, zorder=3))


def draw_panel_b(ax, windows: list[dict[str, str]]) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel_label(ax, "B")
    panel_title(ax, "Operational landmark at actual charttime", "Nominal checkpoints are schematic and not drawn to elapsed-time scale")

    xs = [0.08, 0.245, 0.41, 0.575, 0.74, 0.905]
    axis_y = 0.68
    ax.add_line(Line2D([0.055, 0.93], [axis_y, axis_y], color=GREY, lw=1.0))
    for row, x in zip(windows, xs):
        ax.add_line(Line2D([x, x], [axis_y - 0.018, axis_y + 0.018], color=GREY, lw=0.9))
        ax.text(x, axis_y + 0.042, row["checkpoint"], ha="center", va="bottom", fontsize=6.5, fontweight="bold", color=BLACK)
        ax.text(x, axis_y - 0.047, f"{int(float(row['nominal_end_h']))} h", ha="center", va="top", fontsize=6.0, color=GREY)
        draw_open_closed_window(ax, x, 0.60)
    ax.text(0.055, 0.552, "4-h GCS-M search window", ha="left", va="top", fontsize=6.0, color=GREY)
    ax.text(xs[3] - 0.010, 0.552, "(end − 4 h, end]", ha="left", va="top", fontsize=6.0, color=GREY)
    ax.add_patch(Circle((0.065, 0.502), 0.007, fc=WHITE, ec=NAVY, lw=0.9))
    ax.text(0.081, 0.502, "Open left", ha="left", va="center", fontsize=6.0, color=NAVY)
    ax.add_patch(Circle((0.220, 0.502), 0.007, fc=NAVY, ec=NAVY, lw=0.9))
    ax.text(0.236, 0.502, "Closed right", ha="left", va="center", fontsize=6.0, color=NAVY)

    low_x = xs[2] - 0.055
    high_x = xs[3] - 0.045
    record_y = 0.83
    ax.add_patch(Circle((low_x, record_y), 0.015, fc=NAVY, ec=WHITE, lw=0.8, zorder=4))
    ax.add_patch(Circle((high_x, record_y), 0.017, fc=NAVY, ec=WHITE, lw=0.8, zorder=4))
    ax.add_patch(FancyArrowPatch((low_x + 0.025, record_y), (high_x - 0.025, record_y), arrowstyle="-|>", mutation_scale=8, lw=1.2, color=NAVY))
    ax.text(low_x, record_y + 0.052, "GCS-M 1–4", ha="center", va="bottom", fontsize=6.2, fontweight="bold", color=NAVY)
    ax.text(high_x, record_y + 0.052, "GCS-M 5–6", ha="center", va="bottom", fontsize=6.2, fontweight="bold", color=NAVY)
    ax.text((low_x + high_x) / 2, record_y + 0.018, "adjacent checkpoints", ha="center", va="bottom", fontsize=6.0, color=GREY)

    ax.add_line(Line2D([high_x, high_x], [0.20, record_y - 0.018], color=NAVY, lw=1.0, ls=(0, (2, 2))))
    ax.text(high_x + 0.030, 0.772, "t_HMIS = actual charttime\n(before the nominal W3 end)", ha="left", va="center", fontsize=6.0, color=NAVY, linespacing=1.15)

    unknown_y = 0.405
    ux = [0.17, 0.27, 0.37]
    ax.text(0.055, unknown_y + 0.045, "Invalid sequence", ha="left", va="center", fontsize=6.0, fontweight="bold", color=CORAL)
    for x, label, face in zip(ux, ["1–4", "?", "5–6"], [NAVY, WHITE, NAVY]):
        ax.add_patch(Circle((x, unknown_y), 0.011, fc=face, ec=CORAL if label == "?" else NAVY, lw=1.0))
        ax.text(x, unknown_y - 0.027, label, ha="center", va="top", fontsize=6.0, color=BLACK)
    ax.add_line(Line2D([ux[0] + 0.015, ux[1] - 0.015], [unknown_y, unknown_y], color=GREY, lw=0.8))
    ax.add_line(Line2D([ux[1] + 0.015, ux[2] - 0.015], [unknown_y, unknown_y], color=GREY, lw=0.8))
    ax.add_line(Line2D([ux[1] - 0.010, ux[1] + 0.010], [unknown_y - 0.010, unknown_y + 0.010], color=CORAL, lw=1.2))
    ax.add_line(Line2D([ux[1] - 0.010, ux[1] + 0.010], [unknown_y + 0.010, unknown_y - 0.010], color=CORAL, lw=1.2))
    ax.text(high_x + 0.035, unknown_y, "Unknown checkpoints\nare not bridged", ha="left", va="center", fontsize=6.0, color=CORAL, linespacing=1.08)

    vent_y = 0.20
    ventilator_icon(ax, 0.07, vent_y + 0.002, 0.9, NAVY)
    vent_start, vent_end = 0.15, 0.80
    ax.add_line(Line2D([vent_start, vent_end], [vent_y, vent_y], color=LIGHT_GREY, lw=7.0, zorder=1))
    ax.add_line(Line2D([vent_start, high_x], [vent_y, vent_y], color=NAVY, lw=3.0, zorder=2))
    ax.add_line(Line2D([high_x, vent_end], [vent_y, vent_y], color=GREEN, lw=3.0, zorder=2))
    ax.add_line(Line2D([vent_start, vent_start], [vent_y - 0.018, vent_y + 0.018], color=NAVY, lw=1.2, zorder=4))
    ax.add_patch(Circle((vent_end, vent_y), 0.010, fc=WHITE, ec=GREY, lw=1.1, zorder=4))
    ax.add_patch(Circle((high_x, vent_y), 0.018, fc=GREEN, ec=WHITE, lw=1.0, zorder=5))
    ax.text((vent_start + vent_end) / 2, vent_y - 0.052, "vent_start ≤ t_HMIS < vent_end", ha="center", va="top", fontsize=6.1, fontweight="bold", color=BLACK)
    ax.text(vent_start, vent_y + 0.038, "concurrent IMV interval", ha="left", va="bottom", fontsize=6.0, color=GREY)
    ax.add_patch(FancyArrowPatch((high_x + 0.022, vent_y), (0.855, vent_y), arrowstyle="-|>", mutation_scale=9, lw=1.4, color=GREEN))
    ax.text(0.862, vent_y + 0.050, "N=176", ha="left", va="bottom", fontsize=9.0, fontweight="bold", color=GREEN)
    ax.text(0.862, vent_y + 0.018, "risk set", ha="left", va="top", fontsize=6.0, color=GREEN)


def d2x(day: float) -> float:
    return 0.065 + (day / 35.0) * 0.87


def draw_panel_c(ax, o: dict[str, int]) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel_label(ax, "C")
    panel_title(ax, "Candidate cessation and confirmation", "Illustrative path; timing is schematic, not patient-derived")

    base_y = 0.57
    ax.add_line(Line2D([d2x(0), d2x(35)], [base_y, base_y], color=GREY, lw=1.1))
    for day in [0, 7, 14, 21, 28, 35]:
        x = d2x(day)
        ax.add_line(Line2D([x, x], [base_y - 0.015, base_y + 0.015], color=GREY, lw=0.8))
        label_x = x - 0.014 if day == 28 else x
        label_ha = "right" if day == 28 else "center"
        ax.text(label_x, base_y - 0.035, str(day), ha=label_ha, va="top", fontsize=6.0, color=GREY)
    ax.text(d2x(17.5), base_y - 0.095, "Days after t_HMIS", ha="center", va="top", fontsize=6.0, color=GREY)
    ax.add_patch(Circle((d2x(0), base_y), 0.012, fc=NAVY, ec=NAVY))
    ax.text(d2x(0), base_y + 0.035, "t_HMIS", ha="left", va="bottom", fontsize=6.0, fontweight="bold", color=NAVY)

    c1, r1 = 3.0, 7.0
    c2, confirm2 = 11.5, 18.5
    clock_icon(ax, d2x(c1) - 0.022, base_y + 0.15, 0.8, CORAL)
    ax.add_patch(Circle((d2x(c1), base_y), 0.013, fc=WHITE, ec=CORAL, lw=1.4))
    ax.text(d2x(c1), base_y + 0.195, "C1", ha="center", va="bottom", fontsize=6.0, fontweight="bold", color=CORAL)
    ax.add_line(Line2D([d2x(c1), d2x(c1 + 7)], [base_y + 0.14, base_y + 0.14], color=CORAL, lw=2.0))
    ax.text((d2x(c1) + d2x(c1 + 7)) / 2, base_y + 0.166, "+168 h", ha="center", va="bottom", fontsize=6.0, color=CORAL)
    ax.add_patch(Polygon([[d2x(r1), base_y + 0.014], [d2x(r1) + 0.012, base_y - 0.010], [d2x(r1) - 0.012, base_y - 0.010]], closed=True, fc=CORAL, ec=CORAL))
    ax.text(d2x(r1), base_y - 0.070, "IMV resumes; C1 fails\nrisk continues", ha="center", va="top", fontsize=6.0, color=CORAL, linespacing=1.08)

    ax.add_patch(Circle((d2x(c2), base_y), 0.013, fc=WHITE, ec=GREEN, lw=1.5))
    ax.text(d2x(c2), base_y + 0.035, "C2", ha="center", va="bottom", fontsize=6.0, fontweight="bold", color=GREEN)
    ax.add_line(Line2D([d2x(c2), d2x(confirm2)], [base_y + 0.14, base_y + 0.14], color=GREEN, lw=2.2))
    ax.text((d2x(c2) + d2x(confirm2)) / 2, base_y + 0.166, "+168 h", ha="center", va="bottom", fontsize=6.0, color=GREEN)
    check_x, check_y = d2x(confirm2), base_y + 0.155
    ax.add_line(Line2D([check_x - 0.010, check_x - 0.002], [check_y, check_y - 0.010], color=GREEN, lw=1.8))
    ax.add_line(Line2D([check_x - 0.002, check_x + 0.014], [check_y - 0.010, check_y + 0.012], color=GREEN, lw=1.8))
    ax.add_patch(FancyArrowPatch((d2x(confirm2), base_y + 0.215), (d2x(c2), base_y + 0.215), arrowstyle="-|>", mutation_scale=8, lw=1.0, color=GREEN))
    ax.text((d2x(c2) + d2x(confirm2)) / 2, base_y + 0.238, "event time backdated to C2 start", ha="center", va="bottom", fontsize=6.0, color=GREEN)

    day28_x = d2x(28)
    day35_x = d2x(35)
    ax.add_line(Line2D([day28_x, day28_x], [0.48, 0.84], color=GREY, lw=0.9, ls=(0, (3, 2))))
    ax.add_line(Line2D([day28_x, day28_x], [0.08, 0.17], color=GREY, lw=0.9, ls=(0, (3, 2))))
    ax.text(day28_x, 0.85, "Day 28 boundary", ha="center", va="bottom", fontsize=6.0, color=GREY)
    ax.add_line(Line2D([d2x(21), day28_x], [0.13, 0.13], color=NAVY, lw=2.0))
    ax.add_line(Line2D([day28_x, day35_x], [0.13, 0.13], color=GREY, lw=2.0, ls=(0, (2, 2))))
    ax.text(d2x(21.8), 0.166, "candidate start\nallowed", ha="center", va="bottom", fontsize=6.0, color=NAVY, linespacing=1.02)
    ax.text(d2x(33.3), 0.166, "confirm only\n(read to day 35)", ha="center", va="bottom", fontsize=6.0, color=GREY, linespacing=1.25)
    ax.text(0.50, 0.055, "Post-day-28 death may invalidate a pending candidate,\nbut is not a new competing death.", ha="center", va="top", fontsize=6.0, color=GREY, linespacing=1.08)

    summary_y = 0.37
    outcome_marker(ax, 0.08, summary_y, GREEN, "circle")
    ax.text(0.105, summary_y + 0.012, f"Confirmed event  {o['confirmed_event']}/{176}", ha="left", va="bottom", fontsize=6.2, fontweight="bold", color=GREEN)
    ax.text(0.105, summary_y - 0.020, f"{o['full_168h']} full 168-h  |  {o['alive_discharge_surrogate']} alive-discharge surrogate", ha="left", va="top", fontsize=6.0, color=GREEN)
    hospital_icon(ax, 0.90, summary_y - 0.007, 0.8, GREEN)

    outcome_marker(ax, 0.08, 0.28, CORAL, "diamond")
    ax.text(0.105, 0.28, f"Death before a confirmed event  {o['death_before_event']}/{176}", ha="left", va="center", fontsize=6.0, color=CORAL)

    outcome_marker(ax, 0.08, 0.21, GREY, "square")
    ax.text(0.105, 0.21, f"No event or death by day 28  {o['no_event_by_day28']}/{176}", ha="left", va="center", fontsize=6.0, color=GREY)


def build_figure(source_dir: Path, out_dir: Path) -> Path:
    cohort = read_keyed_counts(source_dir / "Figure_1_cohort_counts.csv")
    outcome = read_keyed_counts(source_dir / "Figure_1_outcome_counts.csv")
    windows = read_windows(source_dir / "Figure_1_checkpoint_windows.csv")
    assert_contract(cohort, outcome, windows)

    fig = plt.figure(figsize=(FIG_W_MM / MM_PER_INCH, FIG_H_MM / MM_PER_INCH), facecolor=WHITE)
    gs = fig.add_gridspec(1, 3, width_ratios=[0.87, 1.66, 1.30], left=0.025, right=0.985, bottom=0.055, top=0.96, wspace=0.075)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    draw_panel_a(axes[0], cohort)
    draw_panel_b(axes[1], windows)
    draw_panel_c(axes[2], outcome)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / "Figure_1_redesign_draft"
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig,
        axes=axes,
        panel_ids=["A", "B", "C"],
        json_out=stem.with_suffix(".alignment.json"),
        overlay_svg=stem.with_suffix(".alignment.svg"),
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=False,
        strict=True,
        exemptions=[
            {
                "panels": ["A", "B", "C"],
                "checks": ["panel-width"],
                "reason": "Intentional asymmetric evidence hierarchy: the operational landmark timeline is the hero panel.",
            }
        ],
    )
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    fig.savefig(stem.with_suffix(".tiff"), dpi=600)
    plt.close(fig)
    return stem


def main() -> int:
    parser = argparse.ArgumentParser(description="Build redesigned Figure 1 draft from aggregate-only source data.")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    stem = build_figure(args.source_dir, args.output_dir)
    print(f"Wrote Figure 1 draft: {stem}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
