import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import pandas as pd
import numpy as np
from PIL import Image
from io import BytesIO
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from streamlit_image_coordinates import streamlit_image_coordinates

# ==========================
# Page Configuration
# ==========================
st.set_page_config(layout="wide", page_title="Pass Map Dashboard (Interactive)")

# ==========================
# Configuration
# ==========================
st.title("Pass Map Dashboard")

FIELD_X, FIELD_Y = 120.0, 80.0       # StatsBomb pitch dimensions
HALF_LINE_X = FIELD_X / 2             # 60
FINAL_THIRD_LINE_X = 80

BOX_X_MIN = 102
BOX_Y_MIN = 18
BOX_Y_MAX = 62

GOAL_X = 120
GOAL_Y = 40

# Lanes for Switch Pass (StatsBomb pitch: y goes from 0 to 80)
LANE_LEFT_MIN = 53.33
LANE_RIGHT_MAX = 26.67

# xT grid dimensions
NX, NY = 16, 12

# Colours
COLOR_SUCCESS = "#B0B0B0"
COLOR_FAIL = "#D45B5B"
COLOR_PROGRESSIVE = "#2F80ED"
COLOR_SWITCH = "#DAA520"
COLOR_FORWARD = "#27AE60"

# ==========================
# xT GRID
# ==========================
x_progress = np.linspace(0.01, 1.00, NX)
y_central = 1 - np.abs(np.linspace(0, 1, NY) - 0.5) * 2

XT_GRID = np.zeros((NY, NX))
for iy in range(NY):
    for ix in range(NX):
        XT_GRID[iy, ix] = 0.80 * x_progress[ix] + 0.20 * x_progress[ix] * y_central[iy]
XT_GRID = (XT_GRID - XT_GRID.min()) / (XT_GRID.max() - XT_GRID.min() + 1e-9)


def zone_index(x, y):
    x = np.clip(x, 0, FIELD_X - 1e-9)
    y = np.clip(y, 0, FIELD_Y - 1e-9)
    ix = int((x / FIELD_X) * NX)
    iy = int((y / FIELD_Y) * NY)
    return ix, iy


def xt_value(x, y):
    ix, iy = zone_index(x, y)
    return XT_GRID[iy, ix]


# ==========================
# DATA
# ==========================
matches_data = {
    "Vs Connecticut": [
        ("PASS WON", 26.75, 68.34, 8.97, 51.05, None),
        ("PASS WON", 31.24, 51.22, 34.57, 72.50, None),
        ("PASS WON", 36.06, 46.90, 44.37, 57.04, None),
        ("PASS WON", 48.36, 64.02, 58.17, 51.72, None),
        ("PASS WON", 58.17, 64.02, 62.49, 55.21, None),
        ("PASS WON", 54.51, 49.72, 64.82, 61.69, None),
        ("PASS WON", 42.21, 70.84, 34.90, 76.49, None),
        ("PASS WON", 43.54, 75.32, 36.73, 67.84, None),
        ("PASS WON", 32.24, 53.96, 6.81, 38.50, None),
        ("PASS WON", 33.57, 65.77, 36.56, 75.57, None),
        ("PASS WON", 37.39, 61.11, 43.04, 75.41, None),
        ("PASS WON", 65.49, 53.63, 56.18, 70.42, None),
        ("PASS WON", 55.68, 48.15, 46.87, 30.86, None),
        ("PASS WON", 52.02, 22.05, 46.70, 41.99, None),
        ("PASS WON", 62.16, 35.51, 71.80, 35.18, None),
        ("PASS WON", 54.02, 33.35, 63.99, 22.55, None),
        ("PASS WON", 60.00, 22.21, 76.62, 32.85, None),
        ("PASS WON", 87.10, 9.41, 77.45, 16.23, None),
        ("PASS WON", 62.66, 20.05, 117.18, 8.25, None),
        ("PASS WON", 98.90, 43.49, 103.22, 47.15, None),
        ("PASS WON", 70.31, 45.98, 82.28, 60.11, None),
        ("PASS WON", 85.10, 75.24, 101.39, 74.08, None),
        ("PASS WON", 53.18, 67.59, 39.05, 59.62, None),
        ("PASS WON", 55.18, 49.64, 54.85, 13.07, None),
        ("PASS WON", 68.64, 19.22, 49.03, 24.37, None),
        ("PASS WON", 53.35, 22.71, 59.34, 30.19, None),
        ("PASS WON", 44.37, 24.71, 40.05, 46.82, None),
        ("PASS WON", 43.88, 39.34, 41.38, 73.08, None),
        ("PASS WON", 56.84, 53.46, 70.81, 76.24, None),
        ("PASS WON", 82.77, 12.24, 91.42, 4.59, None),
        ("PASS WON", 108.04, 11.74, 115.69, 58.29, None),
        ("PASS WON", 93.08, 3.93, 111.03, 13.74, None),
        ("PASS WON", 84.60, 17.89, 96.74, 22.05, None),
        ("PASS WON", 58.34, 16.06, 65.65, 2.43, None),
        ("PASS WON", 52.02, 8.58, 44.37, 15.73, None),
        ("PASS WON", 61.00, 23.21, 49.36, 15.23, None),
        ("PASS WON", 32.74, 30.69, 50.03, 33.02, None),
        ("PASS WON", 51.85, 33.68, 60.66, 40.00, None),
        ("PASS WON", 79.95, 60.45, 98.23, 60.28, None),
        ("PASS WON", 31.24, 52.14, 39.05, 72.08, None),
        ("PASS WON", 39.72, 48.98, 33.40, 57.62, None),
        ("PASS WON", 70.64, 51.47, 61.00, 51.64, None),
        ("PASS LOST", 53.35, 19.55, 73.96, 11.24, None),
        ("PASS LOST", 63.82, 20.55, 88.76, 22.55, None),
        ("PASS LOST", 85.60, 27.86, 94.41, 37.17, None),
        ("PASS LOST", 77.79, 27.53, 96.41, 25.37, None),
        ("PASS LOST", 91.09, 27.86, 109.54, 50.47, None),
        ("PASS LOST", 58.17, 26.04, 95.41, 40.33, None),
        ("PASS LOST", 53.35, 28.53, 73.80, 27.86, None),
        ("PASS LOST", 53.35, 34.02, 84.60, 58.62, None),
        ("PASS LOST", 56.18, 49.48, 97.07, 62.11, None),
        ("PASS LOST", 34.23, 74.91, 65.65, 78.57, None),
    ],
    "Vs Nashville": [
        ("PASS WON", 21.27, 14.23, 29.25, 31.02, None),
        ("PASS WON", 29.41, 23.38, 34.40, 64.60, None),
        ("PASS WON", 41.55, 39.67, 41.88, 6.92, None),
        ("PASS WON", 44.54, 32.52, 43.54, 14.23, None),
        ("PASS WON", 23.59, 56.46, 34.57, 47.48, None),
        ("PASS WON", 30.58, 64.44, 21.10, 49.48, None),
        ("PASS WON", 33.24, 59.78, 44.04, 71.75, None),
        ("PASS WON", 33.07, 56.79, 49.53, 69.59, None),
        ("PASS WON", 61.50, 71.58, 54.68, 75.57, None),
        ("PASS WON", 63.16, 50.81, 78.45, 67.26, None),
        ("PASS WON", 63.49, 76.90, 84.44, 62.77, None),
        ("PASS WON", 76.96, 56.96, 86.93, 57.79, None),
        ("PASS WON", 82.61, 59.12, 96.41, 68.43, None),
        ("PASS WON", 79.78, 35.35, 106.21, 11.74, None),
        ("PASS WON", 45.37, 49.64, 40.72, 32.02, None),
        ("PASS LOST", 78.62, 64.94, 96.57, 67.10, None),
        ("PASS LOST", 85.43, 68.76, 106.05, 77.74, None),
    ],
    "Vs Seongnam": [
        ("PASS WON", 28.08, 28.53, 29.75, 8.25, None),
        ("PASS WON", 33.74, 26.54, 29.41, 43.82, None),
        ("PASS WON", 28.08, 47.15, 31.57, 64.60, None),
        ("PASS WON", 39.39, 43.82, 51.69, 53.46, None),
        ("PASS WON", 43.88, 46.15, 55.84, 40.66, None),
        ("PASS WON", 47.03, 49.97, 44.04, 28.03, None),
        ("PASS WON", 47.53, 50.81, 71.97, 33.18, None),
        ("PASS WON", 67.65, 52.63, 64.32, 33.85, None),
        ("PASS WON", 73.63, 65.10, 69.31, 73.25, None),
        ("PASS WON", 77.29, 63.27, 79.12, 72.91, None),
        ("PASS WON", 81.61, 56.62, 93.91, 73.75, None),
        ("PASS WON", 86.43, 66.43, 81.78, 54.96, None),
        ("PASS WON", 111.03, 71.42, 99.56, 67.59, None),
        ("PASS WON", 89.76, 59.62, 97.74, 48.98, None),
        ("PASS WON", 88.43, 52.47, 96.41, 74.24, None),
        ("PASS WON", 87.93, 50.97, 77.12, 27.70, None),
        ("PASS WON", 81.61, 53.63, 74.30, 27.03, None),
        ("PASS WON", 79.28, 51.14, 94.91, 70.42, None),
        ("PASS WON", 52.85, 32.85, 65.49, 25.37, None),
        ("PASS WON", 82.77, 33.18, 69.31, 47.65, None),
        ("PASS LOST", 72.14, 16.56, 78.45, 1.60, None),
        ("PASS LOST", 79.62, 27.53, 97.07, 47.98, None),
        ("PASS LOST", 91.75, 50.14, 109.70, 65.77, None),
        ("PASS LOST", 96.41, 56.79, 107.04, 67.26, None),
    ],
    "Vs Red Bull": [
        ("PASS WON", 39.39, 19.39, 52.35, 4.76, None),
        ("PASS WON", 63.82, 7.92, 72.63, 1.43, None),
        ("PASS WON", 70.47, 11.91, 80.95, 13.74, None),
        ("PASS WON", 64.49, 22.55, 97.24, 10.24, None),
        ("PASS WON", 32.07, 35.51, 43.04, 28.20, None),
        ("PASS WON", 53.52, 46.32, 54.02, 33.68, None),
        ("PASS WON", 77.12, 48.64, 84.94, 50.14, None),
        ("PASS WON", 78.12, 52.47, 117.52, 69.42, None),
        ("PASS WON", 88.76, 65.93, 97.40, 76.74, None),
        ("PASS WON", 82.61, 69.26, 86.60, 77.40, None),
        ("PASS WON", 78.62, 66.26, 79.62, 78.40, None),
        ("PASS WON", 83.61, 75.91, 62.49, 57.12, None),
        ("PASS WON", 34.40, 50.14, 88.76, 75.41, None),
        ("PASS WON", 56.68, 64.27, 78.29, 64.27, None),
        ("PASS WON", 51.85, 73.25, 54.18, 78.07, None),
        ("PASS WON", 41.05, 57.45, 46.04, 74.91, None),
        ("PASS WON", 37.39, 60.61, 41.71, 73.91, None),
        ("PASS WON", 30.41, 63.44, 36.89, 77.40, None),
        ("PASS WON", 26.09, 63.94, 28.42, 76.74, None),
        ("PASS WON", 22.43, 56.62, 22.10, 76.41, None),
        ("PASS WON", 33.90, 64.77, 25.42, 73.58, None),
        ("PASS LOST", 41.88, 42.49, 56.18, 52.97, None),
        ("PASS LOST", 37.56, 41.16, 46.37, 53.96, None),
        ("PASS LOST", 54.68, 56.96, 54.85, 64.44, None),
        ("PASS LOST", 51.69, 68.43, 66.15, 76.57, None),
    ],
}

# ==========================
# Helpers
# ==========================
def has_video_value(v) -> bool:
    return pd.notna(v) and str(v).strip() != ""


def distance_to_goal(x, y):
    return np.sqrt((GOAL_X - x) ** 2 + (GOAL_Y - y) ** 2)


def get_lane(y):
    if y >= LANE_LEFT_MIN:
        return "left"
    elif y < LANE_RIGHT_MAX:
        return "right"
    else:
        return "center"


def is_switch_pass(x_start, y_start, y_end) -> bool:
    if x_start >= FINAL_THIRD_LINE_X:
        return False
    lane_start = get_lane(y_start)
    lane_end = get_lane(y_end)
    if lane_start == "left" and lane_end == "right":
        return True
    if lane_start == "right" and lane_end == "left":
        return True
    return False


def is_progressive_pass_original(x_start, y_start, x_end, y_end) -> bool:
    """Original progressive rule (25% closer to goal, x_start >= 35)."""
    if x_start < 35:
        return False
    start_dist = distance_to_goal(x_start, y_start)
    end_dist = distance_to_goal(x_end, y_end)
    if start_dist == 0:
        return False
    reduction_pct = (start_dist - end_dist) / start_dist
    return reduction_pct >= 0.25


def progressive_wyscout(x_start, x_end) -> bool:
    """Wyscout progressive pass rule."""
    dist_start = FIELD_X - x_start
    dist_end = FIELD_X - x_end
    closer_by = dist_start - dist_end

    start_own = x_start < HALF_LINE_X
    end_own = x_end < HALF_LINE_X
    start_opp = x_start >= HALF_LINE_X
    end_opp = x_end >= HALF_LINE_X

    if start_own and end_own:
        return closer_by >= 30.0
    if (start_own and end_opp) or (start_opp and end_own):
        return closer_by >= 15.0
    if start_opp and end_opp:
        return closer_by >= 10.0
    return False


# ==========================
# Build DataFrames
# ==========================
dfs_by_match = {}
for match_name, events in matches_data.items():
    dfm = pd.DataFrame(
        events,
        columns=["type", "x_start", "y_start", "x_end", "y_end", "video"],
    )
    dfm["match"] = match_name
    dfm["number"] = np.arange(1, len(dfm) + 1)
    dfm["is_won"] = dfm["type"].str.contains("WON", case=False)
    dfm["outcome"] = np.where(dfm["is_won"], "successful", "failed")

    # Original progressive (from code 1)
    dfm["progressive"] = dfm.apply(
        lambda row: row["is_won"] and is_progressive_pass_original(
            row["x_start"], row["y_start"], row["x_end"], row["y_end"]
        ),
        axis=1,
    )

    # Switch pass (from code 1)
    dfm["switch"] = dfm.apply(
        lambda row: is_switch_pass(row["x_start"], row["y_start"], row["y_end"]),
        axis=1,
    )

    # Forward pass (from code 2)
    dfm["is_forward"] = dfm["x_end"] > dfm["x_start"]

    # Progressive Wyscout (from code 2)
    dfm["is_progressive_wyscout"] = dfm.apply(
        lambda row: progressive_wyscout(row["x_start"], row["x_end"]),
        axis=1,
    )

    # xT values (from code 2)
    dfm["xt_start"] = dfm.apply(lambda r: xt_value(r["x_start"], r["y_start"]), axis=1)
    dfm["xt_end"] = dfm.apply(lambda r: xt_value(r["x_end"], r["y_end"]), axis=1)
    dfm["delta_xt"] = np.where(
        dfm["outcome"].eq("successful"),
        dfm["xt_end"] - dfm["xt_start"],
        0.0,
    )

    dfs_by_match[match_name] = dfm

df_all = pd.concat(dfs_by_match.values(), ignore_index=True)
full_data = {"All Matches": df_all}
full_data.update(dfs_by_match)


# ==========================
# Stats (merged from both codes)
# ==========================
def compute_stats(df: pd.DataFrame) -> dict:
    total_passes = len(df)
    successful = int(df["is_won"].sum())
    unsuccessful = total_passes - successful
    accuracy = (successful / total_passes * 100.0) if total_passes else 0.0

    # --- From code 1 ---

    # Progressive (original rule, only successful)
    progressive_total = int(df["progressive"].sum())
    progressive_unsuccessful = int(
        (~df["is_won"] & df.apply(
            lambda row: is_progressive_pass_original(
                row["x_start"], row["y_start"], row["x_end"], row["y_end"]
            ), axis=1
        )).sum()
    )
    progressive_attempted = progressive_total + progressive_unsuccessful
    progressive_accuracy = (
        progressive_total / progressive_attempted * 100.0
        if progressive_attempted
        else 0.0
    )

    key_passes = int(df["video"].apply(has_video_value).sum())

    # To the Final Third
    to_final_third = (df["x_start"] < FINAL_THIRD_LINE_X) & (df["x_end"] >= FINAL_THIRD_LINE_X)
    to_final_third_total = int(to_final_third.sum())
    to_final_third_success = int((to_final_third & df["is_won"]).sum())
    to_final_third_accuracy = (
        (to_final_third_success / to_final_third_total * 100.0)
        if to_final_third_total else 0.0
    )

    # Into the Box
    to_box = (
        (df["x_end"] >= BOX_X_MIN)
        & (df["y_end"] >= BOX_Y_MIN)
        & (df["y_end"] <= BOX_Y_MAX)
    )
    box_total = int(to_box.sum())
    box_success = int((to_box & df["is_won"]).sum())
    box_accuracy = (box_success / box_total * 100.0) if box_total else 0.0

    # Switch Pass
    switch_total = int(df["switch"].sum())
    switch_success = int((df["switch"] & df["is_won"]).sum())
    switch_unsuccess = switch_total - switch_success
    switch_accuracy = (switch_success / switch_total * 100.0) if switch_total else 0.0
    switch_pct_of_total = (switch_total / total_passes * 100.0) if total_passes else 0.0

    # --- From code 2 ---

    # Forward passes
    forward_total = int(df["is_forward"].sum())
    pct_forward = (forward_total / total_passes * 100.0) if total_passes else 0.0

    # Progressive Wyscout
    prog_wyscout_total = int(df["is_progressive_wyscout"].sum())
    pct_progressive_wyscout = (
        prog_wyscout_total / total_passes * 100.0
    ) if total_passes else 0.0

    prog_wyscout_success = int(
        (df["is_progressive_wyscout"] & df["is_won"]).sum()
    )
    prog_wyscout_accuracy = (
        prog_wyscout_success / prog_wyscout_total * 100.0
    ) if prog_wyscout_total else 0.0

    # xT — only for successful progressive (Wyscout) passes
    prog_success_mask = df["is_progressive_wyscout"] & (df["outcome"] == "successful")
    xt_prog_sum = float(df.loc[prog_success_mask, "delta_xt"].sum())
    xt_prog_mean = (
        float(df.loc[prog_success_mask, "delta_xt"].mean())
        if prog_success_mask.any() else 0.0
    )

    # xT total (all successful passes)
    xt_total_sum = float(df.loc[df["outcome"] == "successful", "delta_xt"].sum())
    xt_total_mean = (
        float(df.loc[df["outcome"] == "successful", "delta_xt"].mean())
        if (df["outcome"] == "successful").any() else 0.0
    )

    return {
        # General
        "total_passes": total_passes,
        "successful_passes": successful,
        "unsuccessful_passes": unsuccessful,
        "accuracy_pct": round(accuracy, 2),
        "key_passes": key_passes,
        # Progressive (original)
        "progressive_attempted": progressive_attempted,
        "progressive_successful": progressive_total,
        "progressive_accuracy_pct": round(progressive_accuracy, 2),
        # Final Third
        "to_final_third_total": to_final_third_total,
        "to_final_third_success": to_final_third_success,
        "to_final_third_accuracy_pct": round(to_final_third_accuracy, 2),
        # Box
        "box_total": box_total,
        "box_success": box_success,
        "box_accuracy_pct": round(box_accuracy, 2),
        # Switch
        "switch_total": switch_total,
        "switch_success": switch_success,
        "switch_unsuccess": switch_unsuccess,
        "switch_accuracy_pct": round(switch_accuracy, 2),
        "switch_pct_of_total": round(switch_pct_of_total, 2),
        # Forward
        "forward_total": forward_total,
        "pct_forward": round(pct_forward, 2),
        # Progressive Wyscout
        "prog_wyscout_total": prog_wyscout_total,
        "prog_wyscout_success": prog_wyscout_success,
        "pct_progressive_wyscout": round(pct_progressive_wyscout, 2),
        "prog_wyscout_accuracy_pct": round(prog_wyscout_accuracy, 2),
        # xT
        "xt_prog_sum": round(xt_prog_sum, 4),
        "xt_prog_mean": round(xt_prog_mean, 4),
        "xt_total_sum": round(xt_total_sum, 4),
        "xt_total_mean": round(xt_total_mean, 4),
    }


# ==========================
# Draw pass map
# ==========================
FIG_W, FIG_H = 7.9, 5.3
FIG_DPI = 110


def draw_pass_map(df: pd.DataFrame, title: str):
    pitch = Pitch(
        pitch_type="statsbomb",
        pitch_color="#f5f5f5",
        line_color="#4a4a4a",
    )
    fig, ax = pitch.draw(figsize=(FIG_W, FIG_H))
    fig.set_dpi(FIG_DPI)

    ax.axvline(x=FINAL_THIRD_LINE_X, color="#FFD54F", linewidth=1.2, alpha=0.25)
    ax.axvline(x=HALF_LINE_X, color="#aaaaaa", linewidth=0.8, alpha=0.15,
               linestyle="--")

    START_DOT_SIZE = 45

    for _, row in df.iterrows():
        is_lost = not row["is_won"]
        is_prog = bool(row["progressive"])
        is_sw = bool(row["switch"])
        is_prog_w = bool(row["is_progressive_wyscout"])
        has_vid = has_video_value(row["video"])

        # Colour hierarchy: fail > switch > progressive (original) >
        #                    progressive Wyscout > success
        if is_lost:
            if is_sw:
                color = COLOR_SWITCH
                alpha = 0.60
            else:
                color = COLOR_FAIL
                alpha = 0.55
        elif is_sw:
            color = COLOR_SWITCH
            alpha = 0.85
        elif is_prog:
            color = COLOR_PROGRESSIVE
            alpha = 0.82
        elif is_prog_w:
            color = COLOR_PROGRESSIVE
            alpha = 0.65
        else:
            color = COLOR_SUCCESS
            alpha = 0.25

        pitch.arrows(
            row["x_start"], row["y_start"],
            row["x_end"], row["y_end"],
            color=color, width=1.55, headwidth=2.25, headlength=2.25,
            ax=ax, zorder=3, alpha=alpha,
        )

        if has_vid:
            pitch.scatter(
                row["x_start"], row["y_start"],
                s=95, marker="o", facecolors="none",
                edgecolors="#FFD54F", linewidths=2.0, ax=ax, zorder=4,
            )

        pitch.scatter(
            row["x_start"], row["y_start"],
            s=START_DOT_SIZE, marker="o", color=color,
            edgecolors="white", linewidths=0.8, ax=ax, zorder=5, alpha=alpha,
        )

    ax.set_title(title, fontsize=12)

    legend_elements = [
        Line2D([0], [0], color=COLOR_SUCCESS, lw=2.5, alpha=0.25,
               label="Successful Pass"),
        Line2D([0], [0], color=COLOR_FAIL, lw=2.5,
               label="Unsuccessful Pass"),
        Line2D([0], [0], color=COLOR_PROGRESSIVE, lw=2.5,
               label="Progressive Pass"),
        Line2D([0], [0], color=COLOR_SWITCH, lw=2.5,
               label="Switch Pass"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markeredgecolor="white", markersize=6, label="Start point (click)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markeredgecolor="#FFD54F", markeredgewidth=2, markersize=7,
               label="Has video"),
    ]

    legend = ax.legend(
        handles=legend_elements, loc="upper left", bbox_to_anchor=(0.01, 0.99),
        frameon=True, facecolor="white", edgecolor="#cccccc", shadow=False,
        fontsize="x-small", labelspacing=0.5, borderpad=0.5,
    )
    legend.get_frame().set_alpha(1.0)

    arrow = FancyArrowPatch(
        (0.45, 0.05), (0.55, 0.05), transform=fig.transFigure,
        arrowstyle="-|>", mutation_scale=15, linewidth=2, color="#333333",
    )
    fig.patches.append(arrow)
    fig.text(0.5, 0.02, "Attack Direction",
             ha="center", va="center", fontsize=9, color="#333333")

    fig.tight_layout()
    fig.canvas.draw()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=FIG_DPI)
    buf.seek(0)
    img_obj = Image.open(buf)
    return img_obj, ax, fig


# ==========================
# Sidebar
# ==========================
st.sidebar.header("Match Selection")
selected_match = st.sidebar.radio(
    "Choose the match", list(full_data.keys()), index=0
)

st.sidebar.header("Pass Filter")
pass_filter = st.sidebar.radio(
    "Filter passes",
    [
        "All Passes",
        "Successful Only",
        "Unsuccessful Only",
        "Progressive Only (Original)",
        "Progressive Only (Wyscout)",
        "Forward Only",
        "To Final Third",
        "Switch Only",
    ],
    index=0,
)

df = full_data[selected_match].copy()

if pass_filter == "Successful Only":
    df = df[df["is_won"]].reset_index(drop=True)
elif pass_filter == "Unsuccessful Only":
    df = df[~df["is_won"]].reset_index(drop=True)
elif pass_filter == "Progressive Only (Original)":
    df = df[df["progressive"]].reset_index(drop=True)
elif pass_filter == "Progressive Only (Wyscout)":
    df = df[df["is_progressive_wyscout"]].reset_index(drop=True)
elif pass_filter == "Forward Only":
    df = df[df["is_forward"]].reset_index(drop=True)
elif pass_filter == "To Final Third":
    mask = (df["x_start"] < FINAL_THIRD_LINE_X) & (df["x_end"] >= FINAL_THIRD_LINE_X)
    df = df[mask].reset_index(drop=True)
elif pass_filter == "Switch Only":
    df = df[df["switch"]].reset_index(drop=True)

stats = compute_stats(df)

# ==========================
# Caption
# ==========================
st.caption("Click the start dot to select the pass event.")

# ==========================
# Layout
# ==========================
col_stats, col_right = st.columns([1, 2], gap="large")

with col_stats:
    st.subheader("General Statistics")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Passes", stats["total_passes"])
    c2.metric("Successful", stats["successful_passes"])
    c3.metric("Accuracy", f'{stats["accuracy_pct"]:.1f}%')

    st.divider()

    # --- Forward Passes (from code 2) ---
    st.subheader("Forward Passes")
    f1, f2 = st.columns(2)
    f1.metric("Total Forward", stats["forward_total"])
    f2.metric("% Forward", f'{stats["pct_forward"]:.1f}%')

    st.divider()

    # --- Progressive Wyscout (from code 2) ---
    st.subheader("Progressive Passes (Wyscout)")
    pw1, pw2, pw3 = st.columns(3)
    pw1.metric("Total", stats["prog_wyscout_total"])
    pw2.metric("Successful", stats["prog_wyscout_success"])
    pw3.metric("Accuracy", f'{stats["prog_wyscout_accuracy_pct"]:.1f}%')

    pw4, pw5 = st.columns(2)
    pw4.metric("% of Total", f'{stats["pct_progressive_wyscout"]:.1f}%')
    pw5.metric("xT Σ (prog.)", f'{stats["xt_prog_sum"]:.4f}')

    st.divider()

    # --- xT Summary (from code 2) ---
    st.subheader("Expected Threat (xT)")
    xt1, xt2 = st.columns(2)
    xt1.metric("xT Total (successful)", f'{stats["xt_total_sum"]:.4f}')
    xt2.metric("xT Mean (successful)", f'{stats["xt_total_mean"]:.4f}')

    xt3, xt4 = st.columns(2)
    xt3.metric("xT Σ Progressive", f'{stats["xt_prog_sum"]:.4f}')
    xt4.metric("xT Mean Progressive", f'{stats["xt_prog_mean"]:.4f}')

    st.divider()

    # --- Progressive Original (from code 1) ---
    st.subheader("Progressive Passes (Original)")
    p1, p2, p3 = st.columns(3)
    p1.metric("Total", stats["progressive_attempted"])
    p2.metric("Successful", stats["progressive_successful"])
    p3.metric("Accuracy", f'{stats["progressive_accuracy_pct"]:.1f}%')

    st.divider()

    # --- To the Final Third (from code 1) ---
    st.subheader("To the Final Third")
    c7, c8, c9 = st.columns(3)
    c7.metric("Total", stats["to_final_third_total"])
    c8.metric("Successful", stats["to_final_third_success"])
    c9.metric("Accuracy", f'{stats["to_final_third_accuracy_pct"]:.1f}%')

    st.divider()

    # --- Into the Box (from code 1) ---
    st.subheader("Passes Into the Box")
    d1, d2, d3 = st.columns(3)
    d1.metric("Total", stats["box_total"])
    d2.metric("Successful", stats["box_success"])
    d3.metric("Accuracy", f'{stats["box_accuracy_pct"]:.1f}%')

    st.divider()

    # --- Switch Passes (from code 1) ---
    st.subheader("Switch Passes")
    s1, s2 = st.columns(2)
    s1.metric("Total", stats["switch_total"])
    s2.metric("Successful", stats["switch_success"])
    sw1, sw2 = st.columns(2)
    sw1.metric("Accuracy", f'{stats["switch_accuracy_pct"]:.1f}%')
    sw2.metric("% of Total Passes", f'{stats["switch_pct_of_total"]:.1f}%')

with col_right:
    st.subheader("Pass Map (click the start dot)")

    img_obj, ax, fig = draw_pass_map(df, title=f"Pass Map — {selected_match}")

    DISPLAY_WIDTH = 780
    click = streamlit_image_coordinates(img_obj, width=DISPLAY_WIDTH)

    selected_pass = None

    if click is not None:
        real_w, real_h = img_obj.size
        disp_w = click["width"]
        disp_h = click["height"]

        pixel_x = click["x"] * (real_w / disp_w)
        pixel_y = click["y"] * (real_h / disp_h)
        mpl_pixel_y = real_h - pixel_y

        field_x, field_y = ax.transData.inverted().transform((pixel_x, mpl_pixel_y))

        df_sel = df.copy()
        df_sel["dist"] = np.sqrt(
            (df_sel["x_start"] - field_x) ** 2
            + (df_sel["y_start"] - field_y) ** 2
        )

        RADIUS = 5.0
        candidates = df_sel[df_sel["dist"] < RADIUS].copy()

        if not candidates.empty:
            candidates = candidates.sort_values(by="dist", ascending=True)
            selected_pass = candidates.iloc[0]

    plt.close(fig)

    st.divider()
    st.subheader("Selected Event")

    if selected_pass is None:
        st.info("Click the start dot to inspect the pass details.")
    else:
        st.success(
            f"Selected pass: #{int(selected_pass['number'])} ({selected_pass['type']})"
        )

        # Position info
        det1, det2 = st.columns(2)
        with det1:
            st.write(
                f"**Start:** ({selected_pass['x_start']:.2f}, "
                f"{selected_pass['y_start']:.2f})"
            )
        with det2:
            st.write(
                f"**End:** ({selected_pass['x_end']:.2f}, "
                f"{selected_pass['y_end']:.2f})"
            )

        # Flags
        tag1, tag2, tag3, tag4 = st.columns(4)
        tag1.write(
            f"**Progressive (Orig.):** "
            f"{'✅' if selected_pass['progressive'] else '❌'}"
        )
        tag2.write(
            f"**Progressive (Wyscout):** "
            f"{'✅' if selected_pass['is_progressive_wyscout'] else '❌'}"
        )
        tag3.write(
            f"**Forward:** "
            f"{'✅' if selected_pass['is_forward'] else '❌'}"
        )
        tag4.write(
            f"**Switch:** "
            f"{'✅' if selected_pass['switch'] else '❌'}"
        )

        # xT info
        xt_col1, xt_col2, xt_col3 = st.columns(3)
        xt_col1.metric("xT Start", f"{selected_pass['xt_start']:.4f}")
        xt_col2.metric("xT End", f"{selected_pass['xt_end']:.4f}")
        xt_col3.metric(
            "ΔxT",
            f"{selected_pass['delta_xt']:.4f}",
            delta=f"{selected_pass['delta_xt']:.4f}"
            if selected_pass["delta_xt"] != 0 else None,
        )

        # Video
        if has_video_value(selected_pass["video"]):
            try:
                st.video(selected_pass["video"])
            except Exception:
                st.error(f"Video file not found: {selected_pass['video']}")
        else:
            st.warning("No video is attached to this event.")

    # ==========================
    # Data Table (expandable)
    # ==========================
    with st.expander("📊 Full Pass Data Table"):
        display_cols = [
            "number", "type", "outcome",
            "x_start", "y_start", "x_end", "y_end",
            "is_forward", "progressive", "is_progressive_wyscout",
            "switch", "xt_start", "xt_end", "delta_xt",
        ]
        st.dataframe(
            df[display_cols].style.format({
                "x_start": "{:.2f}", "y_start": "{:.2f}",
                "x_end": "{:.2f}", "y_end": "{:.2f}",
                "xt_start": "{:.4f}", "xt_end": "{:.4f}",
                "delta_xt": "{:.4f}",
            }),
            use_container_width=True,
            height=400,
        )
