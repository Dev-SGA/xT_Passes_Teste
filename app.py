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
from matplotlib.patches import FancyArrowPatch, Rectangle
from streamlit_image_coordinates import streamlit_image_coordinates
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, ListedColormap, LinearSegmentedColormap

# ==========================
# Page Configuration
# ==========================
st.set_page_config(layout="wide", page_title="Pass Map Dashboard (Interactive)")

# ==========================
# CSS
# ==========================
st.markdown(
    """
    <style>
    /* Container spacing */
    .small-metric { padding: 6px 8px; }

    /* Labels (small) */
    .small-metric .label {
      font-size: 12px;
      color: #ffffff;
      margin-bottom: 3px;
      opacity: 0.95;
    }

    /* Main value */
    .small-metric .value {
      font-size: 18px;
      font-weight: 600;
      color: #ffffff;
    }

    /* Delta / secondary text */
    .small-metric .delta {
      font-size: 11px;
      color: #e6e6e6;
      margin-top: 4px;
    }

    /* Section titles inside expanders */
    .stats-section-title {
      font-size: 14px;
      font-weight: 600;
      margin-bottom: 6px;
      color: #ffffff;
    }

    /* Expander headers */
    .streamlit-expanderHeader {
      color: #ffffff !important;
    }

    .streamlit-expander {
      background: rgba(255,255,255,0.02);
    }

    /* ===== Filter sidebar background ===== */
    .filter-panel {
      background: linear-gradient(168deg, rgba(30, 39, 56, 0.92) 0%, rgba(22, 28, 40, 0.97) 100%);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 14px;
      padding: 24px 18px 20px 18px;
      box-shadow: 0 4px 24px rgba(0,0,0,0.25), 0 1px 4px rgba(0,0,0,0.12);
      backdrop-filter: blur(6px);
    }

    .filter-panel h3 {
      font-size: 15px;
      color: #c8d6e5;
      letter-spacing: 0.5px;
      margin-bottom: 8px;
    }

    .filter-panel .filter-divider {
      border: none;
      border-top: 1px solid rgba(255,255,255,0.07);
      margin: 14px 0;
    }

    /* Make Streamlit subheaders slightly lighter on dark bg when we still use them */
    .stSubheader { color: #ffffff !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

def small_metric(label: str, value: str, delta: str | None = None):
    html = f"""
    <div class="small-metric">
      <div class="label">{label}</div>
      <div class="value">{value}</div>
    """
    if delta is not None:
        html += f'<div class="delta">{delta}</div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# ==========================
# Configuration
# ==========================
st.title("Pass Map Dashboard")

FIELD_X, FIELD_Y = 120.0, 80.0
HALF_LINE_X = FIELD_X / 2
FINAL_THIRD_LINE_X = 80

BOX_X_MIN = 102
BOX_Y_MIN = 18
BOX_Y_MAX = 62

GOAL_X = 120
GOAL_Y = 40

LANE_LEFT_MIN = 53.33
LANE_RIGHT_MAX = 26.67

NX, NY = 16, 12

LATERAL_MIN_DIST = 12.0

# Colours
COLOR_SUCCESS = "#ffffff"   # successful passes now white
COLOR_FAIL = "#D45B5B"
COLOR_PROGRESSIVE = "#2F80ED"
COLOR_SWITCH = "#DAA520"
COLOR_FORWARD = "#27AE60"
COLOR_BACKWARD = "#E67E22"
COLOR_LATERAL = "#8E44AD"

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


def progressive_wyscout(x_start, x_end) -> bool:
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


def classify_pass_direction(x_start, y_start, x_end, y_end) -> str:
    dx = x_end - x_start
    dy = y_end - y_start
    dist = np.sqrt(dx ** 2 + dy ** 2)

    angle_deg = np.degrees(np.arctan2(abs(dy), dx))

    if angle_deg <= 45.0:
        return "forward"
    elif angle_deg >= 135.0:
        return "backward"
    else:
        if dist > LATERAL_MIN_DIST:
            return "lateral"
        else:
            if dx >= 0:
                return "forward"
            else:
                return "backward"


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

    dfm["switch"] = dfm.apply(
        lambda row: is_switch_pass(row["x_start"], row["y_start"], row["y_end"]),
        axis=1,
    )

    dfm["direction"] = dfm.apply(
        lambda row: classify_pass_direction(
            row["x_start"], row["y_start"], row["x_end"], row["y_end"]
        ),
        axis=1,
    )
    dfm["is_forward"] = dfm["direction"] == "forward"
    dfm["is_backward"] = dfm["direction"] == "backward"
    dfm["is_lateral"] = dfm["direction"] == "lateral"

    dfm["is_progressive_wyscout"] = dfm.apply(
        lambda row: progressive_wyscout(row["x_start"], row["x_end"]),
        axis=1,
    )

    dfm["xt_start"] = dfm.apply(lambda r: xt_value(r["x_start"], r["y_start"]), axis=1)
    dfm["xt_end"] = dfm.apply(lambda r: xt_value(r["x_end"], r["y_end"]), axis=1)
    dfm["delta_xt"] = np.where(
        dfm["outcome"].eq("successful"),
        dfm["xt_end"] - dfm["xt_start"],
        0.0,
    )

    dfm["pass_distance"] = np.sqrt(
        (dfm["x_end"] - dfm["x_start"]) ** 2
        + (dfm["y_end"] - dfm["y_start"]) ** 2
    )

    dfs_by_match[match_name] = dfm

df_all = pd.concat(dfs_by_match.values(), ignore_index=True)
full_data = {"All Matches": df_all}
full_data.update(dfs_by_match)


# ==========================
# Stats
# ==========================
def compute_stats(df: pd.DataFrame) -> dict:
    total_passes = len(df)
    successful = int(df["is_won"].sum())
    unsuccessful = total_passes - successful
    accuracy = (successful / total_passes * 100.0) if total_passes else 0.0

    key_passes = int(df["video"].apply(has_video_value).sum())

    forward_total = int(df["is_forward"].sum())
    forward_success = int((df["is_forward"] & df["is_won"]).sum())
    pct_forward = (forward_total / total_passes * 100.0) if total_passes else 0.0

    backward_total = int(df["is_backward"].sum())
    backward_success = int((df["is_backward"] & df["is_won"]).sum())
    pct_backward = (backward_total / total_passes * 100.0) if total_passes else 0.0

    lateral_total = int(df["is_lateral"].sum())
    lateral_success = int((df["is_lateral"] & df["is_won"]).sum())
    pct_lateral = (lateral_total / total_passes * 100.0) if total_passes else 0.0

    switch_total = int(df["switch"].sum())
    switch_success = int((df["switch"] & df["is_won"]).sum())
    switch_unsuccess = switch_total - switch_success
    switch_accuracy = (switch_success / switch_total * 100.0) if switch_total else 0.0
    switch_pct_of_total = (switch_total / total_passes * 100.0) if total_passes else 0.0

    prog_wyscout_total = int(df["is_progressive_wyscout"].sum())
    prog_wyscout_success = int(
        (df["is_progressive_wyscout"] & df["is_won"]).sum()
    )
    pct_progressive_wyscout = (
        prog_wyscout_total / total_passes * 100.0
    ) if total_passes else 0.0
    prog_wyscout_accuracy = (
        prog_wyscout_success / prog_wyscout_total * 100.0
    ) if prog_wyscout_total else 0.0

    prog_success_mask = df["is_progressive_wyscout"] & (df["outcome"] == "successful")
    xt_prog_sum = float(df.loc[prog_success_mask, "delta_xt"].sum())
    xt_prog_mean = (
        float(df.loc[prog_success_mask, "delta_xt"].mean())
        if prog_success_mask.any() else 0.0
    )

    xt_total_sum = float(df.loc[df["outcome"] == "successful", "delta_xt"].sum())
    xt_total_mean = (
        float(df.loc[df["outcome"] == "successful", "delta_xt"].mean())
        if (df["outcome"] == "successful").any() else 0.0
    )

    positive_xt_mask = (df["outcome"] == "successful") & (df["delta_xt"] > 0)
    positive_xt_total = int(positive_xt_mask.sum())
    positive_xt_sum = float(df.loc[positive_xt_mask, "delta_xt"].sum())
    positive_xt_mean = (
        float(df.loc[positive_xt_mask, "delta_xt"].mean())
        if positive_xt_mask.any() else 0.0
    )

    return {
        "total_passes": total_passes,
        "successful_passes": successful,
        "unsuccessful_passes": unsuccessful,
        "accuracy_pct": round(accuracy, 2),
        "key_passes": key_passes,
        "forward_total": forward_total,
        "forward_success": forward_success,
        "pct_forward": pct_forward,
        "backward_total": backward_total,
        "backward_success": backward_success,
        "pct_backward": pct_backward,
        "lateral_total": lateral_total,
        "lateral_success": lateral_success,
        "pct_lateral": pct_lateral,
        "switch_total": switch_total,
        "switch_success": switch_success,
        "switch_unsuccess": switch_unsuccess,
        "switch_accuracy_pct": round(switch_accuracy, 2),
        "switch_pct_of_total": round(switch_pct_of_total, 2),
        "prog_wyscout_total": prog_wyscout_total,
        "prog_wyscout_success": prog_wyscout_success,
        "pct_progressive_wyscout": round(pct_progressive_wyscout, 2),
        "prog_wyscout_accuracy_pct": round(prog_wyscout_accuracy, 2),
        "xt_prog_sum": round(xt_prog_sum, 4),
        "xt_prog_mean": round(xt_prog_mean, 4),
        "xt_total_sum": round(xt_total_sum, 4),
        "xt_total_mean": round(xt_total_mean, 4),
        "positive_xt_total": positive_xt_total,
        "positive_xt_sum": round(positive_xt_sum, 4),
        "positive_xt_mean": round(positive_xt_mean, 4),
    }


# ==========================
# Draw pass map
# ==========================
FIG_W, FIG_H = 7.9, 5.3
FIG_DPI = 110


def draw_pass_map(df: pd.DataFrame, title: str):
    # pitch with white lines for better contrast
    pitch = Pitch(
        pitch_type="statsbomb",
        pitch_color="#1a1a2e",
        line_color="#ffffff",
        line_alpha=0.95,
    )
    fig, ax = pitch.draw(figsize=(FIG_W, FIG_H))
    fig.set_facecolor("#1a1a2e")
    fig.set_dpi(FIG_DPI)

    ax.axvline(x=FINAL_THIRD_LINE_X, color="#FFD54F", linewidth=1.0, alpha=0.20)
    ax.axvline(x=HALF_LINE_X, color="#ffffff", linewidth=0.6, alpha=0.12, linestyle="--")

    START_DOT_SIZE = 45

    for _, row in df.iterrows():
        is_lost = not row["is_won"]
        is_sw = bool(row["switch"])
        is_prog_w = bool(row["is_progressive_wyscout"])
        has_vid = has_video_value(row["video"])

        # New contrast rules:
        # - Successful: white, very transparent (to reduce clutter)
        # - Progressive: more opaque to stand out
        # - Unsuccessful: more opaque to stand out
        if is_lost:
            color = COLOR_FAIL
            alpha = 0.92  # more visible
        elif is_sw:
            color = COLOR_SWITCH
            alpha = 0.92
        elif is_prog_w:
            color = COLOR_PROGRESSIVE
            alpha = 0.88  # more opaque
        else:
            # successful ordinary passes -> white, very transparent
            color = COLOR_SUCCESS
            alpha = 0.03  # very transparent

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
                edgecolors="#FFD54F", linewidths=2.0, ax=ax, zorder=5,
            )

        # start dot: keep visible edge so white dots still readable on dark bg
        pitch.scatter(
            row["x_start"], row["y_start"],
            s=START_DOT_SIZE, marker="o", color=color,
            edgecolors="white", linewidths=0.8, ax=ax, zorder=6, alpha=alpha,
        )

    # matplotlib title (inside figure) in white
    ax.set_title(title, fontsize=12, color="#ffffff", pad=8)

    # Simplified legend: Successful, Unsuccessful, Progressive (labels in white)
    legend_elements = [
        Line2D([0], [0], color="#ffffff", lw=2.5, label="Successful", alpha=0.5),
        Line2D([0], [0], color=COLOR_FAIL, lw=2.5, label="Unsuccessful", alpha=0.9),
        Line2D([0], [0], color=COLOR_PROGRESSIVE, lw=2.5, label="Progressive", alpha=0.9),
    ]

    legend = ax.legend(
        handles=legend_elements, loc="upper left", bbox_to_anchor=(0.01, 0.99),
        frameon=True, facecolor="#1a1a2e", edgecolor="#444466", shadow=False,
        fontsize="x-small", labelspacing=0.5, borderpad=0.5,
    )
    # ensure legend text color is white
    for txt in legend.get_texts():
        txt.set_color("white")

    legend.get_frame().set_alpha(0.92)

    arrow = FancyArrowPatch(
        (0.45, 0.05), (0.55, 0.05), transform=fig.transFigure,
        arrowstyle="-|>", mutation_scale=15, linewidth=2, color="#cccccc",
    )
    fig.patches.append(arrow)
    fig.text(0.5, 0.02, "Attack Direction",
             ha="center", va="center", fontsize=9, color="#cccccc")

    fig.tight_layout()
    fig.canvas.draw()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=FIG_DPI, facecolor=fig.get_facecolor())
    buf.seek(0)
    img_obj = Image.open(buf)
    return img_obj, ax, fig


# ==========================
# Draw zone heatmap (same size & style as pass map)
# ==========================
def draw_corridor_heatmap(df: pd.DataFrame, title: str = "Zone Heatmap — Completed Passes"):
    df_success = df[df["is_won"]].copy()

    x_bins = np.linspace(0.0, FIELD_X, 7)

    left_y0, left_y1 = LANE_LEFT_MIN, FIELD_Y
    right_y0, right_y1 = 0.0, LANE_RIGHT_MAX
    center_y0, center_y1 = LANE_RIGHT_MAX, LANE_LEFT_MIN

    corridors = {
        "left": (left_y0, left_y1),
        "center": (center_y0, center_y1),
        "right": (right_y0, right_y1),
    }

    counts = {}
    for cname, (y0, y1) in corridors.items():
        arr = np.zeros(6, dtype=int)
        for i in range(6):
            x0, x1 = x_bins[i], x_bins[i + 1]
            mask = (
                (df_success["x_end"] >= x0)
                & (df_success["x_end"] < x1)
                & (df_success["y_end"] >= y0)
                & (df_success["y_end"] < y1)
            )
            arr[i] = int(mask.sum())
        counts[cname] = arr

    all_vals = np.concatenate([counts[c] for c in counts])
    vmax = max(1, int(all_vals.max()))

    # Same size as pass map
    pitch = Pitch(
        pitch_type="statsbomb",
        pitch_color="#1a1a2e",
        line_color="#ffffff",
        line_alpha=0.95,
    )
    fig, ax = pitch.draw(figsize=(FIG_W, FIG_H))
    fig.set_facecolor("#1a1a2e")
    fig.set_dpi(FIG_DPI)

    # Create a white -> red colormap
    cmap = LinearSegmentedColormap.from_list("white_red", ["#ffffff", "#ffecec", "#ffbfbf", "#ff8080", "#ff3b3b", "#ff0000"])
    norm = Normalize(vmin=0, vmax=vmax)

    # threshold for deciding text color: if background is light (low value), use black text
    text_light_threshold = max(1, vmax * 0.35)

    for cname, (y0, y1) in corridors.items():
        arr = counts[cname]
        for i in range(6):
            x0, x1 = x_bins[i], x_bins[i + 1]
            value = arr[i]
            color = cmap(norm(value))
            rect = Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                facecolor=color, edgecolor=(1.0, 1.0, 1.0, 0.12),
                linewidth=0.6, alpha=0.95, zorder=2,
            )
            ax.add_patch(rect)

            cx = (x0 + x1) / 2.0
            cy = (y0 + y1) / 2.0

            # If the cell background is light (value small), use black text for contrast. Otherwise white.
            if value <= text_light_threshold:
                text_color = "#000000"
            else:
                text_color = "#ffffff"

            fontw = "700" if value >= vmax * 0.5 else "600"

            ax.text(
                cx, cy, str(value),
                ha="center", va="center",
                color=text_color,
                fontsize=11, fontweight=fontw, zorder=4,
            )

    # matplotlib title (inside figure) in white
    ax.set_title(title, fontsize=12, color="#ffffff", pad=8)

    # Dashed corridor separators (kept for subtle structure)
    ax.axhline(y=LANE_LEFT_MIN, color="#ffffff", linewidth=0.5, alpha=0.15,
               linestyle="--", zorder=3)
    ax.axhline(y=LANE_RIGHT_MAX, color="#ffffff", linewidth=0.5, alpha=0.15,
               linestyle="--", zorder=3)

    # Remove colorbar and corridor description text: leave only the map

    arrow = FancyArrowPatch(
        (0.45, 0.05), (0.55, 0.05), transform=fig.transFigure,
        arrowstyle="-|>", mutation_scale=15, linewidth=2, color="#cccccc",
    )
    fig.patches.append(arrow)
    fig.text(0.5, 0.02, "Attack Direction",
             ha="center", va="center", fontsize=9, color="#cccccc")

    fig.tight_layout()
    fig.canvas.draw()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=FIG_DPI, facecolor=fig.get_facecolor())
    buf.seek(0)
    img_obj = Image.open(buf)
    return img_obj, ax, fig


# ==========================
# Layout
# ==========================
st.caption("Click the start dot to select the pass event.")

col_filters, col_field, col_stats = st.columns([0.9, 2, 1], gap="large")

# LEFT: Filters column with styled panel
with col_filters:
    st.markdown('<div class="filter-panel">', unsafe_allow_html=True)

    st.markdown("### 🏟️ Match Selection")
    selected_match = st.selectbox("Choose the match", list(full_data.keys()), index=0)

    st.markdown('<hr class="filter-divider">', unsafe_allow_html=True)

    st.markdown("### 🎯 Pass Filter")
    pass_filter = st.radio(
        "Filter passes",
        [
            "All Passes",
            "Successful Only",
            "Unsuccessful Only",
            "Progressive Only (All)",
            "Positive xT Only (Successful)",
        ],
        index=0,
    )

    st.markdown('</div>', unsafe_allow_html=True)

# Ensure session state keys
if "heat_selection" not in st.session_state:
    st.session_state["heat_selection"] = None
if "last_match" not in st.session_state:
    st.session_state["last_match"] = selected_match
if "last_filter" not in st.session_state:
    st.session_state["last_filter"] = pass_filter

# Clear heat selection automatically if match or filter changed
if st.session_state["last_match"] != selected_match:
    st.session_state["heat_selection"] = None
    st.session_state["last_match"] = selected_match
if st.session_state["last_filter"] != pass_filter:
    st.session_state["heat_selection"] = None
    st.session_state["last_filter"] = pass_filter

# MIDDLE: Field + heatmap + selection details
with col_field:
    df_base = full_data[selected_match].copy()

    if pass_filter == "All Passes":
        df_base = df_base.reset_index(drop=True)
    elif pass_filter == "Successful Only":
        df_base = df_base[df_base["is_won"]].reset_index(drop=True)
    elif pass_filter == "Unsuccessful Only":
        df_base = df_base[~df_base["is_won"]].reset_index(drop=True)
    elif pass_filter == "Progressive Only (All)":
        df_base = df_base[df_base["is_progressive_wyscout"]].reset_index(drop=True)
    elif pass_filter == "Positive xT Only (Successful)":
        df_base = df_base[(df_base["outcome"] == "successful") & (df_base["delta_xt"] > 0)].reset_index(drop=True)

    # Apply heatmap selection if exists (selection stored uses x_end/y_end ranges)
    df_to_draw = df_base
    if st.session_state["heat_selection"] is not None:
        sel = st.session_state["heat_selection"]
        df_to_draw = df_base[
            (df_base["x_end"] >= sel["x0"])
            & (df_base["x_end"] < sel["x1"])
            & (df_base["y_end"] >= sel["y0"])
            & (df_base["y_end"] < sel["y1"])
        ].reset_index(drop=True)

    stats = compute_stats(df_to_draw)

    # Display width used for both maps to ensure identical rendering width (alignment)
    DISPLAY_WIDTH = 780

    # ---- Pass Map (kept first as requested) ----
    st.markdown('<h4 style="color:#ffffff; margin:0 0 6px 0;">Pass Map (clique no start dot)</h4>', unsafe_allow_html=True)
    img_obj, ax, fig = draw_pass_map(df_to_draw, title=f"Pass Map — {selected_match}")

    # interactive clickable image (streamlit_image_coordinates displays the image)
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

        df_sel = df_to_draw.copy()
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

    # ---- Zone Heatmap (second) ----
    st.markdown('<h4 style="color:#ffffff; margin:6px 0 6px 0;">Zone Heatmap (clique em um quadrante para filtrar o Pass Map)</h4>', unsafe_allow_html=True)
    heat_img, hax, hfig = draw_corridor_heatmap(df_base)

    # Use streamlit_image_coordinates so user can click the heatmap -> we map the click to the corridor+column
    heat_click = streamlit_image_coordinates(heat_img, width=DISPLAY_WIDTH)

    if heat_click is not None:
        real_w, real_h = heat_img.size
        disp_w = heat_click["width"]
        disp_h = heat_click["height"]

        pixel_x = heat_click["x"] * (real_w / disp_w)
        pixel_y = heat_click["y"] * (real_h / disp_h)
        mpl_pixel_y = real_h - pixel_y

        # convert pixel -> data coords using heatmap axis transform
        field_x, field_y = hax.transData.inverted().transform((pixel_x, mpl_pixel_y))

        # x bins and corridor ranges must match draw_corridor_heatmap()
        x_bins = np.linspace(0.0, FIELD_X, 7)
        # clamp index into [0,5]
        ix = np.searchsorted(x_bins, field_x, side="right") - 1
        ix = max(0, min(5, ix))
        x0, x1 = x_bins[ix], x_bins[ix + 1]

        if field_y >= LANE_LEFT_MIN:
            cname = "left"
            y0, y1 = LANE_LEFT_MIN, FIELD_Y
        elif field_y < LANE_RIGHT_MAX:
            cname = "right"
            y0, y1 = 0.0, LANE_RIGHT_MAX
        else:
            cname = "center"
            y0, y1 = LANE_RIGHT_MAX, LANE_LEFT_MIN

        # store selection in session_state (so reruns keep it)
        st.session_state["heat_selection"] = {
            "ix": int(ix),
            "corridor": cname,
            "x0": float(x0),
            "x1": float(x1),
            "y0": float(y0),
            "y1": float(y1),
        }

    # close heatmap figure (we already have heat_img)
    plt.close(hfig)

    # Show info about current heatmap selection and clear button
    if st.session_state["heat_selection"] is not None:
        sel = st.session_state["heat_selection"]
        sel_mask = (
            (df_base["x_end"] >= sel["x0"])
            & (df_base["x_end"] < sel["x1"])
            & (df_base["y_end"] >= sel["y0"])
            & (df_base["y_end"] < sel["y1"])
        )
        sel_count = int(sel_mask.sum())
        st.markdown(
            f"<div style='color:#ffffff; margin-top:6px;'>"
            f"<strong>Filtro aplicado:</strong> corredor <code>{sel['corridor']}</code>, coluna X #{sel['ix']+1} — {sel_count} passes</div>",
            unsafe_allow_html=True,
        )
        if st.button("Limpar filtro do quadrante"):
            st.session_state["heat_selection"] = None
    else:
        st.markdown("<div style='color:#cfcfcf; margin-top:6px;'>Nenhum quadrante selecionado.</div>", unsafe_allow_html=True)

    # display static heatmap for alignment (we already used streamlit_image_coordinates which rendered it; re-display for consistent layout)
    st.image(heat_img, width=DISPLAY_WIDTH)

    # ---- Selected Event (moved below both maps) ----
    st.divider()
    st.subheader("Selected Event")

    if selected_pass is None:
        st.info("Click the start dot to inspect the pass details.")
    else:
        st.success(
            f"Selected pass: #{int(selected_pass['number'])} ({selected_pass['type']})"
        )

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

        dir_emoji = {"forward": "⬆️", "backward": "⬇️", "lateral": "↔️"}
        direction_label = selected_pass["direction"].capitalize()
        emoji = dir_emoji.get(selected_pass["direction"], "")

        tag1, tag2, tag3 = st.columns(3)
        tag1.write(f"**Direction:** {emoji} {direction_label}")
        tag2.write(
            f"**Progressive (Wyscout):** "
            f"{'✅' if selected_pass['is_progressive_wyscout'] else '❌'}"
        )
        tag3.write(
            f"**Switch:** "
            f"{'✅' if selected_pass['switch'] else '❌'}"
        )

        xt_col1, xt_col2, xt_col3, xt_col4 = st.columns(4)
        xt_col1.metric("Distance", f"{selected_pass['pass_distance']:.1f}m")
        xt_col2.metric("xT Start", f"{selected_pass['xt_start']:.4f}")
        xt_col3.metric("xT End", f"{selected_pass['xt_end']:.4f}")
        xt_col4.metric(
            "ΔxT",
            f"{selected_pass['delta_xt']:.4f}",
            delta=f"{selected_pass['delta_xt']:.4f}"
            if selected_pass["delta_xt"] != 0 else None,
        )

        if has_video_value(selected_pass["video"]):
            try:
                st.video(selected_pass["video"])
            except Exception:
                st.error(f"Video file not found: {selected_pass['video']}")
        else:
            st.warning("No video is attached to this event.")

    # Data Table
    with st.expander("📊 Full Pass Data Table"):
        display_cols = [
            "number", "type", "outcome", "direction",
            "x_start", "y_start", "x_end", "y_end",
            "pass_distance",
            "is_forward", "is_backward", "is_lateral",
            "is_progressive_wyscout",
            "switch", "xt_start", "xt_end", "delta_xt",
        ]
        st.dataframe(
            df_to_draw[display_cols].style.format({
                "x_start": "{:.2f}", "y_start": "{:.2f}",
                "x_end": "{:.2f}", "y_end": "{:.2f}",
                "pass_distance": "{:.1f}",
                "xt_start": "{:.4f}", "xt_end": "{:.4f}",
                "delta_xt": "{:.4f}",
            }),
            use_container_width=True,
            height=400,
        )

# RIGHT: Statistics
with col_stats:
    with st.expander("General Statistics", expanded=False):
        st.markdown('<div class="stats-section-title">Overview</div>', unsafe_allow_html=True)
        row1, row2, row3 = st.columns(3)
        with row1:
            small_metric("Total Passes", f"{stats['total_passes']}")
        with row2:
            small_metric("Successful", f"{stats['successful_passes']}")
        with row3:
            small_metric("Accuracy", f"{stats['accuracy_pct']:.1f}%")

        st.markdown("<hr style='margin:6px 0 8px 0;'>", unsafe_allow_html=True)
        st.markdown('<div class="stats-section-title">Pass Direction</div>', unsafe_allow_html=True)
        dir1, dir2, dir3 = st.columns(3)
        with dir1:
            small_metric("⬆️ Forward", f"{stats['forward_total']} ({stats['pct_forward']:.0f}%)")
        with dir2:
            small_metric("⬇️ Backward", f"{stats['backward_total']} ({stats['pct_backward']:.0f}%)")
        with dir3:
            small_metric("↔️ Lateral", f"{stats['lateral_total']} ({stats['pct_lateral']:.0f}%)")

    with st.expander("Advanced Statistics", expanded=False):
        st.markdown('<div class="stats-section-title">Progressive Passes (Wyscout)</div>', unsafe_allow_html=True)
        pw1, pw2, pw3, pw4 = st.columns(4)
        with pw1:
            small_metric("Total", f"{stats['prog_wyscout_total']}")
        with pw2:
            small_metric("Successful", f"{stats['prog_wyscout_success']}")
        with pw3:
            small_metric("Accuracy", f"{stats['prog_wyscout_accuracy_pct']:.1f}%")
        with pw4:
            small_metric("% of Total", f"{stats['pct_progressive_wyscout']:.1f}%")

        st.markdown("<hr style='margin:6px 0 8px 0;'>", unsafe_allow_html=True)

        st.markdown('<div class="stats-section-title">Expected Threat (xT)</div>', unsafe_allow_html=True)
        xt1, xt2 = st.columns(2)
        with xt1:
            small_metric("xT Σ (Progressive)", f"{stats['xt_prog_sum']:.1f}")
        with xt2:
            small_metric("xT Mean (Progressive)", f"{stats['xt_prog_mean']:.1f}")

        xt3, xt4 = st.columns(2)
        with xt3:
            small_metric("xT Σ (Positive ΔxT)", f"{stats['positive_xt_sum']:.1f}")
        with xt4:
            small_metric("xT Mean (Positive ΔxT)", f"{stats['positive_xt_mean']:.1f}")

    st.divider()
    st.caption("Notas: 'Progressive' segue a definição Wyscout; ΔxT só é contabilizado para passes bem-sucedidos.")
