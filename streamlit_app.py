"""
DECODE X 2026 — STABILIS Dashboard
Stage 1 (Baseline) → Stage 2 (Generalization) → Stage 3 (Future)
All metrics preserved exactly as provided. Stage 3 values are user-inputs.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import json, pathlib

# ─────────────────────────── page config ────────────────────────────────────
st.set_page_config(
    page_title="DECODE X — STABILIS Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────── colour palette ─────────────────────────────────
GREEN = "#28a745"
AMBER = "#ffc107"
RED   = "#dc3545"
GRAY  = "#6c757d"
BG_CARD = "#f8f9fa"

# ─────────────────────────── helper functions ───────────────────────────────

def fmt_inr(v, decimals=0):
    """Format a number as ₹ with Indian comma grouping."""
    if v is None:
        return "N/A"
    if abs(v) >= 1e7:
        return f"₹{v/1e7:,.2f} Cr"
    if abs(v) >= 1e5:
        return f"₹{v/1e5:,.2f} L"
    return f"₹{v:,.{decimals}f}"

def fmt_pct(v, decimals=2):
    if v is None:
        return "N/A"
    return f"{v:.{decimals}f}%"

def fmt_num(v, decimals=4):
    if v is None:
        return "N/A"
    return f"{v:,.{decimals}f}"

def delta_color(delta, higher_is_better=True):
    """Return GREEN / RED / GRAY based on direction."""
    if delta is None or delta == 0:
        return GRAY
    if higher_is_better:
        return GREEN if delta > 0 else RED
    else:
        return GREEN if delta < 0 else RED

def metric_card(label, value, delta_text=None, color=None):
    """Render a styled metric card using markdown."""
    delta_html = ""
    if delta_text is not None and color is not None:
        delta_html = f'<span style="color:{color};font-size:0.85rem;">{delta_text}</span>'
    st.markdown(
        f"""
        <div style="background:{BG_CARD};border-radius:10px;padding:16px 20px;
                     margin-bottom:8px;border-left:4px solid {color or GRAY};">
            <div style="color:#555;font-size:0.80rem;text-transform:uppercase;">{label}</div>
            <div style="font-size:1.5rem;font-weight:700;margin:4px 0;">{value}</div>
            {delta_html}
        </div>""",
        unsafe_allow_html=True,
    )

# ─────────────────────────── STAGE 1 DATA (frozen) ─────────────────────────
S1 = {
    "erpu_mean":        473_409.98,
    "erpu_median":      213_432.22,
    "total_users":      5_798,
    "total_baskets":    41_278,
    "net_revenue":      3_715_345_323.11,
    "return_rate_pct":  8.98,
    "top10_share":      53.08,
    "top1_share":       19.96,
    "gini":             0.6428,
    "auc":              0.7958,
    "brier":            0.1810,
    "top_decile_lift":  1.61,
    "r2_basket":        0.2434,
    "poisson_disp":     4.21,
    "hhi":              0.0025,
    # Bootstrap CIs
    "erpu_mean_ci":     (435_231.31, 513_944.36),
    "erpu_median_ci":   (205_248.90, 223_690.38),
    "top10_ci":         (0.4975, 0.5679),
    # Cluster stability
    "silhouette":       0.4016,
    "ari":              0.9624,
    # ERPU decomposition
    "E_freq":           4.38,
    "E_basket_gross":   97_634,
    "E_return_rate":    0.0990,
    "ERPU_structural":  385_660,
}

S1_CLUSTERS = pd.DataFrame([
    {"Segment": "High-Value Loyalists",           "Users": 1246, "Net Share %": 75.66, "Return Share %": 34.18,
     "Avg Net (₹)": 1_259_903, "Avg Freq": 10.67, "Avg Recency (days)": 27},
    {"Segment": "Core Repeat Buyers",             "Users": 2153, "Net Share %": 23.05, "Return Share %":  5.04,
     "Avg Net (₹)":   222_174, "Avg Freq":  2.13, "Avg Recency (days)": 55},
    {"Segment": "Dormant / One-time Buyers",      "Users":  960, "Net Share %":  2.92, "Return Share %": 22.08,
     "Avg Net (₹)":    63_196, "Avg Freq":  1.29, "Avg Recency (days)": 251},
    {"Segment": "Return-Dominant / Adj. Risk",    "Users":   24, "Net Share %": -1.63, "Return Share %": 38.70,
     "Avg Net (₹)": -1_412_101, "Avg Freq": 3.54, "Avg Recency (days)": 118},
])

# ─────────────────────────── STAGE 2 DATA (frozen) ─────────────────────────
# Stage 2 Interim Validation (holdout, same regime)
S2_VAL = {
    "validation_users": 2_681,
    "training_users":   4_381,
    "train_auc":        0.8019,
    "val_auc":          0.8688,
    "train_brier":      0.1780,
    "val_brier":        0.1852,
    "train_ece":        0.0369,
    "val_ece":          0.1905,
    "recalibrated":     True,
    "recal_method":     "Isotonic Regression",
    "overfit_class":    "MODERATE OVERFIT — Recalibration recommended",
    "auc_drop_pct":     0.0,
    "lift_drop_pct":    0.0,
    "erpu_inflation_pct": 561.81,
    "score_cv_pct":     11.60,
    "erpu_mean":        64_732,
    "erpu_median":      0,
    "top10_share":      66.82,
    "top1_share":       22.68,
    "gini":             0.8334,
    "lift":             2.47,
    # Bootstrap CIs (validation set)
    "erpu_mean_ci":     (57_032, 73_247),
    "top10_ci":         (0.631, 0.702),
    "gini_ci":          (0.8153, 0.8491),
    # Module specs
    "moduleB_disp":     15.55,
    "moduleC_r2":       0.4085,
}

# Stage 2 Shock Recalibration
S2_SHOCK = {
    "users":            1_423,
    "baskets":          2_282,
    "net_revenue":      209_319_301,
    "return_rate_pct":  4.82,
    "erpu_mean":        147_097,
    "erpu_median":      95_005,
    "top10_share":      42.18,
    "top1_share":       12.87,
    "gini":             0.5566,
    "auc":              0.7884,
    "brier":            0.1338,
    "top_decile_lift":  3.18,
    "r2_basket":        0.2518,
    "poisson_disp":     1.43,
    "hhi":              0.0018,
    "ari":              0.8863,
}

# ─────────────────────────── STAGE 2 composite (use validation's model perf
#                              + shock's business KPIs for comparison table)
S2 = {
    "erpu_mean":        S2_VAL["erpu_mean"],
    "erpu_median":      S2_VAL["erpu_median"],
    "return_rate_pct":  S2_SHOCK["return_rate_pct"],
    "top10_share":      S2_VAL["top10_share"],
    "top1_share":       S2_VAL["top1_share"],
    "gini":             S2_VAL["gini"],
    "auc":              S2_VAL["val_auc"],
    "brier":            S2_VAL["val_brier"],
    "top_decile_lift":  S2_VAL["lift"],
    "r2_basket":        S2_VAL["moduleC_r2"],
}

# ─────────────────────────── helpers for change table ───────────────────────

CORE_METRICS = [
    # (label, key, format_fn, higher_is_better)
    ("Mean ERPU (₹)",          "erpu_mean",        lambda v: fmt_inr(v),       True),
    ("Return Rate (%)",        "return_rate_pct",  lambda v: fmt_pct(v),       False),
    ("Top 10 % User Share",    "top10_share",      lambda v: fmt_pct(v),       None),   # ambiguous
    ("Top 1 % User Share",     "top1_share",       lambda v: fmt_pct(v),       None),
    ("Gini Coefficient",       "gini",             lambda v: fmt_num(v),       None),
    ("AUC",                    "auc",              lambda v: fmt_num(v),       True),
    ("Brier Score",            "brier",            lambda v: fmt_num(v),       False),
    ("Top-Decile Lift",        "top_decile_lift",  lambda v: fmt_num(v, 2),    True),
    ("Basket-Value R²",        "r2_basket",        lambda v: fmt_num(v),       True),
]

def build_change_df(stage_a: dict, stage_b: dict, metrics=CORE_METRICS):
    rows = []
    for label, key, fmt, hib in metrics:
        va = stage_a.get(key)
        vb = stage_b.get(key)
        if va is not None and vb is not None:
            abs_d = vb - va
            pct_d = (abs_d / abs(va) * 100) if va != 0 else None
        else:
            abs_d = pct_d = None
        rows.append({
            "Metric": label,
            "Stage A": va,
            "Stage B": vb,
            "Δ (abs)": abs_d,
            "Δ (%)": pct_d,
            "higher_is_better": hib,
        })
    return pd.DataFrame(rows)

def style_change_row(row):
    """Return list of CSS styles for a change-table row."""
    d = row["Δ (abs)"]
    hib = row["higher_is_better"]
    if d is None or hib is None:
        color = GRAY
    elif (d > 0 and hib) or (d < 0 and not hib):
        color = GREEN
    elif (d < 0 and hib) or (d > 0 and not hib):
        color = RED
    else:
        color = GRAY
    return [f"color: {color}" if c in ("Δ (abs)", "Δ (%)") else "" for c in row.index]

# ═══════════════════════════ SIDEBAR — Stage 3 inputs ══════════════════════

with st.sidebar:
    st.header("📥 Stage 3 Inputs")
    st.caption("Enter Stage 3 metrics once available. Leave blank / 0 for unavailable.")
    s3_erpu      = st.number_input("Mean ERPU (₹)", min_value=0.0, value=0.0, step=1000.0, format="%.2f")
    s3_return    = st.number_input("Return Rate (%)", min_value=0.0, value=0.0, step=0.1, format="%.2f")
    s3_top10     = st.number_input("Top 10 % Share (%)", min_value=0.0, value=0.0, step=0.1, format="%.2f")
    s3_top1      = st.number_input("Top 1 % Share (%)", min_value=0.0, value=0.0, step=0.1, format="%.2f")
    s3_gini      = st.number_input("Gini Coefficient", min_value=0.0, value=0.0, step=0.01, format="%.4f")
    s3_auc       = st.number_input("AUC", min_value=0.0, max_value=1.0, value=0.0, step=0.01, format="%.4f")
    s3_brier     = st.number_input("Brier Score", min_value=0.0, value=0.0, step=0.01, format="%.4f")
    s3_lift      = st.number_input("Top-Decile Lift", min_value=0.0, value=0.0, step=0.1, format="%.2f")
    s3_r2        = st.number_input("Basket-Value R²", min_value=0.0, value=0.0, step=0.01, format="%.4f")

    s3_available = any(v > 0 for v in [s3_erpu, s3_return, s3_top10, s3_top1,
                                        s3_gini, s3_auc, s3_brier, s3_lift, s3_r2])

S3 = {
    "erpu_mean":       s3_erpu   if s3_erpu   > 0 else None,
    "return_rate_pct": s3_return  if s3_return  > 0 else None,
    "top10_share":     s3_top10   if s3_top10   > 0 else None,
    "top1_share":      s3_top1    if s3_top1    > 0 else None,
    "gini":            s3_gini    if s3_gini    > 0 else None,
    "auc":             s3_auc     if s3_auc     > 0 else None,
    "brier":           s3_brier   if s3_brier   > 0 else None,
    "top_decile_lift": s3_lift    if s3_lift    > 0 else None,
    "r2_basket":       s3_r2      if s3_r2      > 0 else None,
}

# ═══════════════════════════ TITLE ═════════════════════════════════════════

st.title("📊 DECODE X 2026 — STABILIS")
st.caption("Stage-wise analysis · All numeric values preserved exactly as computed")

# ═══════════════════════════ TABS ══════════════════════════════════════════

tab1, tab2, tab3 = st.tabs([
    "🏠 Baseline (Stage 1)",
    "🔬 Generalization (Stage 2)",
    "🔄 Changes & Impact",
])

# ═══════════════════════════ TAB 1 — STAGE 1 ══════════════════════════════

with tab1:
    st.header("Stage 1 — Baseline Analysis (Full Dataset · Year 1)")

    # ── KPI cards row 1 ───
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Mean ERPU", fmt_inr(S1["erpu_mean"]), color=GREEN)
    with c2:
        metric_card("Median ERPU", fmt_inr(S1["erpu_median"]), color=GREEN)
    with c3:
        metric_card("Net Revenue", fmt_inr(S1["net_revenue"]), color=GREEN)
    with c4:
        metric_card("Total Users", f"{S1['total_users']:,}", color=GREEN)

    # ── KPI cards row 2 ───
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Return Rate", fmt_pct(S1["return_rate_pct"]), color=AMBER)
    with c2:
        metric_card("Top 10 % Share", fmt_pct(S1["top10_share"]), color=AMBER)
    with c3:
        metric_card("Top 1 % Share", fmt_pct(S1["top1_share"]), color=AMBER)
    with c4:
        metric_card("Gini", fmt_num(S1["gini"]), color=AMBER)

    st.divider()

    # ── Model performance ───
    st.subheader("Model Performance")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("AUC (Module A)", fmt_num(S1["auc"]), color=GREEN)
    with c2:
        metric_card("Brier Score", fmt_num(S1["brier"]), color=GREEN)
    with c3:
        metric_card("Top-Decile Lift", fmt_num(S1["top_decile_lift"], 2), color=GREEN)
    with c4:
        metric_card("Basket R²", fmt_num(S1["r2_basket"]), color=GREEN)

    st.divider()

    # ── ERPU Bootstrap CIs ─────
    st.subheader("Bootstrap 95 % Confidence Intervals")
    ci_df = pd.DataFrame([
        {"Metric": "ERPU Mean",           "Point Est.": fmt_inr(S1["erpu_mean"]),
         "CI Low": fmt_inr(S1["erpu_mean_ci"][0]),  "CI High": fmt_inr(S1["erpu_mean_ci"][1])},
        {"Metric": "ERPU Median",          "Point Est.": fmt_inr(S1["erpu_median"]),
         "CI Low": fmt_inr(S1["erpu_median_ci"][0]),"CI High": fmt_inr(S1["erpu_median_ci"][1])},
        {"Metric": "Top 10 % Share",       "Point Est.": fmt_pct(S1["top10_share"]),
         "CI Low": fmt_pct(S1["top10_ci"][0]*100),  "CI High": fmt_pct(S1["top10_ci"][1]*100)},
    ])
    st.dataframe(ci_df, width="stretch", hide_index=True)

    st.divider()

    # ── ERPU Decomposition ─────
    st.subheader("ERPU Structural Decomposition")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("E[Freq]", fmt_num(S1["E_freq"], 2), color=GRAY)
    with c2:
        metric_card("E[Basket Gross]", fmt_inr(S1["E_basket_gross"]), color=GRAY)
    with c3:
        metric_card("E[Return Rate]", fmt_pct(S1["E_return_rate"]*100), color=GRAY)
    with c4:
        metric_card("ERPU Structural", fmt_inr(S1["ERPU_structural"]), color=GRAY)
    st.caption("ERPU_structural = E[freq] × E[basket_gross] × (1 − return_rate)")

    st.divider()

    # ── Cluster table ───
    st.subheader("Customer Segments (K = 4)")
    st.dataframe(S1_CLUSTERS, width="stretch", hide_index=True)

    # cluster donut chart
    fig_cluster = go.Figure(go.Pie(
        labels=S1_CLUSTERS["Segment"],
        values=S1_CLUSTERS["Users"],
        hole=0.4,
        marker_colors=["#28a745", "#17a2b8", "#ffc107", "#dc3545"],
        textinfo="label+percent",
    ))
    fig_cluster.update_layout(title="User Distribution by Segment", height=380)
    st.plotly_chart(fig_cluster, use_container_width=True)  # plotly keeps old param

    st.divider()

    # ── Cluster stability ───
    st.subheader("Cluster Stability")
    c1, c2 = st.columns(2)
    with c1:
        metric_card("Silhouette Score", fmt_num(S1["silhouette"]), color=GREEN)
    with c2:
        metric_card("ARI (10 resamples)", fmt_num(S1["ari"]), color=GREEN)

# ═══════════════════════════ TAB 2 — STAGE 2 ══════════════════════════════

with tab2:
    st.header("Stage 2 — Generalization & Shock Analysis")

    sub1, sub2 = st.tabs(["🧪 Interim Validation (Holdout)", "💥 Shock Recalibration"])

    # ── Sub-tab: Validation ────
    with sub1:
        st.subheader("Holdout Validation — Same Regime")
        st.info(
            f"Training users: **{S2_VAL['training_users']:,}** · "
            f"Validation users: **{S2_VAL['validation_users']:,}**"
        )

        # Discrimination
        st.markdown("#### Discrimination")
        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("Train AUC", fmt_num(S2_VAL["train_auc"]),
                        f"Val AUC: {fmt_num(S2_VAL['val_auc'])} (Δ +8.34 %)", GREEN)
        with c2:
            metric_card("Top-Decile Lift (Val)", fmt_num(S2_VAL["lift"], 2),
                        "vs Train 1.58x → 2.47x (Δ +56.48 %)", GREEN)
        with c3:
            metric_card("AUC Drop", fmt_pct(S2_VAL["auc_drop_pct"]),
                        "No degradation on validation", GREEN)

        # Calibration
        st.markdown("#### Calibration")
        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("Train Brier", fmt_num(S2_VAL["train_brier"]), color=GREEN)
        with c2:
            metric_card("Val Brier", fmt_num(S2_VAL["val_brier"]),
                        "Slight increase as expected", AMBER)
        with c3:
            metric_card("Val ECE", fmt_num(S2_VAL["val_ece"]),
                        f"Recalibrated via {S2_VAL['recal_method']}", AMBER)

        # ERPU & Concentration on validation
        st.markdown("#### ERPU & Concentration (Validation Set)")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("Mean ERPU", fmt_inr(S2_VAL["erpu_mean"]),
                        "Δ −80.07 % vs training", RED)
        with c2:
            metric_card("Median ERPU", fmt_inr(S2_VAL["erpu_median"]),
                        "Median = 0 (61.4 % inactive)", RED)
        with c3:
            metric_card("Top 10 % Share", fmt_pct(S2_VAL["top10_share"]),
                        "Δ +5.20 %", AMBER)
        with c4:
            metric_card("Gini", fmt_num(S2_VAL["gini"]),
                        "Δ +5.79 %", AMBER)

        # Overfitting
        st.markdown("#### Overfitting Diagnosis")
        st.warning(S2_VAL["overfit_class"])
        c1, c2 = st.columns(2)
        with c1:
            metric_card("ERPU Inflation %", fmt_pct(S2_VAL["erpu_inflation_pct"]),
                        "Training ERPU >> Validation ERPU", RED)
        with c2:
            metric_card("Score CV %", fmt_pct(S2_VAL["score_cv_pct"]),
                        "Moderate instability", AMBER)

        # Bootstrap CIs (Val)
        st.markdown("#### Bootstrap 95 % CIs (Validation, 1 000 resamples)")
        val_ci = pd.DataFrame([
            {"Metric": "Mean ERPU",    "Point": fmt_inr(S2_VAL["erpu_mean"]),
             "CI Low": fmt_inr(S2_VAL["erpu_mean_ci"][0]),
             "CI High": fmt_inr(S2_VAL["erpu_mean_ci"][1])},
            {"Metric": "Top 10 %",     "Point": fmt_pct(S2_VAL["top10_share"]),
             "CI Low": fmt_pct(S2_VAL["top10_ci"][0]*100),
             "CI High": fmt_pct(S2_VAL["top10_ci"][1]*100)},
            {"Metric": "Gini",         "Point": fmt_num(S2_VAL["gini"]),
             "CI Low": fmt_num(S2_VAL["gini_ci"][0]),
             "CI High": fmt_num(S2_VAL["gini_ci"][1])},
        ])
        st.dataframe(val_ci, width="stretch", hide_index=True)

        st.caption("Narrow CIs → stable estimates → robust to sampling variation.")

    # ── Sub-tab: Shock ────
    with sub2:
        st.subheader("Shock Recalibration — Regime Change")
        st.error(
            "The shock period introduced **material disruption**: "
            "users −65 %, baskets −87 %, net revenue −87 %."
        )

        # KPIs
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("Users", f"{S2_SHOCK['users']:,}",
                        "−65.0 % vs baseline Y2", RED)
        with c2:
            metric_card("Mean ERPU", fmt_inr(S2_SHOCK["erpu_mean"]),
                        "−63.5 % vs baseline Y2", RED)
        with c3:
            metric_card("Return Rate", fmt_pct(S2_SHOCK["return_rate_pct"]),
                        "Down from 7.79 % (improvement)", GREEN)
        with c4:
            metric_card("Gini", fmt_num(S2_SHOCK["gini"]),
                        "−12.5 % — less concentrated", GREEN)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("AUC", fmt_num(S2_SHOCK["auc"]),
                        "Partially resilient", AMBER)
        with c2:
            metric_card("Brier Score", fmt_num(S2_SHOCK["brier"]),
                        "Improved vs baseline", GREEN)
        with c3:
            metric_card("Top-Decile Lift", fmt_num(S2_SHOCK["top_decile_lift"], 2),
                        "Strong lift under shock", GREEN)
        with c4:
            metric_card("Top 10 % Share", fmt_pct(S2_SHOCK["top10_share"]),
                        "−18.9 % vs baseline Y2", GREEN)

        # Before vs After table
        st.markdown("#### Before vs After (Baseline Y2 → Shock)")
        shock_compare = pd.DataFrame([
            {"Metric": "Users",              "Baseline Y2": "4,068",     "Shock": "1,423",     "Δ": "−65.0 %"},
            {"Metric": "Baskets",            "Baseline Y2": "17,691",    "Shock": "2,282",     "Δ": "−87.1 %"},
            {"Metric": "Net Revenue",        "Baseline Y2": "₹164.04 Cr","Shock": "₹20.93 Cr","Δ": "−87.2 %"},
            {"Metric": "ERPU Mean",          "Baseline Y2": "₹4.03 L",  "Shock": "₹1.47 L",  "Δ": "−63.5 %"},
            {"Metric": "ERPU Median",        "Baseline Y2": "₹1.85 L",  "Shock": "₹0.95 L",  "Δ": "−48.7 %"},
            {"Metric": "Return % of Gross",  "Baseline Y2": "7.79 %",   "Shock": "4.82 %",   "Δ": "−38.2 %"},
            {"Metric": "Top 10 % Share",     "Baseline Y2": "52.0 %",   "Shock": "42.2 %",   "Δ": "−18.9 %"},
            {"Metric": "Gini",               "Baseline Y2": "0.6357",   "Shock": "0.5566",   "Δ": "−12.5 %"},
        ])
        st.dataframe(shock_compare, width="stretch", hide_index=True)

        # Structural risks
        st.markdown("#### Structural Risks Remaining")
        st.markdown("""
        - **Revenue concentration**: Top 10 % users hold 42.2 % of net — churn in this segment is catastrophic.
        - **Return rate** at 4.82 % — if trending upward, erodes margins faster than volume can compensate.
        - **Frequency** is the primary shock driver in ERPU decomposition.
        - Adjustment SKU exclusion changes ERPU by only +0.44 % — robust.
        """)

# ═══════════════════════════ TAB 3 — CHANGES & IMPACT ══════════════════════

with tab3:
    st.header("Changes & Impact")

    # ── Stage 1 → Stage 2 ───
    st.subheader("Stage 1 → Stage 2 (Validation)")

    change_12 = build_change_df(S1, S2)

    # Format for display — build with string dtype to avoid FutureWarning
    disp_rows = []
    for idx, (label, key, fmt, hib) in enumerate(CORE_METRICS):
        va = change_12.at[idx, "Stage A"]
        vb = change_12.at[idx, "Stage B"]
        d  = change_12.at[idx, "Δ (abs)"]
        p  = change_12.at[idx, "Δ (%)"]
        disp_rows.append({
            "Metric":  label,
            "Stage 1": fmt(va) if va is not None else "N/A",
            "Stage 2": fmt(vb) if vb is not None else "N/A",
            "Δ (abs)": f"{d:+,.4f}" if d is not None else "—",
            "Δ (%)": f"{p:+.2f} %" if p is not None else "—",
        })
    disp_12 = pd.DataFrame(disp_rows)
    st.dataframe(disp_12, width="stretch", hide_index=True)

    # Color-coded narrative
    st.markdown(f"""
    <div style="background:#e8f5e9;padding:12px;border-radius:8px;margin-bottom:8px;">
        <b style="color:{GREEN};">✅ AUC improved</b> from 0.7958 → 0.8688 (+9.17 %) on validation — discrimination generalises well.
    </div>
    <div style="background:#e8f5e9;padding:12px;border-radius:8px;margin-bottom:8px;">
        <b style="color:{GREEN};">✅ Top-Decile Lift improved</b> from 1.61 → 2.47 (+53.42 %) — model separates high-value users better on holdout.
    </div>
    <div style="background:#fce4ec;padding:12px;border-radius:8px;margin-bottom:8px;">
        <b style="color:{RED};">⚠️ Mean ERPU dropped</b> from ₹4.73 L → ₹0.65 L (−86.32 %) — driven by a higher proportion of
        inactive users (61.4 %) in the validation sample.
    </div>
    <div style="background:#fff8e1;padding:12px;border-radius:8px;margin-bottom:8px;">
        <b style="color:{AMBER};">⚠ Concentration increased</b>: Top 10 % share rose from 53.08 % → 66.82 % and Gini from 0.6428 → 0.8334.
        Revenue is more concentrated in the validation cohort.
    </div>
    <div style="background:#fff8e1;padding:12px;border-radius:8px;margin-bottom:8px;">
        <b style="color:{AMBER};">⚠ Brier Score marginally worse</b> (0.1810 → 0.1852) — expected on holdout; recalibration applied.
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Visualisation: multi-stage comparison ───
    st.subheader("Key Metrics Across Stages")

    vis_metrics = [
        ("Mean ERPU (₹)", "erpu_mean",       True),
        ("Top 10 % Share (%)", "top10_share", None),
        ("Gini Coefficient",   "gini",        None),
        ("Return Rate (%)",    "return_rate_pct", False),
        ("AUC",                "auc",         True),
        ("Top-Decile Lift",    "top_decile_lift", True),
    ]

    col_left, col_right = st.columns(2)
    for i, (title, key, hib) in enumerate(vis_metrics):
        container = col_left if i % 2 == 0 else col_right
        with container:
            stages  = ["Stage 1", "Stage 2"]
            vals    = [S1.get(key), S2.get(key)]
            colors  = [GREEN, AMBER]

            if s3_available and S3.get(key) is not None:
                stages.append("Stage 3")
                vals.append(S3[key])
                colors.append("#6f42c1")

            fig = go.Figure()
            for s, v, c in zip(stages, vals, colors):
                if v is not None:
                    fig.add_trace(go.Bar(
                        x=[s], y=[v], name=s,
                        marker_color=c,
                        text=[f"{v:,.2f}"],
                        textposition="outside",
                    ))

            # Placeholder for Stage 3 if not provided
            if not (s3_available and S3.get(key) is not None):
                fig.add_trace(go.Bar(
                    x=["Stage 3"], y=[0], name="Stage 3 (TBD)",
                    marker_color="#ddd",
                    marker_line=dict(width=2, color="#999"),
                    text=["N/A"],
                    textposition="outside",
                ))

            fig.update_layout(
                title=title,
                showlegend=False,
                height=300,
                yaxis_title="",
                margin=dict(t=40, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)  # plotly keeps old param

    st.divider()

    # ── Stage 2 → Stage 3 comparison ──────────────────────────────────────
    st.subheader("Stage 2 → Stage 3")

    if not s3_available:
        st.info(
            "🔮 **Stage 3 data not yet available.** "
            "Enter Stage 3 metrics in the sidebar to enable this comparison."
        )
    else:
        change_23 = build_change_df(S2, S3)
        disp_rows_23 = []
        for idx, (label, key, fmt, hib) in enumerate(CORE_METRICS):
            va = change_23.at[idx, "Stage A"]
            vb = change_23.at[idx, "Stage B"]
            d  = change_23.at[idx, "Δ (abs)"]
            p  = change_23.at[idx, "Δ (%)"]
            disp_rows_23.append({
                "Metric":  label,
                "Stage 2": fmt(va) if va is not None else "N/A",
                "Stage 3": fmt(vb) if vb is not None else "N/A",
                "Δ (abs)": f"{d:+,.4f}" if d is not None else "—",
                "Δ (%)": f"{p:+.2f} %" if p is not None else "—",
            })
        disp_23 = pd.DataFrame(disp_rows_23)
        st.dataframe(disp_23, width="stretch", hide_index=True)

        st.markdown(
            "Stage 3 directive imposed new constraints — review deltas above for "
            "shifts in concentration, returns, and model performance."
        )

    st.divider()

    # ── Stage 3 potential directions ──────────────────────────────────────
    st.subheader("Stage 3 — Optimization Levers (Provisional)")
    st.markdown("""
    | # | Lever | Description |
    |---|-------|-------------|
    | 1 | **Frequency Uplift** | Target mid-tier users with purchase acceleration campaigns |
    | 2 | **Basket Value Expansion** | Use Module C time-slot coefficients for promotion targeting |
    | 3 | **Return Rate Reduction** | Root-cause analysis on top return SKUs; policy intervention modelling |
    | 4 | **Concentration De-risking** | LTV-based diversification to reduce top-10 % dependency below 50 % |
    | 5 | **Churn Prevention** | Module A probabilities → tiered retention for at-risk high-value users |
    | 6 | **Dynamic Pricing** | Gamma GLM predictions for basket value × demand elasticity |
    """)
    st.caption("No final decisions made. Awaiting Stage 3 analysis.")

# ═══════════════════════════ FOOTER ════════════════════════════════════════
st.divider()
st.caption("DECODE X 2026 · Case STABILIS · Dashboard generated 2026-02-28 · All metrics frozen from pipeline outputs")
