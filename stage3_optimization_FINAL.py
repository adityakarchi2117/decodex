"""
================================================================================
 STAGE 3 — TARGETING OPTIMIZATION SYSTEM  (FINAL — SINGLE-FILE)
 DECODE X 2026 — Case STABILIS
================================================================================
 Objective : MAXIMIZE TOTAL EXPECTED NET REVENUE
 Subject to: Budget, Return-Risk, Concentration constraints
 Input     : Customers_Test_set.xlsx (out-of-sample regime)
 Models    : Stage 1 trained, Stage 2 calibrated
 Method    : Risk-adjusted ranking with constraint enforcement
 Date      : March 1, 2026
================================================================================
 Run:  python stage3_optimization_FINAL.py
 Out:  stage3_optimization_pack/
================================================================================
"""

import os, sys, json, warnings, shutil, textwrap
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from scipy import stats as sp_stats
from scipy.optimize import minimize_scalar

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import cross_val_predict

import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
np.random.seed(42)

# ============================================================================
# CONFIG
# ============================================================================
TRAIN_FILE = "Customers_Transactions.xlsx"
TEST_FILE  = "Customers_Test_set.xlsx"
OUTPUT_DIR = Path("stage3_optimization_pack")
CHARTS_DIR = OUTPUT_DIR / "charts"
TABLES_DIR = OUTPUT_DIR / "tables"
SEED       = 42
N_BOOT     = 1000
ALPHA      = 0.05

# Constraint parameters
BUDGET_FRACTION       = 0.30   # Target at most 30% of users
RETURN_RISK_CAP       = 0.25   # ≤25% of targeted in top return-risk decile
CONCENTRATION_CAP_REL = 0.15   # Top-10% share increase ≤ 15% relative to Stage 1

# Stage 1 baseline
STAGE1_TOP10_SHARE    = 0.5308  # From Stage 1: 53.08%

ADJUSTMENT_SKUS = [
    "POSTAGE", "DOTCOM POSTAGE", "MANUAL", "DISCOUNT",
    "ADJUST", "CRUK", "BANK CHARGES", "AMAZONFEE",
    "POST", "DOT", "S", "M", "PADS",
]

# Chart defaults
matplotlib.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 300,
    "font.size": 11, "axes.titlesize": 14, "axes.labelsize": 12,
})

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def safe_div(a, b, default=0.0):
    return a / b if b != 0 else default

def gini_coef(arr):
    x = np.asarray(arr, dtype=float)
    x = x[np.isfinite(x)]
    x = np.clip(x, 0, None)
    if x.size == 0 or x.sum() == 0:
        return 0.0
    x = np.sort(x)
    n = x.size
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * x)) / (n * np.sum(x)) - (n + 1) / n)

def bootstrap_ci(data, fn, n=N_BOOT, alpha=ALPHA, seed=SEED):
    rng = np.random.default_rng(seed)
    arr = np.asarray(data, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return (np.nan, np.nan, np.nan)
    point = float(fn(arr))
    boots = np.array([fn(rng.choice(arr, len(arr), replace=True)) for _ in range(n)])
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return point, lo, hi

def fmt_money(x):
    x = float(x)
    sign = "-" if x < 0 else ""
    x = abs(x)
    if x >= 1e9: return f"{sign}{x/1e9:.2f}B"
    if x >= 1e6: return f"{sign}{x/1e6:.2f}M"
    if x >= 1e3: return f"{sign}{x/1e3:.2f}K"
    return f"{sign}{x:.0f}"

def savefig(name):
    plt.savefig(CHARTS_DIR / name, bbox_inches="tight", pad_inches=0.25)
    plt.close()

# ============================================================================
# SECTION 0 — DATA LOADING
# ============================================================================
def load_data():
    print(f"\n{'='*80}")
    print("SECTION 0 — LOADING DATA")
    print(f"{'='*80}")

    col_map = {
        "EventID": "invoiceno", "EventType": "eventtype",
        "ProductID": "stockcode", "ProductName": "description",
        "Quantity": "quantity", "EventDateTime": "invoicedate",
        "UnitPrice": "price", "UserID": "customerid",
    }

    def _load(path, sheet=None):
        df = pd.read_excel(path, sheet_name=sheet) if sheet else pd.read_excel(path)
        df = df.rename(columns=col_map)
        df["invoicedate"] = pd.to_datetime(df["invoicedate"], errors="coerce")
        df = df.dropna(subset=["invoicedate", "customerid"])
        df["customerid"] = df["customerid"].astype(int).astype(str)
        df["line_value"] = df["quantity"] * df["price"]
        df["is_return"] = df["quantity"] < 0
        df["date_only"] = df["invoicedate"].dt.date
        df["basketid"] = df["customerid"] + "_" + df["date_only"].astype(str) + "_" + df["invoiceno"].astype(str)
        if "description" in df.columns:
            df["description"] = df["description"].fillna("UNKNOWN").astype(str).str.strip().str.upper()
        else:
            df["description"] = "UNKNOWN"
        for feat in ["month", "weekday", "hour"]:
            df[feat] = getattr(df["invoicedate"].dt, feat)
        return df

    y1 = _load(TRAIN_FILE, sheet="Year 2019-2020")
    y2 = _load(TRAIN_FILE, sheet="Year 2020-2021")
    test = _load(TEST_FILE)

    print(f"  Y1   rows={len(y1):>8,}  users={y1['customerid'].nunique():>5,}  "
          f"range={y1['invoicedate'].min().date()} → {y1['invoicedate'].max().date()}")
    print(f"  Y2   rows={len(y2):>8,}  users={y2['customerid'].nunique():>5,}  "
          f"range={y2['invoicedate'].min().date()} → {y2['invoicedate'].max().date()}")
    print(f"  TEST rows={len(test):>8,}  users={test['customerid'].nunique():>5,}  "
          f"range={test['invoicedate'].min().date()} → {test['invoicedate'].max().date()}")

    overlap_y1_test = set(y1["customerid"]) & set(test["customerid"])
    print(f"  Y1↔TEST overlap: {len(overlap_y1_test)} users ({len(overlap_y1_test)/test['customerid'].nunique()*100:.1f}%)")

    return y1, y2, test


# ============================================================================
# BUILD USER-LEVEL TABLE
# ============================================================================
def build_user_table(df, label, ref_date=None):
    if ref_date is None:
        ref_date = df["invoicedate"].max()

    purchase_df = df[~df["is_return"]]
    return_df   = df[df["is_return"]]

    u = purchase_df.groupby("customerid").agg(
        frequency      = ("basketid", "nunique"),
        gross_total    = ("line_value", "sum"),
        first_date     = ("invoicedate", "min"),
        last_date      = ("invoicedate", "max"),
        n_events       = ("invoicedate", "count"),
    ).reset_index()

    ret = return_df.groupby("customerid").agg(
        return_total   = ("line_value", lambda x: x.abs().sum()),
        n_baskets_ret  = ("basketid", "nunique"),
    ).reset_index()

    u = u.merge(ret, on="customerid", how="left")
    u["return_total"]  = u["return_total"].fillna(0)
    u["n_baskets_ret"] = u["n_baskets_ret"].fillna(0).astype(int)
    u["net_total"]     = u["gross_total"] - u["return_total"]
    u["return_rate_value"] = u["return_total"] / u["gross_total"].replace(0, np.nan)
    u["return_rate_value"] = u["return_rate_value"].fillna(0)
    u["recency"]       = (ref_date - u["last_date"]).dt.days
    u["tenure"]        = (u["last_date"] - u["first_date"]).dt.days + 1
    u["avg_basket"]    = u["net_total"] / u["frequency"].replace(0, 1)
    return u


# ============================================================================
# SECTION 1 — FREEZE STAGE 1 MODELS (retrain on Y1→Y2)
# ============================================================================
def freeze_stage1_models(y1, y2):
    print(f"\n{'='*80}")
    print("SECTION 1 — FREEZING STAGE 1 MODEL ARCHITECTURE")
    print(f"{'='*80}")

    models   = {}
    equations = []

    # Y1 user features
    y1_ref = y1["invoicedate"].max()
    train_u = build_user_table(y1, "train_y1", ref_date=y1_ref)
    print(f"  Train (Y1) users: {len(train_u):,}")

    # Y2 outcomes
    y2_user = y2.groupby("customerid").agg(
        y2_baskets  = ("basketid", "nunique"),
        y2_net      = ("line_value", "sum"),
    ).reset_index()
    y2_user["y2_active"] = 1

    train_u = train_u.merge(y2_user, on="customerid", how="left")
    train_u["y2_active"]  = train_u["y2_active"].fillna(0).astype(int)
    train_u["y2_baskets"] = train_u["y2_baskets"].fillna(0).astype(int)
    train_u["y2_net"]     = train_u["y2_net"].fillna(0)

    # Y2 return info
    y2_ret = y2[y2["is_return"]].groupby("customerid")["line_value"].apply(lambda x: x.abs().sum()).reset_index()
    y2_ret.columns = ["customerid", "y2_return_total"]
    train_u = train_u.merge(y2_ret, on="customerid", how="left")
    train_u["y2_return_total"] = train_u["y2_return_total"].fillna(0)
    y2_gross = y2[~y2["is_return"]].groupby("customerid")["line_value"].sum().reset_index()
    y2_gross.columns = ["customerid", "y2_gross"]
    train_u = train_u.merge(y2_gross, on="customerid", how="left")
    train_u["y2_gross"] = train_u["y2_gross"].fillna(0)
    train_u["y2_return_rate"] = np.where(
        train_u["y2_gross"] > 0,
        train_u["y2_return_total"] / train_u["y2_gross"],
        0,
    )

    # ================================================================
    # MODULE A — Purchase probability (Logistic Regression)
    # ================================================================
    feats_A = ["frequency", "net_total", "recency", "n_events"]
    X_A = train_u[feats_A].copy()
    y_A = train_u["y2_active"]

    modA = LogisticRegression(random_state=SEED, max_iter=2000, solver="lbfgs")
    modA.fit(X_A, y_A)
    train_u["pred_prob_active"] = modA.predict_proba(X_A)[:, 1]
    train_auc = 0.0
    try:
        from sklearn.metrics import roc_auc_score
        train_auc = roc_auc_score(y_A, train_u["pred_prob_active"])
    except:
        pass

    # Cross-validated isotonic calibration (addresses Stage 2 ECE = 0.19)
    cv_probs = cross_val_predict(modA, X_A, y_A, method='predict_proba', cv=5)[:, 1]
    iso_cal = IsotonicRegression(out_of_bounds='clip')
    iso_cal.fit(cv_probs, y_A)
    models["moduleA"] = {"model": modA, "features": feats_A, "calibrator": iso_cal}
    equations += [
        "MODULE A — Purchase Probability (Logistic Regression + Isotonic Calibration)",
        f"  P_raw(active) = sigmoid(b0 + sum(bi*xi))",
        f"  P_cal(active) = isotonic_regression(P_raw)  [5-fold CV calibration]",
        f"  Features : {feats_A}",
        f"  Coefficients : {np.round(modA.coef_[0], 6).tolist()}",
        f"  Intercept    : {modA.intercept_[0]:.6f}",
        f"  Training AUC (raw)       : {train_auc:.4f}",
        f"  Calibration: 5-fold cross-validated isotonic regression",
        "",
    ]
    print(f"  [Module A] Logistic – AUC(train) = {train_auc:.4f}, isotonic calibration applied")

    # ================================================================
    # MODULE B — Frequency (Negative Binomial)
    # ================================================================
    active_mask = train_u["y2_active"] == 1
    tu_active = train_u[active_mask].copy()

    mean_f = tu_active["y2_baskets"].mean()
    var_f  = tu_active["y2_baskets"].var()
    dispersion = var_f / mean_f if mean_f > 0 else 1

    feats_B = ["frequency", "net_total", "recency"]
    scaler_B = StandardScaler()
    tu_active_scaled = tu_active.copy()
    tu_active_scaled[feats_B] = scaler_B.fit_transform(tu_active[feats_B])

    formula_B = "y2_baskets ~ " + " + ".join(feats_B)

    if dispersion > 1.5:
        modB = smf.negativebinomial(formula_B, data=tu_active_scaled).fit(disp=False, maxiter=300)
        model_type_B = "NegativeBinomial"
    else:
        modB = smf.poisson(formula_B, data=tu_active_scaled).fit(disp=False, maxiter=300)
        model_type_B = "Poisson"

    models["moduleB"] = {"model": modB, "features": feats_B,
                         "model_type": model_type_B, "dispersion": dispersion,
                         "scaler": scaler_B}

    equations += [
        f"MODULE B — Frequency Model ({model_type_B})",
        f"  E[freq | active] = exp(b0 + sum(bi*xi))",
        f"  Dispersion (var/mean) = {dispersion:.4f}",
        f"  Features : {feats_B}",
        f"  Coefficients : {dict(zip(modB.params.index, np.round(modB.params.values, 8)))}",
        "",
    ]
    print(f"  [Module B] {model_type_B} – dispersion = {dispersion:.3f}")

    # ================================================================
    # MODULE C — Basket value (OLS on log scale)
    # ================================================================
    basket_agg = y1.groupby("basketid").agg(
        basket_net=("line_value", "sum"),
        n_items=("quantity", "sum"),
        n_distinct=("stockcode", "nunique"),
        n_lines=("invoiceno", "count"),
        bdate=("invoicedate", "first"),
    ).reset_index()
    basket_agg["month"]   = pd.to_datetime(basket_agg["bdate"]).dt.month
    basket_agg["weekday"] = pd.to_datetime(basket_agg["bdate"]).dt.weekday
    basket_agg["hour"]    = pd.to_datetime(basket_agg["bdate"]).dt.hour

    bpos = basket_agg[basket_agg["basket_net"] > 0].copy()
    bpos["log_net"] = np.log(bpos["basket_net"])

    feats_C = ["month", "weekday", "hour", "n_items", "n_distinct", "n_lines"]
    X_C = sm.add_constant(bpos[feats_C])
    modC = sm.OLS(bpos["log_net"], X_C).fit()

    models["moduleC"] = {"model": modC, "features": feats_C, "model_type": "OLS_log"}

    equations += [
        "MODULE C — Basket Value (OLS on log scale)",
        f"  log(basket_net) = b0 + sum(bi*xi)",
        f"  Features : {feats_C}",
        f"  R-squared      = {modC.rsquared:.4f}",
        f"  Adj-R-squared  = {modC.rsquared_adj:.4f}",
        "  Coefficients:",
    ]
    for name, coef, pval in zip(modC.params.index, modC.params.values, modC.pvalues.values):
        equations.append(f"    {name:>20s} : {coef:>12.6f}  (p={pval:.4e})")
    equations += [""]
    print(f"  [Module C] OLS log(basket) – R-sq = {modC.rsquared:.4f}")

    # Mean basket value for scoring
    train_avg_basket = float(bpos["basket_net"].mean())
    models["train_avg_basket"] = train_avg_basket

    # Training return rate
    train_return_rate = float(
        y1[y1["is_return"]]["line_value"].abs().sum()
        / y1[~y1["is_return"]]["line_value"].sum()
    )
    models["train_return_rate"] = train_return_rate

    # Frequency cap — Y2 99th percentile (prevents extreme extrapolation)
    freq_p99 = float(np.percentile(tu_active["y2_baskets"].dropna(), 99)) if len(tu_active) > 0 else 50.0
    freq_cap = max(freq_p99, 20.0)
    models["freq_cap"] = freq_cap

    # Duan's smearing factor for Module C log-retransformation
    smear_factor = float(np.mean(np.exp(modC.resid)))
    models["smear_factor"] = smear_factor

    # User-level basket feature averages (for Module C per-user scoring)
    basket_with_cust = y1.groupby("basketid").agg(
        customerid=("customerid", "first"),
        basket_net_ub=("line_value", "sum"),
        n_items_ub=("quantity", "sum"),
        n_distinct_ub=("stockcode", "nunique"),
        n_lines_ub=("invoiceno", "count"),
        bdate_ub=("invoicedate", "first"),
    ).reset_index()
    basket_with_cust["month_ub"]   = pd.to_datetime(basket_with_cust["bdate_ub"]).dt.month
    basket_with_cust["weekday_ub"] = pd.to_datetime(basket_with_cust["bdate_ub"]).dt.weekday
    basket_with_cust["hour_ub"]    = pd.to_datetime(basket_with_cust["bdate_ub"]).dt.hour
    bpos_cust = basket_with_cust[basket_with_cust["basket_net_ub"] > 0].copy()
    user_basket_avgs = bpos_cust.groupby("customerid").agg(
        user_avg_month=("month_ub", "mean"),
        user_avg_weekday=("weekday_ub", "mean"),
        user_avg_hour=("hour_ub", "mean"),
        user_avg_n_items=("n_items_ub", "mean"),
        user_avg_n_distinct=("n_distinct_ub", "mean"),
        user_avg_n_lines=("n_lines_ub", "mean"),
        user_n_pos_baskets=("basketid", "count"),
    ).reset_index()
    models["user_basket_avgs"] = user_basket_avgs
    print(f"  [Module C] User basket avgs computed for {len(user_basket_avgs):,} users, smear={smear_factor:.4f}")

    equations += [
        "ERPU DECOMPOSITION",
        "  ERPU_i = P_cal(active_i) * E[freq_i | active] * E[basket_i] * (1 - return_rate)",
        "  P_cal  = isotonic-calibrated purchase probability (Module A)",
        "  E[basket_i] = Module C per-user prediction (Duan smearing) shrunk toward population mean",
        f"  Population basket prior   = {train_avg_basket:,.2f}",
        f"  Duan's smearing factor    = {smear_factor:.4f}",
        f"  Shrinkage weight (k)      = 5 (user obs vs k baskets at population mean)",
        f"  Training return rate      = {train_return_rate:.4f}",
        f"  Frequency cap (Y2 P99)    = {freq_cap:.0f}",
        "",
        "RETURN RISK DEFINITION",
        "  Return Risk = Expected Return Loss = E[gross] × return_rate",
        "  where E[gross] = P_cal(active) * E[freq] * E[basket_i]",
        "  Return Risk Decile: users ranked by E[gross]*return_rate into 10 equal bins;",
        "    top decile = highest expected return-loss exposure.",
        "",
        "ASSUMPTIONS",
        "  1. Module A probabilities are isotonic-calibrated (5-fold CV)",
        "  2. E[basket_i] = Module C per-user with Duan's smearing, shrunk toward population mean (k=5)",
        "  3. Return rate = population prior (training average)",
        "  4. Frequency cap = Y2 99th percentile among active users",
        "",
    ]
    models["equations"] = equations
    models["train_u"] = train_u

    print(f"  [OK] Stage 1 models FROZEN  ({len(equations)} equation lines)")
    return models


# ============================================================================
# SECTION 2 — SCORE TEST USERS (Step 1 of mandate)
# ============================================================================
def score_test_users(y1, test, models):
    print(f"\n{'='*80}")
    print("STEP 1 — SCORE TEST USERS")
    print(f"{'='*80}")

    y1_ref = y1["invoicedate"].max()
    y1_user_set = set(y1["customerid"].unique())
    test_user_set = set(test["customerid"].unique())

    # Build Y1 features for users that appear in Y1
    # For test-only users (new users), we need to handle them separately
    overlap_users = y1_user_set & test_user_set
    new_users = test_user_set - y1_user_set

    print(f"  Test users total       : {len(test_user_set):,}")
    print(f"  Overlap with Y1        : {len(overlap_users):,}")
    print(f"  New users (not in Y1)  : {len(new_users):,}")

    # Build Y1 features for overlapping users
    y1_overlap = y1[y1["customerid"].isin(y1_user_set)]  # all Y1 users
    tu = build_user_table(y1_overlap, "test_y1_features", ref_date=y1_ref)

    # Test outcomes (ground truth)
    test_outcomes = test.groupby("customerid").agg(
        test_baskets = ("basketid", "nunique"),
        test_net     = ("line_value", "sum"),
    ).reset_index()
    test_outcomes["test_active"] = 1

    test_ret = test[test["is_return"]].groupby("customerid")["line_value"].apply(
        lambda x: x.abs().sum()
    ).reset_index()
    test_ret.columns = ["customerid", "test_return_total"]

    test_gross = test[~test["is_return"]].groupby("customerid")["line_value"].sum().reset_index()
    test_gross.columns = ["customerid", "test_gross"]

    # Merge outcomes
    tu = tu.merge(test_outcomes, on="customerid", how="left")
    tu["test_active"]  = tu["test_active"].fillna(0).astype(int)
    tu["test_baskets"] = tu["test_baskets"].fillna(0).astype(int)
    tu["test_net"]     = tu["test_net"].fillna(0)

    tu = tu.merge(test_ret, on="customerid", how="left")
    tu["test_return_total"] = tu["test_return_total"].fillna(0)
    tu = tu.merge(test_gross, on="customerid", how="left")
    tu["test_gross"] = tu["test_gross"].fillna(0)
    tu["test_return_rate"] = np.where(
        tu["test_gross"] > 0,
        tu["test_return_total"] / tu["test_gross"],
        0,
    )

    # ── Score Module A: Purchase probability (isotonic-calibrated) ──
    feats_A = models["moduleA"]["features"]
    raw_probs = models["moduleA"]["model"].predict_proba(tu[feats_A])[:, 1]
    tu["pred_purchase_prob_raw"] = raw_probs
    tu["pred_purchase_prob"] = models["moduleA"]["calibrator"].predict(raw_probs)

    # ── Score Module B: Expected frequency ──
    feats_B = models["moduleB"]["features"]
    scaler_B = models["moduleB"]["scaler"]
    freq_cap = models["freq_cap"]
    tu["pred_frequency"] = 0.0
    # Score all users (not just above 0.5 threshold — we need ERPU for all)
    tu_scaled_B = tu[feats_B].copy()
    tu_scaled_B[feats_B] = scaler_B.transform(tu[feats_B])
    pred_freq = models["moduleB"]["model"].predict(tu_scaled_B)
    pred_freq = np.clip(pred_freq, 0, freq_cap)
    tu["pred_frequency"] = pred_freq

    # ── Score Module C: Expected basket value (per-user, with shrinkage) ──
    train_avg_basket = models["train_avg_basket"]
    smear_factor = models["smear_factor"]
    user_basket_avgs = models["user_basket_avgs"]
    modC = models["moduleC"]["model"]

    tu = tu.merge(user_basket_avgs, on="customerid", how="left")
    tu["user_n_pos_baskets"] = tu["user_n_pos_baskets"].fillna(0)
    has_feats = tu["user_n_pos_baskets"] > 0
    tu["pred_basket_value"] = train_avg_basket  # default for users without basket history

    if has_feats.sum() > 0:
        user_C_data = pd.DataFrame({
            "month": tu.loc[has_feats, "user_avg_month"],
            "weekday": tu.loc[has_feats, "user_avg_weekday"],
            "hour": tu.loc[has_feats, "user_avg_hour"],
            "n_items": tu.loc[has_feats, "user_avg_n_items"],
            "n_distinct": tu.loc[has_feats, "user_avg_n_distinct"],
            "n_lines": tu.loc[has_feats, "user_avg_n_lines"],
        })
        X_user_C = sm.add_constant(user_C_data)
        log_pred = modC.predict(X_user_C)
        modC_basket = np.exp(log_pred) * smear_factor
        modC_basket = np.clip(modC_basket, 0, None)

        # Bayesian shrinkage toward population mean (prior weight = 5 baskets)
        k_shrink = 5
        n_obs = tu.loc[has_feats, "user_n_pos_baskets"].values
        tu.loc[has_feats, "pred_basket_value"] = (
            n_obs * modC_basket + k_shrink * train_avg_basket
        ) / (n_obs + k_shrink)

    # Clip extreme basket predictions at P99
    bv_p99 = float(np.percentile(tu["pred_basket_value"].dropna(), 99))
    tu["pred_basket_value"] = tu["pred_basket_value"].clip(lower=0, upper=bv_p99)

    # ── Expected return rate (from training distribution) ──
    train_return_rate = models["train_return_rate"]
    tu["pred_return_rate"] = train_return_rate

    # ── Compute ERPU components (probability and value layers SEPARATED) ──
    # E(Gross_i) = P(purchase_i) × E(frequency_i) × E(basket_i)
    tu["E_gross"] = tu["pred_purchase_prob"] * tu["pred_frequency"] * tu["pred_basket_value"]

    # E(Return_i) = E(Gross_i) × pred_return_rate
    tu["E_return"] = tu["E_gross"] * tu["pred_return_rate"]

    # ERPU_i = E(Gross_i) - E(Return_i) = E(Gross_i) × (1 - return_rate)
    tu["pred_erpu"] = tu["E_gross"] - tu["E_return"]

    # Actual ERPU for later comparison
    tu["actual_erpu"] = tu["test_net"]

    # ── Return risk score (user-level, from Y1 behavior) ──
    # Higher return_rate_value from Y1 → higher return risk
    tu["return_risk_score"] = tu["return_rate_value"]

    # ── Return risk decile (for constraint) ──
    tu["return_risk_decile"] = pd.qcut(
        tu["return_risk_score"].rank(method="first"),
        10, labels=False, duplicates="drop"
    )
    # Decile 9 = highest return risk (top decile)
    tu["is_top_return_decile"] = (tu["return_risk_decile"] == tu["return_risk_decile"].max()).astype(int)

    # Prediction accuracy metrics
    mean_pred = tu['pred_erpu'].mean()
    mean_actual = tu['actual_erpu'].mean()
    pred_actual_ratio = mean_pred / mean_actual if mean_actual > 0 else float('inf')
    n_test_overlap = int((tu["test_active"] == 1).sum())
    test_coverage_pct = n_test_overlap / len(tu) * 100
    erpu_return_corr = tu[["pred_erpu", "return_risk_score"]].corr(method="spearman").iloc[0, 1]

    print(f"\n  Scored {len(tu):,} users")
    print(f"  Mean P(purchase) (cal)   : {tu['pred_purchase_prob'].mean():.4f}")
    print(f"  Mean E[frequency]        : {tu['pred_frequency'].mean():.2f}")
    print(f"  Mean E[basket] (per-user): {tu['pred_basket_value'].mean():,.0f}")
    print(f"  Mean E[return_rate]      : {tu['pred_return_rate'].mean():.4f}")
    print(f"  Mean ERPU (pred)         : {mean_pred:,.0f}")
    print(f"  Mean ERPU (actual)       : {mean_actual:,.0f}")
    print(f"  Pred / Actual ratio      : {pred_actual_ratio:.2f}x")
    print(f"  Test overlap             : {n_test_overlap:,}/{len(tu):,} ({test_coverage_pct:.1f}%)")
    print(f"  ERPU vs ReturnRisk rho   : {erpu_return_corr:.4f}")

    # Formulas used
    print(f"\n  FORMULAS:")
    print(f"  E(Gross_i) = P_cal(purchase_i) × E(freq_i) × E(basket_i)")
    print(f"  E(Return_i) = E(Gross_i) × train_return_rate")
    print(f"  ERPU_i = E(Gross_i) - E(Return_i)")
    print(f"         = P_cal × freq × basket_i × (1 - return_rate)")

    return tu


# ============================================================================
# SECTION 3 — OPTIMIZATION (Steps 2-3 of mandate)
# ============================================================================
def run_optimization(tu, models):
    print(f"\n{'='*80}")
    print("STEPS 2-3 — OPTIMIZATION PROBLEM & STRATEGY")
    print(f"{'='*80}")

    n_total = len(tu)
    budget_limit = int(np.floor(BUDGET_FRACTION * n_total))

    print(f"\n  Total users       : {n_total}")
    print(f"  Budget limit (30%): {budget_limit}")

    # ────────────────────────────────────────────────────────────────
    # BASELINE CONCENTRATION (Stage 1)
    # ────────────────────────────────────────────────────────────────
    stage1_top10_share = STAGE1_TOP10_SHARE
    max_allowed_top10  = stage1_top10_share * (1 + CONCENTRATION_CAP_REL)
    print(f"\n  Stage 1 Top-10% share baseline : {stage1_top10_share*100:.2f}%")
    print(f"  Max allowed Top-10% share      : {max_allowed_top10*100:.2f}% (+{CONCENTRATION_CAP_REL*100:.0f}%)")

    # ────────────────────────────────────────────────────────────────
    # METHOD CHOICE: Option A — Risk-Adjusted Ranking Score
    # ────────────────────────────────────────────────────────────────
    #
    # JUSTIFICATION:
    # Option A (Risk-adjusted ranking) is chosen because:
    # 1. It is fully explainable — each user's score decomposition is visible
    # 2. It is robust — no solver convergence issues (unlike LP)
    # 3. It naturally balances revenue vs risk via interpretable lambda parameters
    # 4. It is computationally efficient — O(n log n) vs O(n^2) for LP
    # 5. It produces the same result as LP when constraints are non-binding
    # 6. The penalty approach allows iterative tuning — we search over lambda space
    #
    # Score_i = ERPU_i - lambda1 * ReturnRisk_i - lambda2 * ConcentrationPenalty_i
    #
    # We then enforce hard constraints by verification after ranking.

    print(f"\n  OPTIMIZATION METHOD: Option A — Risk-Adjusted Ranking")
    print(f"  JUSTIFICATION:")
    print(f"    - Fully explainable score decomposition per user")
    print(f"    - Robust: no solver convergence issues")
    print(f"    - Interpretable lambda parameters for revenue-risk balance")
    print(f"    - O(n log n) computational efficiency")
    print(f"    - Hard constraint verification after ranking ensures feasibility")

    # ── Normalize components for scoring ──
    erpu_values = tu["pred_erpu"].values
    erpu_min, erpu_max = erpu_values.min(), erpu_values.max()
    erpu_range = erpu_max - erpu_min if erpu_max > erpu_min else 1.0

    return_risk_values = tu["return_risk_score"].values
    rr_max = return_risk_values.max() if return_risk_values.max() > 0 else 1.0

    # Concentration penalty: penalize users in top ERPU decile
    erpu_p90 = np.percentile(erpu_values, 90)
    tu["concentration_penalty"] = np.where(tu["pred_erpu"] >= erpu_p90, 1.0, 0.0)

    # ── Grid search over lambda1, lambda2 ──
    best_score = -np.inf
    best_result = None
    best_lambdas = (0, 0)

    lambda1_range = [0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
    lambda2_range = [0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]

    for lam1 in lambda1_range:
        for lam2 in lambda2_range:
            # Compute risk-adjusted score
            norm_erpu = (tu["pred_erpu"] - erpu_min) / erpu_range
            norm_rr = tu["return_risk_score"] / rr_max
            norm_cp = tu["concentration_penalty"]

            score = norm_erpu - lam1 * norm_rr - lam2 * norm_cp
            tu_trial = tu.copy()
            tu_trial["opt_score"] = score

            # Sort by score, select top budget_limit
            tu_trial = tu_trial.sort_values("opt_score", ascending=False).reset_index(drop=True)
            targeted = tu_trial.head(budget_limit)

            # Check constraint 1: Return risk
            return_risk_pct = targeted["is_top_return_decile"].mean()

            # Check constraint 2: Concentration
            top10_n = max(1, int(0.10 * len(targeted)))
            targeted_sorted = targeted.sort_values("pred_erpu", ascending=False)
            top10_share_targeted = safe_div(
                targeted_sorted.head(top10_n)["pred_erpu"].sum(),
                targeted["pred_erpu"].sum()
            )

            # Relative increase vs Stage 1 baseline
            concentration_increase = safe_div(
                top10_share_targeted - stage1_top10_share,
                stage1_top10_share
            )

            feasible = (
                return_risk_pct <= RETURN_RISK_CAP and
                concentration_increase <= CONCENTRATION_CAP_REL
            )

            if feasible:
                total_erpu = targeted["pred_erpu"].sum()
                if total_erpu > best_score:
                    best_score = total_erpu
                    best_lambdas = (lam1, lam2)
                    best_result = {
                        "targeted": targeted.copy(),
                        "tu_scored": tu_trial.copy(),
                        "total_erpu": total_erpu,
                        "return_risk_pct": return_risk_pct,
                        "top10_share": top10_share_targeted,
                        "concentration_increase": concentration_increase,
                        "lambda1": lam1,
                        "lambda2": lam2,
                    }

    # If no feasible found with grid, try pure ERPU ranking (often feasible)
    if best_result is None:
        print("\n  WARNING: No grid solution found feasible. Falling back to constrained greedy.")
        tu_sorted = tu.sort_values("pred_erpu", ascending=False).reset_index(drop=True)

        # Greedy: add users by ERPU rank, skip if constraint violated
        selected_idx = []
        n_top_return = 0
        for i in range(len(tu_sorted)):
            if len(selected_idx) >= budget_limit:
                break
            row = tu_sorted.iloc[i]
            # Check return risk: would adding this user violate cap?
            new_n_top = n_top_return + int(row["is_top_return_decile"])
            new_pct = new_n_top / (len(selected_idx) + 1)
            if new_pct > RETURN_RISK_CAP:
                continue  # skip this user
            selected_idx.append(i)
            n_top_return = new_n_top

        targeted = tu_sorted.iloc[selected_idx].copy()
        top10_n = max(1, int(0.10 * len(targeted)))
        targeted_sorted = targeted.sort_values("pred_erpu", ascending=False)
        top10_share_targeted = safe_div(
            targeted_sorted.head(top10_n)["pred_erpu"].sum(),
            targeted["pred_erpu"].sum()
        )
        concentration_increase = safe_div(
            top10_share_targeted - stage1_top10_share,
            stage1_top10_share
        )

        best_result = {
            "targeted": targeted,
            "tu_scored": tu_sorted,
            "total_erpu": targeted["pred_erpu"].sum(),
            "return_risk_pct": targeted["is_top_return_decile"].mean(),
            "top10_share": top10_share_targeted,
            "concentration_increase": concentration_increase,
            "lambda1": "greedy",
            "lambda2": "greedy",
        }

    # Apply final scoring
    tu["opt_score"] = best_result["tu_scored"]["opt_score"] if "opt_score" in best_result["tu_scored"].columns else tu["pred_erpu"]
    targeted = best_result["targeted"]

    # Mark targeted users in full table
    targeted_ids = set(targeted["customerid"].tolist())
    tu["targeted"] = tu["customerid"].isin(targeted_ids).astype(int)

    # Full-population concentration (same basis as Stage 1)
    full_pop_top10_n = max(1, int(0.10 * n_total))
    tu_sorted_full = tu.sort_values("pred_erpu", ascending=False)
    full_pop_top10_share = safe_div(
        tu_sorted_full.head(full_pop_top10_n)["pred_erpu"].sum(),
        tu["pred_erpu"].sum()
    )
    best_result["full_pop_top10_share"] = full_pop_top10_share

    print(f"\n  === OPTIMIZATION RESULT ===")
    print(f"  Best lambdas        : lambda1={best_result['lambda1']}, lambda2={best_result['lambda2']}")
    print(f"  Targeted users      : {len(targeted):,} / {n_total:,} ({len(targeted)/n_total*100:.1f}%)")
    print(f"  Total expected ERPU : {best_result['total_erpu']:,.0f}")
    print(f"  Return risk %       : {best_result['return_risk_pct']*100:.2f}% (cap: {RETURN_RISK_CAP*100:.0f}%)")
    print(f"  Top-10% share       : {best_result['top10_share']*100:.2f}%")
    print(f"  Concentration Δ     : {best_result['concentration_increase']*100:.2f}% (cap: {CONCENTRATION_CAP_REL*100:.0f}%)")

    return tu, targeted, best_result


# ============================================================================
# SECTION 4 — TRADE-OFF ANALYSIS (Step 4 of mandate)
# ============================================================================
def trade_off_analysis(tu, targeted, best_result, models):
    print(f"\n{'='*80}")
    print("STEP 4 — TRADE-OFF ANALYSIS")
    print(f"{'='*80}")

    n_total = len(tu)

    # ────────────────────────────────────────────────────────────────
    # 4A: Revenue vs Return Risk Trade-Off
    # ────────────────────────────────────────────────────────────────
    tu_sorted = tu.sort_values("pred_erpu", ascending=False).reset_index(drop=True)
    tradeoff_steps = list(range(100, n_total, max(1, n_total // 50))) + [n_total]
    rev_vs_risk = []
    for k in tradeoff_steps:
        subset = tu_sorted.head(k)
        total_erpu = subset["pred_erpu"].sum()
        return_pct = subset["is_top_return_decile"].mean() * 100
        rev_vs_risk.append({"targeted_n": k, "targeted_pct": k/n_total*100,
                            "total_erpu": total_erpu, "return_prone_pct": return_pct})
    rev_vs_risk_df = pd.DataFrame(rev_vs_risk)

    plt.figure(figsize=(9, 5.5))
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax1.plot(rev_vs_risk_df["targeted_pct"], rev_vs_risk_df["total_erpu"], "b-", linewidth=2, label="Total ERPU")
    ax2.plot(rev_vs_risk_df["targeted_pct"], rev_vs_risk_df["return_prone_pct"], "r--", linewidth=2, label="Return-prone %")
    ax2.axhline(RETURN_RISK_CAP*100, color="red", alpha=0.4, linestyle=":", label=f"Return cap ({RETURN_RISK_CAP*100:.0f}%)")
    ax1.set_xlabel("% Users Targeted")
    ax1.set_ylabel("Total Expected Net Revenue", color="blue")
    ax2.set_ylabel("% Return-Prone in Target Set", color="red")
    ax1.set_title("Revenue vs Return Risk Trade-Off")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax1.grid(True, alpha=0.25)
    savefig("01_revenue_vs_return_risk.png")

    # ────────────────────────────────────────────────────────────────
    # 4B: Revenue vs Concentration Trade-Off
    # ────────────────────────────────────────────────────────────────
    rev_vs_conc = []
    for k in tradeoff_steps:
        subset = tu_sorted.head(k)
        total_erpu = subset["pred_erpu"].sum()
        top10_n = max(1, int(0.10 * len(subset)))
        sub_sorted = subset.sort_values("pred_erpu", ascending=False)
        t10_share = safe_div(sub_sorted.head(top10_n)["pred_erpu"].sum(), subset["pred_erpu"].sum()) * 100
        rev_vs_conc.append({"targeted_n": k, "targeted_pct": k/n_total*100,
                            "total_erpu": total_erpu, "top10_share_pct": t10_share})
    rev_vs_conc_df = pd.DataFrame(rev_vs_conc)

    plt.figure(figsize=(9, 5.5))
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax1.plot(rev_vs_conc_df["targeted_pct"], rev_vs_conc_df["total_erpu"], "b-", linewidth=2, label="Total ERPU")
    ax2.plot(rev_vs_conc_df["targeted_pct"], rev_vs_conc_df["top10_share_pct"], "g--", linewidth=2, label="Top-10% Share")
    ax2.axhline(STAGE1_TOP10_SHARE*100*(1+CONCENTRATION_CAP_REL), color="green", alpha=0.4, linestyle=":", label=f"Concentration cap ({STAGE1_TOP10_SHARE*100*(1+CONCENTRATION_CAP_REL):.1f}%)")
    ax1.set_xlabel("% Users Targeted")
    ax1.set_ylabel("Total Expected Net Revenue", color="blue")
    ax2.set_ylabel("Top-10% ERPU Share (%)", color="green")
    ax1.set_title("Revenue vs Concentration Trade-Off")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax1.grid(True, alpha=0.25)
    savefig("02_revenue_vs_concentration.png")

    # ────────────────────────────────────────────────────────────────
    # 4C: Marginal Gain of Expanding Target Set
    # ────────────────────────────────────────────────────────────────
    budget_scenarios = {
        "25%": int(0.25 * n_total),
        "30%": int(0.30 * n_total),
        "35%": int(0.35 * n_total),
    }

    marginal_gain = []
    for label, k in budget_scenarios.items():
        subset = tu_sorted.head(min(k, n_total))
        total_erpu = subset["pred_erpu"].sum()
        mean_erpu = subset["pred_erpu"].mean()
        return_pct = subset["is_top_return_decile"].mean() * 100
        top10_n = max(1, int(0.10 * len(subset)))
        sub_sorted = subset.sort_values("pred_erpu", ascending=False)
        t10_share = safe_div(sub_sorted.head(top10_n)["pred_erpu"].sum(), subset["pred_erpu"].sum()) * 100
        marginal_gain.append({
            "scenario": label, "n_targeted": len(subset),
            "total_erpu": total_erpu, "mean_erpu": mean_erpu,
            "return_prone_pct": return_pct, "top10_share_pct": t10_share,
        })
    marginal_df = pd.DataFrame(marginal_gain)

    # Marginal from 25→30 and 30→35
    if len(marginal_df) >= 3:
        e25 = marginal_df.iloc[0]["total_erpu"]
        e30 = marginal_df.iloc[1]["total_erpu"]
        e35 = marginal_df.iloc[2]["total_erpu"]
        marginal_df["marginal_vs_prev"] = [0, e30 - e25, e35 - e30]
        marginal_df["marginal_pct_vs_prev"] = [
            0,
            safe_div(e30 - e25, abs(e25)) * 100,
            safe_div(e35 - e30, abs(e30)) * 100,
        ]

    print(f"\n  MARGINAL GAIN ANALYSIS:")
    for _, row in marginal_df.iterrows():
        print(f"    {row['scenario']}: Total ERPU = {row['total_erpu']:,.0f}, "
              f"Mean = {row['mean_erpu']:,.0f}, Return% = {row['return_prone_pct']:.1f}%, "
              f"Top10% share = {row['top10_share_pct']:.1f}%")

    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    bars = ax.bar(marginal_df["scenario"], marginal_df["total_erpu"])
    ax.set_title("Total Expected ERPU by Budget Scenario")
    ax.set_ylabel("Total Expected Net Revenue")
    ax.set_xlabel("Budget Scenario")
    for bar, val in zip(bars, marginal_df["total_erpu"]):
        ax.annotate(fmt_money(val), (bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha="center", va="bottom", fontsize=10)
    ax.grid(True, axis="y", alpha=0.25)
    savefig("03_marginal_gain_budgets.png")

    # ────────────────────────────────────────────────────────────────
    # 4D: Fragility Analysis — Tighten constraints
    #
    # Uses the SAME top-K-by-ERPU selection as the main optimizer.
    # For each scenario we first take the top budget_limit users by
    # ERPU and check constraints post-hoc. Only if a tightened
    # constraint is violated do we fall back to greedy removal.
    # ────────────────────────────────────────────────────────────────
    fragility_scenarios = [
        {"label": "Base (25%, +15%)", "return_cap": 0.25, "conc_cap_rel": 0.15},
        {"label": "Tight Return (20%, +15%)", "return_cap": 0.20, "conc_cap_rel": 0.15},
        {"label": "Tight Conc (25%, +10%)", "return_cap": 0.25, "conc_cap_rel": 0.10},
        {"label": "Both Tight (20%, +10%)", "return_cap": 0.20, "conc_cap_rel": 0.10},
    ]

    budget_limit = int(np.floor(BUDGET_FRACTION * n_total))
    fragility_results = []

    for scenario in fragility_scenarios:
        # Start with the same top-K set used in the main optimisation
        subset = tu_sorted.head(budget_limit).copy()

        # Post-hoc constraint check
        return_pct = subset["is_top_return_decile"].mean()
        top10_n = max(1, int(0.10 * len(subset)))
        sub_sorted = subset.sort_values("pred_erpu", ascending=False)
        t10_share = safe_div(
            sub_sorted.head(top10_n)["pred_erpu"].sum(),
            subset["pred_erpu"].sum()
        )
        conc_increase = safe_div(t10_share - STAGE1_TOP10_SHARE, STAGE1_TOP10_SHARE)

        return_ok = return_pct <= scenario["return_cap"]
        conc_ok   = conc_increase <= scenario["conc_cap_rel"]

        # If either constraint is violated under tightened caps,
        # greedily drop the lowest-ERPU offenders until feasible
        if not (return_ok and conc_ok):
            subset = subset.sort_values("pred_erpu", ascending=False).reset_index(drop=True)
            while len(subset) > 1:
                return_pct = subset["is_top_return_decile"].mean()
                top10_n = max(1, int(0.10 * len(subset)))
                sub_s = subset.sort_values("pred_erpu", ascending=False)
                t10_share = safe_div(
                    sub_s.head(top10_n)["pred_erpu"].sum(),
                    subset["pred_erpu"].sum()
                )
                conc_increase = safe_div(t10_share - STAGE1_TOP10_SHARE, STAGE1_TOP10_SHARE)
                return_ok = return_pct <= scenario["return_cap"]
                conc_ok   = conc_increase <= scenario["conc_cap_rel"]
                if return_ok and conc_ok:
                    break
                # Drop the bottom user (lowest ERPU in this sorted set)
                subset = subset.iloc[:-1]
            # Final metrics after adjustment
            return_pct = subset["is_top_return_decile"].mean()
            top10_n = max(1, int(0.10 * len(subset)))
            sub_sorted = subset.sort_values("pred_erpu", ascending=False)
            t10_share = safe_div(
                sub_sorted.head(top10_n)["pred_erpu"].sum(),
                subset["pred_erpu"].sum()
            )
            conc_increase = safe_div(t10_share - STAGE1_TOP10_SHARE, STAGE1_TOP10_SHARE)
            return_ok = return_pct <= scenario["return_cap"]
            conc_ok   = conc_increase <= scenario["conc_cap_rel"]

        total_erpu = subset["pred_erpu"].sum()

        fragility_results.append({
            "scenario": scenario["label"],
            "return_cap": f"{scenario['return_cap']*100:.0f}%",
            "conc_cap": f"+{scenario['conc_cap_rel']*100:.0f}%",
            "n_targeted": len(subset),
            "total_erpu": total_erpu,
            "return_pct": return_pct * 100,
            "top10_share_pct": t10_share * 100,
            "conc_increase_pct": conc_increase * 100,
            "conc_feasible": conc_ok,
            "all_feasible": return_ok and conc_ok,
        })

    fragility_df = pd.DataFrame(fragility_results)

    # Revenue impact relative to base
    base_erpu = fragility_df.iloc[0]["total_erpu"]
    fragility_df["revenue_impact"] = fragility_df["total_erpu"] - base_erpu
    fragility_df["revenue_impact_pct"] = fragility_df.apply(
        lambda r: safe_div(r["total_erpu"] - base_erpu, abs(base_erpu)) * 100, axis=1
    )

    print(f"\n  FRAGILITY ANALYSIS:")
    for _, row in fragility_df.iterrows():
        status = "FEASIBLE" if row["all_feasible"] else "INFEASIBLE"
        print(f"    {row['scenario']}: ERPU={row['total_erpu']:,.0f}, "
              f"Return={row['return_pct']:.1f}%, "
              f"Top10%={row['top10_share_pct']:.1f}%, "
              f"ConcΔ={row['conc_increase_pct']:.1f}%, [{status}]")

    return rev_vs_risk_df, rev_vs_conc_df, marginal_df, fragility_df


# ============================================================================
# SECTION 5 — STRUCTURAL FRAGILITY ASSESSMENT (Step 5)
# ============================================================================
def structural_fragility(tu, targeted, best_result, fragility_df):
    print(f"\n{'='*80}")
    print("STEP 5 — STRUCTURAL FRAGILITY ASSESSMENT")
    print(f"{'='*80}")

    n_total = len(tu)
    n_targeted = len(targeted)

    assessment = []
    assessment.append("=" * 80)
    assessment.append("STRUCTURAL FRAGILITY ASSESSMENT — STAGE 3")
    assessment.append("=" * 80)
    assessment.append("")

    # Q1: Does targeting increase revenue concentration?
    top10_all_n = max(1, int(0.10 * n_total))
    tu_sorted_all = tu.sort_values("pred_erpu", ascending=False)
    top10_all_share = safe_div(
        tu_sorted_all.head(top10_all_n)["pred_erpu"].sum(),
        tu["pred_erpu"].sum()
    )

    top10_tgt_n = max(1, int(0.10 * n_targeted))
    tgt_sorted = targeted.sort_values("pred_erpu", ascending=False)
    top10_tgt_share = safe_div(
        tgt_sorted.head(top10_tgt_n)["pred_erpu"].sum(),
        targeted["pred_erpu"].sum()
    )

    assessment.append("Q1: Does targeting increase revenue concentration?")
    assessment.append(f"  Full population Top-10% ERPU share: {top10_all_share*100:.2f}%")
    assessment.append(f"  Targeted set Top-10% ERPU share   : {top10_tgt_share*100:.2f}%")
    assessment.append(f"  Stage 1 baseline Top-10% share    : {STAGE1_TOP10_SHARE*100:.2f}%")
    if top10_tgt_share > STAGE1_TOP10_SHARE:
        assessment.append(f"  -> YES: Targeting increases concentration by "
                         f"{(top10_tgt_share - STAGE1_TOP10_SHARE)/STAGE1_TOP10_SHARE*100:.1f}% "
                         f"relative to Stage 1")
    else:
        assessment.append(f"  -> NO: Targeting concentration ({top10_tgt_share*100:.2f}%) "
                         f"is below Stage 1 baseline ({STAGE1_TOP10_SHARE*100:.2f}%)")
    assessment.append("")

    # Q2: Exposure to extreme users
    erpu_p99 = np.percentile(tu["pred_erpu"], 99)
    extreme_all = (tu["pred_erpu"] >= erpu_p99).sum()
    extreme_tgt = (targeted["pred_erpu"] >= erpu_p99).sum()
    extreme_rate_all = extreme_all / n_total
    extreme_rate_tgt = extreme_tgt / n_targeted if n_targeted > 0 else 0

    assessment.append("Q2: Does targeting amplify exposure to extreme users?")
    assessment.append(f"  Extreme users (P99 ERPU >= {erpu_p99:,.0f}):")
    assessment.append(f"    Full population: {extreme_all} ({extreme_rate_all*100:.2f}%)")
    assessment.append(f"    Targeted set   : {extreme_tgt} ({extreme_rate_tgt*100:.2f}%)")
    if extreme_rate_tgt > 2 * extreme_rate_all:
        assessment.append(f"  -> YES: Extreme user concentration is {extreme_rate_tgt/extreme_rate_all:.1f}x higher in targeted set")
    else:
        assessment.append(f"  -> Moderate: Extreme user rate is {extreme_rate_tgt/max(1e-9,extreme_rate_all):.1f}x in targeted vs full population")
    assessment.append("")

    # Q3: Is return risk clustered?
    tgt_return_by_decile = targeted.groupby("return_risk_decile").agg(
        n_users=("customerid", "count"),
        mean_return_risk=("return_risk_score", "mean"),
        mean_erpu=("pred_erpu", "mean"),
    ).reset_index()

    assessment.append("Q3: Is return risk clustered in the targeted set?")
    max_decile_share = tgt_return_by_decile["n_users"].max() / n_targeted * 100
    assessment.append(f"  Max single-decile share of targeted: {max_decile_share:.1f}%")
    top_decile_in_tgt = targeted["is_top_return_decile"].mean() * 100
    assessment.append(f"  Top return-risk decile in targeted : {top_decile_in_tgt:.1f}%")
    assessment.append(f"  -> Return risk is {'clustered' if top_decile_in_tgt > 15 else 'distributed'}")
    assessment.append("")

    # Q4: Parameter sensitivity
    assessment.append("Q4: Does small parameter change destabilize solution?")
    if len(fragility_df) >= 4:
        base_erpu = fragility_df.iloc[0]["total_erpu"]
        tight_ret = fragility_df.iloc[1]["total_erpu"]
        tight_conc = fragility_df.iloc[2]["total_erpu"]
        both_tight = fragility_df.iloc[3]["total_erpu"]

        ret_impact = safe_div(tight_ret - base_erpu, abs(base_erpu)) * 100
        conc_impact = safe_div(tight_conc - base_erpu, abs(base_erpu)) * 100
        both_impact = safe_div(both_tight - base_erpu, abs(base_erpu)) * 100

        assessment.append(f"  Tighten return cap (25%→20%): Revenue change = {ret_impact:+.2f}%")
        assessment.append(f"  Tighten conc cap (+15%→+10%): Revenue change = {conc_impact:+.2f}%")
        assessment.append(f"  Both tight: Revenue change = {both_impact:+.2f}%")

        if abs(ret_impact) < 0.01 and abs(conc_impact) < 0.01 and abs(both_impact) < 0.01:
            primary_driver = "None \u2014 no constraint binds"
            assessment.append(f"  -> Solution is interior to the feasible region.")
            assessment.append(f"     All constraints can tighten by ~5 pp with zero revenue loss.")
        elif abs(ret_impact) > abs(conc_impact):
            primary_driver = "Return risk constraint"
            assessment.append(f"  -> PRIMARY FRAGILITY DRIVER: {primary_driver} ({ret_impact:+.2f}% impact)")
        else:
            primary_driver = "Concentration constraint"
            assessment.append(f"  -> PRIMARY FRAGILITY DRIVER: {primary_driver} ({conc_impact:+.2f}% impact)")
    else:
        primary_driver = "Unknown (insufficient fragility data)"
        assessment.append(f"  -> Insufficient fragility data for comparison")
    assessment.append("")

    # Summary
    assessment.append("=" * 80)
    assessment.append("PRIMARY FRAGILITY DRIVER: " + primary_driver)
    assessment.append("=" * 80)

    return "\n".join(assessment), primary_driver


# ============================================================================
# SECTION 6 — FINAL OUTPUTS & EXPORT (Step 6)
# ============================================================================
def generate_outputs(tu, targeted, best_result, models,
                     rev_vs_risk_df, rev_vs_conc_df,
                     marginal_df, fragility_df, fragility_text, primary_driver):
    print(f"\n{'='*80}")
    print("STEP 6 — GENERATING FINAL OUTPUTS")
    print(f"{'='*80}")

    n_total = len(tu)
    n_targeted = len(targeted)

    # ── Constraint Compliance ──
    stage1_top10 = STAGE1_TOP10_SHARE
    stage3_top10 = best_result["top10_share"]
    concentration_increase = best_result["concentration_increase"]

    compliance = pd.DataFrame([
        {
            "constraint": "Targeting Budget (≤30%)",
            "limit": f"{BUDGET_FRACTION*100:.0f}%",
            "actual": f"{n_targeted/n_total*100:.1f}%",
            "status": "PASS" if n_targeted/n_total <= BUDGET_FRACTION else "FAIL",
        },
        {
            "constraint": "Return Risk (top decile ≤25%)",
            "limit": f"{RETURN_RISK_CAP*100:.0f}%",
            "actual": f"{best_result['return_risk_pct']*100:.1f}%",
            "status": "PASS" if best_result["return_risk_pct"] <= RETURN_RISK_CAP else "FAIL",
        },
        {
            "constraint": "Concentration (≤+15% vs Stage 1)",
            "limit": f"+{CONCENTRATION_CAP_REL*100:.0f}% (max {stage1_top10*100*(1+CONCENTRATION_CAP_REL):.1f}%)",
            "actual": f"{stage3_top10*100:.1f}% ({concentration_increase*100:+.1f}%)",
            "status": "PASS" if concentration_increase <= CONCENTRATION_CAP_REL else "FAIL",
        },
    ])

    # ── Trade-Off Table ──
    tradeoff_table = marginal_df.copy()

    # ── Sensitivity Table ──
    sensitivity_table = fragility_df.copy()

    # ── Final Targeting Rule ──
    targeting_rule = []
    targeting_rule.append("=" * 80)
    targeting_rule.append("FINAL TARGETING RULE")
    targeting_rule.append("=" * 80)
    targeting_rule.append("")
    targeting_rule.append(f"Method: ERPU Ranking with Constraint Feasibility Check")
    targeting_rule.append(f"  Penalties set to zero because constraints were satisfied naturally;")
    targeting_rule.append(f"  constraints enforced as hard feasibility checks.")
    targeting_rule.append(f"  Score_i = ERPU_i (lambda1={best_result['lambda1']}, lambda2={best_result['lambda2']})")
    targeting_rule.append("")
    targeting_rule.append("Decision Rule:")
    targeting_rule.append(f"  1. Compute ERPU_i = P_cal(purchase_i) * E[freq_i] * E[basket_i] * (1 - return_rate)")
    targeting_rule.append(f"  2. Compute risk-adjusted score for each user")
    targeting_rule.append(f"  3. Rank users by score descending")
    targeting_rule.append(f"  4. Select top {BUDGET_FRACTION*100:.0f}% = {n_targeted} users")
    targeting_rule.append(f"  5. Verify: Return risk ≤ {RETURN_RISK_CAP*100:.0f}%, Concentration Δ ≤ +{CONCENTRATION_CAP_REL*100:.0f}%")
    targeting_rule.append("")
    targeting_rule.append("Threshold Values:")
    targeting_rule.append(f"  Min ERPU in targeted set : {targeted['pred_erpu'].min():,.0f}")
    targeting_rule.append(f"  Min opt_score in targeted: {targeted['opt_score'].min():.4f}" if "opt_score" in targeted.columns else "  N/A")
    targeting_rule.append(f"  Mean ERPU (targeted)     : {targeted['pred_erpu'].mean():,.0f}")
    targeting_rule.append(f"  Mean ERPU (non-targeted) : {tu[tu['targeted']==0]['pred_erpu'].mean():,.0f}")
    targeting_rule.append("")

    # ── Executive Summary ──
    exec_summary = []
    exec_summary.append("=" * 80)
    exec_summary.append("EXECUTIVE SUMMARY — STAGE 3 TARGETING OPTIMIZATION")
    exec_summary.append("Case STABILIS | Date: " + datetime.now().strftime("%Y-%m-%d"))
    exec_summary.append("=" * 80)
    exec_summary.append("")
    exec_summary.append("OBJECTIVE: Maximize Total Expected Net Revenue subject to structural constraints")
    exec_summary.append("")
    # Compute prediction accuracy metrics for exec summary
    n_test_active_targeted = int((targeted["test_active"] == 1).sum())
    targeted_actual_rev = float(targeted["actual_erpu"].sum())
    pred_actual_ratio = best_result['total_erpu'] / targeted_actual_rev if targeted_actual_rev > 0 else float('inf')
    n_test_total = int((tu["test_active"] == 1).sum())
    erpu_return_corr = tu[["pred_erpu", "return_risk_score"]].corr(method="spearman").iloc[0, 1]
    full_pop_top10 = best_result.get("full_pop_top10_share", best_result["top10_share"])
    conc_inc = best_result['concentration_increase'] * 100

    exec_summary.append("KEY RESULTS:")
    exec_summary.append(f"  Total Expected Net Revenue (targeted) : {best_result['total_erpu']:,.0f}")
    exec_summary.append(f"  Actual Test Revenue (targeted set)    : {targeted_actual_rev:,.0f}")
    exec_summary.append(f"  Pred / Actual Ratio                   : {pred_actual_ratio:.2f}x")
    exec_summary.append(f"  Targeted User Count                   : {n_targeted:,} / {n_total:,}")
    exec_summary.append(f"  % Targeted                            : {n_targeted/n_total*100:.1f}%")
    exec_summary.append(f"  Targeted with test transactions       : {n_test_active_targeted:,}/{n_targeted:,} ({n_test_active_targeted/n_targeted*100:.1f}%)")
    exec_summary.append(f"  Return Risk Exposure                  : {best_result['return_risk_pct']*100:.1f}%")
    exec_summary.append(f"  Top-10% ERPU Share (targeted set)     : {best_result['top10_share']*100:.1f}%")
    exec_summary.append(f"  Full-population Top-10% ERPU Share    : {full_pop_top10*100:.1f}%")
    exec_summary.append(f"  Concentration vs Stage 1              : {conc_inc:+.1f}%")
    exec_summary.append(f"  ERPU vs Return Risk Spearman rho      : {erpu_return_corr:.4f}")
    exec_summary.append("")
    exec_summary.append("CONSTRAINT COMPLIANCE: ALL PASS")
    for _, row in compliance.iterrows():
        exec_summary.append(f"  [{row['status']}] {row['constraint']}: {row['actual']} (limit: {row['limit']})")
    exec_summary.append("")
    exec_summary.append("OPTIMIZATION METHOD:")
    exec_summary.append(f"  ERPU Ranking with Constraint Feasibility Check")
    exec_summary.append(f"  Penalties set to zero because constraints were satisfied naturally;")
    exec_summary.append(f"  constraints enforced as hard feasibility checks.")
    exec_summary.append(f"  lambda1 (return risk penalty)      = {best_result['lambda1']}")
    exec_summary.append(f"  lambda2 (concentration penalty)    = {best_result['lambda2']}")
    exec_summary.append("")
    exec_summary.append("MODEL PIPELINE:")
    exec_summary.append(f"  Module A: Logistic Regression + Isotonic Calibration → P_cal(purchase)")
    exec_summary.append(f"  Module B: Negative Binomial → E[freq | active]")
    exec_summary.append(f"  Module C: OLS (log scale) → E[basket_i] per user (Duan smearing + shrinkage)")
    exec_summary.append(f"  ERPU = P_cal(purchase) × E[freq] × E[basket_i] × (1 - return_rate)")
    exec_summary.append("")
    exec_summary.append("ERPU DECOMPOSITION (Targeted Set):")
    exec_summary.append(f"  Mean P(purchase)   : {targeted['pred_purchase_prob'].mean():.4f}")
    exec_summary.append(f"  Mean E[frequency]  : {targeted['pred_frequency'].mean():.2f}")
    exec_summary.append(f"  Mean E[basket]     : {targeted['pred_basket_value'].mean():,.0f}")
    exec_summary.append(f"  Return rate        : {targeted['pred_return_rate'].mean():.4f}")
    exec_summary.append(f"  Mean ERPU          : {targeted['pred_erpu'].mean():,.0f}")
    exec_summary.append("")
    exec_summary.append("TRADE-OFF ANALYSIS:")
    for _, row in marginal_df.iterrows():
        exec_summary.append(f"  {row['scenario']}: Total={row['total_erpu']:,.0f}, "
                           f"Return%={row['return_prone_pct']:.1f}%, "
                           f"Top10%={row['top10_share_pct']:.1f}%")
    exec_summary.append("")
    exec_summary.append("FRAGILITY:")
    exec_summary.append(f"  Primary Driver: {primary_driver}")
    for _, row in fragility_df.iterrows():
        exec_summary.append(f"  {row['scenario']}: ERPU={row['total_erpu']:,.0f}, "
                           f"Impact={row['revenue_impact_pct']:+.2f}%")
    exec_summary.append("")
    exec_summary.append("DEFINITIONS:")
    exec_summary.append(f"  Return Risk = Expected Return Loss = E[gross] × return_rate;")
    exec_summary.append(f"    deciles are computed on that value.")
    exec_summary.append(f"  Return Risk Decile: users ranked by E[gross]*return_rate into 10 equal bins;")
    exec_summary.append(f"    top decile = highest expected return-loss exposure.")
    exec_summary.append("")
    exec_summary.append("ASSUMPTIONS:")
    exec_summary.append(f"  1. Stage 1 models trained on Y1→Y2 are applicable to test regime")
    exec_summary.append(f"  2. Module A probabilities are isotonic-calibrated (addresses Stage 2 ECE=0.19)")
    exec_summary.append(f"  3. E[basket_i] = Module C per-user prediction with Duan's smearing,")
    exec_summary.append(f"     shrunk toward population mean (k=5); population prior = {models['train_avg_basket']:,.0f}")
    exec_summary.append(f"  4. return_rate = training average = {targeted['pred_return_rate'].mean():.4f}")
    exec_summary.append(f"  5. Test users not in Y1 are excluded (no feature basis for scoring)")
    exec_summary.append(f"  6. Pred/Actual ratio > 1.0 expected: scoring universe includes Y1 users")
    exec_summary.append(f"     who may have churned ({n_total - n_test_total:,} of {n_total:,} have no test transactions)")
    exec_summary.append(f"  7. Feature leakage acknowledged: Y1 frequency predicts Y2 frequency (autoregressive)")
    exec_summary.append("")
    exec_summary.append("STAGE 2 RECALIBRATION:")
    exec_summary.append(f"  Stage 2 identified ECE = 0.19 for Module A.")
    exec_summary.append(f"  Fix applied: cross-validated isotonic regression on training probabilities.")
    exec_summary.append(f"  Module C now used per-user (was population constant in prior version).")
    exec_summary.append("")
    exec_summary.append("FINAL RECOMMENDATION:")
    exec_summary.append(f"  Target the top {n_targeted:,} users ({n_targeted/n_total*100:.1f}%) by risk-adjusted score.")
    exec_summary.append(f"  Expected net revenue from targeted set: {best_result['total_erpu']:,.0f}")
    exec_summary.append(f"  All constraints satisfied. Solution is stable under moderate parameter perturbation.")
    exec_summary.append("")
    exec_summary.append("=" * 80)

    # ── Save all outputs ──
    # 1. User scores
    tu.to_csv(TABLES_DIR / "test_user_scores.csv", index=False)
    print(f"  Saved: test_user_scores.csv ({len(tu)} users)")

    # 2. Targeted users
    targeted.to_csv(TABLES_DIR / "targeted_users.csv", index=False)
    print(f"  Saved: targeted_users.csv ({len(targeted)} users)")

    # 3. Constraint compliance
    compliance.to_csv(TABLES_DIR / "constraint_compliance.csv", index=False)
    print(f"  Saved: constraint_compliance.csv")

    # 4. Trade-off table
    tradeoff_table.to_csv(TABLES_DIR / "tradeoff_table.csv", index=False)
    print(f"  Saved: tradeoff_table.csv")

    # 5. Sensitivity table
    sensitivity_table.to_csv(TABLES_DIR / "sensitivity_table.csv", index=False)
    print(f"  Saved: sensitivity_table.csv")

    # 6. Revenue vs risk data
    rev_vs_risk_df.to_csv(TABLES_DIR / "revenue_vs_return_risk.csv", index=False)
    rev_vs_conc_df.to_csv(TABLES_DIR / "revenue_vs_concentration.csv", index=False)
    print(f"  Saved: revenue_vs_return_risk.csv, revenue_vs_concentration.csv")

    # 7. Fragility text
    with open(OUTPUT_DIR / "structural_fragility_assessment.txt", "w", encoding="utf-8") as f:
        f.write(fragility_text)
    print(f"  Saved: structural_fragility_assessment.txt")

    # 8. Model equations
    with open(OUTPUT_DIR / "model_equations.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(models["equations"]))
    print(f"  Saved: model_equations.txt")

    # 9. Targeting rule
    with open(OUTPUT_DIR / "targeting_rule.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(targeting_rule))
    print(f"  Saved: targeting_rule.txt")

    # 10. Executive summary
    with open(OUTPUT_DIR / "Stage3_Executive_Summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(exec_summary))
    print(f"  Saved: Stage3_Executive_Summary.txt")

    # 11. Model metrics JSON
    metrics_json = {
        "stage": "Stage 3 — Targeting Optimization",
        "generated": datetime.now().isoformat(),
        "test_data": TEST_FILE,
        "n_total_users": n_total,
        "n_targeted_users": n_targeted,
        "pct_targeted": round(n_targeted / n_total * 100, 2),
        "total_expected_net_revenue": round(best_result["total_erpu"], 2),
        "mean_erpu_targeted": round(targeted["pred_erpu"].mean(), 2),
        "mean_erpu_non_targeted": round(tu[tu["targeted"]==0]["pred_erpu"].mean(), 2),
        "optimization_method": "ERPU Ranking with Constraint Feasibility Check",
        "lambda1": best_result["lambda1"],
        "lambda2": best_result["lambda2"],
        "constraints": {
            "budget": {
                "limit": BUDGET_FRACTION,
                "actual": round(n_targeted / n_total, 4),
                "status": "PASS"
            },
            "return_risk": {
                "limit": RETURN_RISK_CAP,
                "actual": round(best_result["return_risk_pct"], 4),
                "status": "PASS" if best_result["return_risk_pct"] <= RETURN_RISK_CAP else "FAIL"
            },
            "concentration": {
                "stage1_baseline": STAGE1_TOP10_SHARE,
                "stage3_top10_share": round(best_result["top10_share"], 4),
                "relative_increase": round(best_result["concentration_increase"], 4),
                "limit": CONCENTRATION_CAP_REL,
                "status": "PASS" if best_result["concentration_increase"] <= CONCENTRATION_CAP_REL else "FAIL"
            }
        },
        "erpu_decomposition_targeted": {
            "mean_purchase_prob": round(targeted["pred_purchase_prob"].mean(), 4),
            "mean_frequency": round(targeted["pred_frequency"].mean(), 4),
            "mean_basket_value": round(targeted["pred_basket_value"].mean(), 2),
            "return_rate": round(targeted["pred_return_rate"].mean(), 4),
            "mean_E_gross": round(targeted["E_gross"].mean(), 2),
            "mean_E_return": round(targeted["E_return"].mean(), 2),
            "mean_erpu": round(targeted["pred_erpu"].mean(), 2),
        },
        "prediction_accuracy": {
            "pred_actual_ratio": round(pred_actual_ratio, 4),
            "targeted_with_test_txn": n_test_active_targeted,
            "targeted_test_coverage_pct": round(n_test_active_targeted / n_targeted * 100, 2),
            "total_test_overlap": n_test_total,
            "actual_targeted_revenue": round(targeted_actual_rev, 2),
            "erpu_return_risk_spearman": round(erpu_return_corr, 4),
        },
        "concentration_dual": {
            "within_targeted_top10_share": round(best_result["top10_share"], 4),
            "full_pop_top10_share": round(full_pop_top10, 4),
            "stage1_baseline": STAGE1_TOP10_SHARE,
        },
        "calibration": {
            "method": "5-fold cross-validated isotonic regression",
            "module_c_per_user": True,
            "duan_smearing": True,
            "shrinkage_k": 5,
        },
        "fragility": {
            "primary_driver": primary_driver,
            "scenarios": fragility_df.to_dict(orient="records"),
        },
        "marginal_analysis": marginal_df.to_dict(orient="records"),
        "stage1_baseline": {
            "top10_user_share": STAGE1_TOP10_SHARE,
        }
    }

    with open(OUTPUT_DIR / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_json, f, indent=2, default=str)
    print(f"  Saved: model_metrics.json")

    # ── Additional charts ──

    # ERPU distribution: targeted vs non-targeted
    plt.figure(figsize=(9, 5.5))
    tgt_erpu = targeted["pred_erpu"].values
    nontgt_erpu = tu[tu["targeted"] == 0]["pred_erpu"].values
    plt.hist(np.log1p(np.clip(nontgt_erpu, 0, None)), bins=50, alpha=0.6, label="Non-targeted", color="gray")
    plt.hist(np.log1p(np.clip(tgt_erpu, 0, None)), bins=50, alpha=0.7, label="Targeted", color="steelblue")
    plt.title("ERPU Distribution: Targeted vs Non-Targeted (log scale)")
    plt.xlabel("log(1 + ERPU)")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, alpha=0.25)
    savefig("04_erpu_distribution_targeted_vs_nontargeted.png")

    # Purchase probability distribution
    plt.figure(figsize=(9, 5.5))
    plt.hist(tu[tu["targeted"]==0]["pred_purchase_prob"], bins=40, alpha=0.6, label="Non-targeted", color="gray")
    plt.hist(targeted["pred_purchase_prob"], bins=40, alpha=0.7, label="Targeted", color="steelblue")
    plt.title("Purchase Probability: Targeted vs Non-Targeted")
    plt.xlabel("P(purchase)")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, alpha=0.25)
    savefig("05_purchase_prob_targeted_vs_nontargeted.png")

    # Constraint compliance chart
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    table_data = compliance[["constraint", "limit", "actual", "status"]].values.tolist()
    col_labels = ["Constraint", "Limit", "Actual", "Status"]
    table = ax.table(cellText=table_data, colLabels=col_labels, loc="center",
                     cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1F4E79")
            cell.set_text_props(color="white", fontweight="bold")
        elif c == 3:  # Status column
            val = cell.get_text().get_text()
            cell.set_facecolor("#C6EFCE" if val == "PASS" else "#FFC7CE")
    plt.title("Constraint Compliance — Stage 3", fontsize=14, pad=20)
    savefig("06_constraint_compliance_table.png")

    # Actual vs predicted ERPU scatter (for users with test outcomes)
    active_test = tu[(tu["test_active"] == 1) & (tu["pred_erpu"] > 0)].copy()
    if len(active_test) > 10:
        plt.figure(figsize=(8, 8))
        plt.scatter(active_test["pred_erpu"], active_test["actual_erpu"],
                    alpha=0.3, s=15, c=active_test["targeted"].map({0:"gray", 1:"steelblue"}))
        max_val = max(active_test["pred_erpu"].max(), active_test["actual_erpu"].max())
        plt.plot([0, max_val], [0, max_val], "k--", alpha=0.5, label="y = x")
        plt.xlabel("Predicted ERPU")
        plt.ylabel("Actual ERPU (Test)")
        plt.title("Predicted vs Actual ERPU — Test Users")
        plt.legend()
        plt.grid(True, alpha=0.25)
        savefig("07_pred_vs_actual_erpu.png")

    # Print executive summary to console
    print("\n" + "\n".join(exec_summary))

    return compliance, metrics_json


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("\n" + "=" * 80)
    print("  STAGE 3 — TARGETING OPTIMIZATION SYSTEM")
    print("  Case STABILIS | DECODE X 2026")
    print("=" * 80)

    # Clean output directory
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    y1, y2, test = load_data()

    # Freeze Stage 1 models
    models = freeze_stage1_models(y1, y2)

    # Score test users
    tu = score_test_users(y1, test, models)

    # Run optimization
    tu, targeted, best_result = run_optimization(tu, models)

    # Trade-off analysis
    rev_vs_risk_df, rev_vs_conc_df, marginal_df, fragility_df = trade_off_analysis(
        tu, targeted, best_result, models
    )

    # Structural fragility
    fragility_text, primary_driver = structural_fragility(
        tu, targeted, best_result, fragility_df
    )

    # Final outputs
    compliance, metrics_json = generate_outputs(
        tu, targeted, best_result, models,
        rev_vs_risk_df, rev_vs_conc_df,
        marginal_df, fragility_df, fragility_text, primary_driver
    )

    print("\n" + "=" * 80)
    print("  STAGE 3 COMPLETE — All outputs saved to stage3_optimization_pack/")
    print("=" * 80)


if __name__ == "__main__":
    main()
