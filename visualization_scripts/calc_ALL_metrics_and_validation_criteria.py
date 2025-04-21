# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 14:10:00 2025

This script reads a CSV file with SOC stock data and computes the following
metrics for each group (based on user-defined grouping_fields), for each
(depth, model) combination:
  1) R2, MedAE, RMSE, NMedAE, NRMSE, NRMSEsd
  2) A one-sample t-test on errors (obs - model) => "Err not stat diff from 0"
  3) Coverage check: percentage of observed validation points that lie within
     the corresponding 90% prediction interval (using the model's uncertainty).
     (OLM and SA models are skipped for coverage)
  4) Bias (mean(pred) - mean(obs)) and margin of error (MoE) from observed data,
     using a 90% CI for GS bias criterion.
  5) Additional numeric fields:
       "Std Dev (obs)", "Coverage (%)", "Bias", "MoE"
  6) Pass/Fail fields:
       "R2>0", "RMSE<StdDevObs", "Err not stat diff from 0",
       "Coverage>90%" (placeholder approach), "Bias<MoE"
Additionally, for each group (Depth, Model), distribution plots are
displayed for observed, predicted, and error distributions with Shapiro–Wilk tests.
@author: André
"""

# ----------------------- Import Libraries -----------------------
import os             # For file/directory operations
import pandas as pd   # For CSV file handling
import numpy as np    # For numerical operations
import matplotlib.pyplot as plt  # For plotting histograms
from sklearn.metrics import r2_score, mean_squared_error  # For model metrics
from scipy import stats  # For one-sample t-test and Shapiro–Wilk test
import time           # For timing script execution

# ----------------------------- User Config -------------------------------
# Example: ["ADMIN","GPW", "site", "ADMIN", "LDSF_quant_SOCstock"] or [] for no grouping
grouping_fields = []  ###  <-- UPDATE  ###

csv_in = (
    r"D:\TerraCarbon\2025_projects\SoilOrganicCarbon\QGIS\data\ICRAF\ldsf_soc_stocks_updated.csv"
)
out_dir = (
    r"D:\TerraCarbon\2025_projects\SoilOrganicCarbon\QGIS\data\ICRAF\metrics"
)
os.makedirs(out_dir, exist_ok=True)

# Observed and modeled column names
obs_20 = "predsoc_stock_20cm"          # Observed SOC stock (t/ha)
mod_20 = "isda SOC stock 20cm"         # isda predicted SOC stock at 20cm

obs_30 = "predsoc_stock_30cm"          # Observed SOC stock for 30cm groups
mod_30_dict = {
    "soilGrids": "soilGrids SOC stock 30cm",  # has uncertainty field
    "OLM2017":  "OLM 2017 SOC stock 30cm",       # no uncertainty
    "SA":       "SA SOC stock 30cm"              # no uncertainty
}

# Uncertainty fields (t/ha)
isda_unc_field = "iSDA stock uncertainty"      # for isda 20cm
soilgrids_unc_field = "soilGrids uncertainty"  # for soilGrids 30cm

# For the VCS "lack of bias" test (one-sample t-test)
alpha_vcs = 0.05  # significance level for one-sample t-test
# For the GS "Bias < half-width of 90% CI" test
alpha_gs_bias = 0.10  # for 90% CI used in GS bias margin-of-error check

shapiro_alpha = 0.05  # significance level for Shapiro–Wilk normality checks

# -------------------- Start Timing ------------------------------
start_time = time.time()  # Record script start time

# ----------------------- Plot Distributions Function -----------------------
def plot_distributions(obs_vals, mod_vals, depth, model, group_label):
    """
    Displays histograms for observed, predicted, and error distributions.
    Runs Shapiro–Wilk test on each distribution and prints "Normal? Yes/No".
    """
    errors = obs_vals - mod_vals
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(f"Dist Plots: {group_label} - Depth:{depth}, Model:{model}")

    # Observed values histogram
    axes[0].hist(obs_vals, bins=30, color="lightblue", edgecolor="black")
    axes[0].set_title("Observed Dist")
    if len(obs_vals) >= 3:
        w_stat, w_p = stats.shapiro(obs_vals)
        normal_flag = "Yes" if w_p > shapiro_alpha else "No"
    else:
        normal_flag = "N/A"
    axes[0].text(0.5, 0.9, f"Normal? {normal_flag}", ha="center",
                 va="top", transform=axes[0].transAxes, color="red", fontsize=10)

    # Predicted values histogram
    axes[1].hist(mod_vals, bins=30, color="lightgreen", edgecolor="black")
    axes[1].set_title("Predicted Dist")
    if len(mod_vals) >= 3:
        w_stat2, w_p2 = stats.shapiro(mod_vals)
        normal_flag2 = "Yes" if w_p2 > shapiro_alpha else "No"
    else:
        normal_flag2 = "N/A"
    axes[1].text(0.5, 0.9, f"Normal? {normal_flag2}", ha="center",
                 va="top", transform=axes[1].transAxes, color="red", fontsize=10)

    # Error values histogram
    axes[2].hist(errors, bins=30, color="lightcoral", edgecolor="black")
    axes[2].set_title("Errors Dist")
    if len(errors) >= 3:
        w_stat3, w_p3 = stats.shapiro(errors)
        normal_flag3 = "Yes" if w_p3 > shapiro_alpha else "No"
    else:
        normal_flag3 = "N/A"
    axes[2].text(0.5, 0.9, f"Normal? {normal_flag3}", ha="center",
                 va="top", transform=axes[2].transAxes, color="red", fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    plt.close(fig)

# ----------------------- Metrics Calculation Function -----------------------
def calc_metrics(dfsub, obs_col, mod_col, unc_col, depth, model, group_label):
    """
    Computes model validation metrics including:
      - R2, MedAE, RMSE, NMedAE, NRMSE, NRMSEsd,
      - Standard deviation of obs, Coverage (%), Bias, MoE.
    Coverage is computed as the percentage of observed values that fall within the
    90% prediction interval: (mod - unc, mod + unc).
    If unc_col is None, coverage is set to NaN.
    Also computes pass/fail labels for various tests.
    """
    d = dfsub.dropna(subset=[obs_col, mod_col])
    if d.empty:
        return None
    obs = d[obs_col].values
    mod = d[mod_col].values

    # Display distribution plots
    plot_distributions(obs, mod, depth, model, group_label)

    errors = obs - mod
    medae = float(np.median(np.abs(errors)))
    rmse = float(np.sqrt(mean_squared_error(obs, mod)))
    r2 = float(r2_score(obs, mod))
    mean_obs = float(np.mean(obs))
    sd_obs = float(np.std(obs, ddof=1))
    nmedae = (medae / mean_obs) if mean_obs != 0 else np.nan
    nrmse = (rmse / mean_obs) if mean_obs != 0 else np.nan
    nrmse_sd = rmse / sd_obs if sd_obs != 0 else np.nan

    # One-sample t-test for bias
    t_stat, p_val = stats.ttest_1samp(errors, 0, nan_policy="omit")
    err_not_diff_label = "Pass" if p_val > alpha_vcs else "Fail"

    # Coverage calculation using uncertainty field if available
    if unc_col is not None and unc_col in d.columns:
        unc_vals = d[unc_col].values
        in_interval = (obs >= (mod - unc_vals)) & (obs <= (mod + unc_vals))
        coverage_pct = (np.sum(in_interval) / len(obs)) * 100
    else:
        coverage_pct = np.nan
    coverage_label = "Pass" if not np.isnan(coverage_pct) and coverage_pct >= 90 else "Fail" if not np.isnan(coverage_pct) else "N/A"

    # Bias and MoE calculation
    bias_val = float(np.mean(mod) - np.mean(obs))
    n = len(obs)
    if n > 1:
        t_crit_90 = stats.t.ppf(1 - alpha_gs_bias/2, n - 1)
        MoE_val = t_crit_90 * sd_obs / np.sqrt(n)
    else:
        MoE_val = np.nan
    bias_label = "Pass" if abs(bias_val) < MoE_val else "Fail"

    r2_label = "Pass" if r2 > 0 else "Fail"
    rmse_stddev_label = "Pass" if rmse < sd_obs else "Fail"

    return {
        "R2": r2,
        "MedAE": medae,
        "RMSE": rmse,
        "NMedAE": nmedae,
        "NRMSE": nrmse,
        "NRMSEsd": nrmse_sd,
        "Std Dev (obs)": sd_obs,
        "Coverage (%)": coverage_pct,
        "Bias": bias_val,
        "MoE": MoE_val,
        "R2>0": r2_label,
        "RMSE<StdDevObs": rmse_stddev_label,
        "Err not stat diff from 0": err_not_diff_label,
        "Coverage>90%": coverage_label,
        "Bias<MoE": bias_label,
    }

# ----------------------- Process Data Function -----------------------
def process_data(dfsub, group_tuple):
    """
    For each group (or entire df if no grouping), compute metrics for:
      - 20cm model: isda (with uncertainty)
      - 30cm models: soilGrids, OLM2017, SA
    For coverage, only isda (20cm) and soilGrids (30cm) are processed;
    OLM2017 and SA are skipped (uncertainty set to None).
    """
    group_label = "-".join([str(g) for g in group_tuple]) if group_tuple else "All"

    # Process 20cm: isda
    res_20 = calc_metrics(
        dfsub, obs_20, mod_20, "iSDA stock uncertainty",
        depth="20cm", model="isda", group_label=group_label
    )
    if res_20:
        row_20 = {"Depth": "20cm", "Model": "isda"}
        for i, f in enumerate(grouping_fields):
            row_20[f] = group_tuple[i] if len(group_tuple) > i else None
        row_20.update(res_20)
        rows.append(row_20)

    # Process 30cm models
    for mk, mc in mod_30_dict.items():
        # Only for soilGrids do we calculate coverage; for OLM2017 and SA, set unc to None
        unc_col = "soilGrids uncertainty" if mk == "soilGrids" else None
        res_30 = calc_metrics(
            dfsub, obs_30, mc, unc_col,
            depth="30cm", model=mk, group_label=group_label
        )
        if res_30:
            row_30 = {"Depth": "30cm", "Model": mk}
            for i, f in enumerate(grouping_fields):
                row_30[f] = group_tuple[i] if len(group_tuple) > i else None
            row_30.update(res_30)
            rows.append(row_30)

# ----------------------- Main Processing -----------------------------
rows = []  # List to store metric rows
df = pd.read_csv(csv_in)  # Read CSV file
df.columns = df.columns.str.strip()  # Remove extra spaces in column names

# Process groups if grouping_fields is not empty
if len(grouping_fields) == 0:
    process_data(df, ())
else:
    grouped = df.groupby(grouping_fields)
    for grp_vals, grp_df in grouped:
        if not isinstance(grp_vals, tuple):
            grp_vals = (grp_vals,)
        process_data(grp_df, grp_vals)

df_out = pd.DataFrame(rows)
grp_suffix = "_".join(grouping_fields) if grouping_fields else "noGrouping"
out_name = f"metrics_{grp_suffix}.csv"
out_path = os.path.join(out_dir, out_name)

# Round numeric columns to 2 decimals for output summary
df_out = df_out.round(2)
df_out.to_csv(out_path, index=False)
print(f"Saved metrics CSV at: {out_path}")
