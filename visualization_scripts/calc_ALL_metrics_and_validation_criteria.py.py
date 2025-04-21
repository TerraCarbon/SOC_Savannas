# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 10:10:24 2025

This script reads a CSV file with SOC stock data and computes the following 
metrics for each group (based on user-defined grouping_fields), for each 
(depth, model) combination:

1) R2, MedAE, RMSE, NMedAE, NRMSE, NRMSEsd
2) A one-sample t-test on errors (obs - model) => "Err not stat diff from 0"
3) Coverage % (approx 90% interval from residual std)
4) Bias (mean(pred) - mean(obs)) and margin of error (MoE) from observed data
5) Additional numeric fields:
   - "Std Dev" (std of the model predictions)
   - "Coverage (%)"
   - "Bias"
   - "MoE"
6) Pass/Fail fields:
   - "R2>0"
   - "RMSE<StdDev"
   - "Err not stat diff from 0"
   - "Coverage>90%"
   - "Bias<MofE"

Additionally, for each group (Depth, Model), we display 3 histograms:
 - Observed distribution
 - Predicted distribution
 - Error distribution (obs - pred)
and run the Shapiro–Wilk test on each distribution, printing whether 
it is normal or not on the plot.

NOTE: Plots are not saved to disk; they are shown via plt.show() and 
closed after each usage.


@author: André
"""

# ------------------------- Imports & Variables -------------------------
#   time for tracking script duration
import time  # For timing

#   os for directory/file manipulation
import os  # For file and directory operations

#   pandas for CSV I/O and data handling
import pandas as pd  # For CSV file reading/writing

#   numpy for numerical computations
import numpy as np  # For numerical operations

#   matplotlib for plotting (histograms, table creation)
import matplotlib.pyplot as plt  # For plotting

#   sklearn for metrics (R², MSE)
from sklearn.metrics import r2_score, mean_squared_error  # For metrics

#   scipy for statistical tests
from scipy import stats  # For t-test, Shapiro–Wilk test


#   Start time tracking
start_time = time.time()

#   --------------- USER VARIABLES ---------------
#   Grouping fields can be changed as needed
# [], "ADMIN", "LDSF_quant_SOCstock", "Vegetation", "GPW", "site"
grouping_fields = []      ###  <-- UPDATE  ###

#   Input CSV file with observed/predicted SOC data
csv_in = r"D:\TerraCarbon\2025_projects\SoilOrganicCarbon\QGIS\data\ICRAF\ldsf_and_DSM_soc_stocks.csv"  

#   Output directory for metrics & table
out_dir = r"D:\TerraCarbon\2025_projects\SoilOrganicCarbon\QGIS\data\ICRAF\metrics" 

#   Observed/Modeled columns for 20 cm
obs_20 = "predsoc_stock_20cm"
mod_20 = "isda SOC stock 20cm"

#   Observed column for 30 cm
obs_30 = "predsoc_stock_30cm"

#   Dict of 30 cm model columns
mod_30_dict = {
    "soilGrids": "soilGrids SOC stock 30cm",
    "OLM2017": "OLM 2017 SOC stock 30cm",
    "SA": "SA SOC stock 30cm"
}

#   Uncertainty fields
isda_unc_field = "iSDA stock uncertainty"
soilgrids_unc_field = "soilGrids uncertainty"

#   Significance levels
alpha_vcs = 0.05     # For VCS bias test
alpha_gs_bias = 0.10 # For GS bias margin-of-error test
shapiro_alpha = 0.05 # For Shapiro–Wilk normality test

#   Make sure output directory exists
os.makedirs(out_dir, exist_ok=True)

# ------------------------- Plot Distributions -------------------------
#   This function plots histograms of observed, predicted, and error.
def plot_distributions(obs_vals, mod_vals, depth, model, group_label):
    #   Compute errors
    errors = obs_vals - mod_vals

    #   Create figure
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    #   Set figure title
    fig.suptitle(f"Dist Plots: {group_label} - Depth:{depth}, Model:{model}")

    #   Plot Observed histogram
    axes[0].hist(obs_vals, bins=30, color="lightblue", edgecolor="black")
    axes[0].set_title("Observed Dist")
    if len(obs_vals) >= 3:
        w_stat, w_p = stats.shapiro(obs_vals)
        normal_flag = "Yes" if w_p > shapiro_alpha else "No"
    else:
        normal_flag = "N/A"
    axes[0].text(
        0.5, 0.9, f"Normal? {normal_flag}", ha="center", va="top",
        transform=axes[0].transAxes, color="red", fontsize=10
    )

    #   Plot Predicted histogram
    axes[1].hist(mod_vals, bins=30, color="lightgreen", edgecolor="black")
    axes[1].set_title("Predicted Dist")
    if len(mod_vals) >= 3:
        w_stat2, w_p2 = stats.shapiro(mod_vals)
        normal_flag2 = "Yes" if w_p2 > shapiro_alpha else "No"
    else:
        normal_flag2 = "N/A"
    axes[1].text(
        0.5, 0.9, f"Normal? {normal_flag2}", ha="center", va="top",
        transform=axes[1].transAxes, color="red", fontsize=10
    )

    #   Plot Errors histogram
    axes[2].hist(errors, bins=30, color="lightcoral", edgecolor="black")
    axes[2].set_title("Errors Dist")
    if len(errors) >= 3:
        w_stat3, w_p3 = stats.shapiro(errors)
        normal_flag3 = "Yes" if w_p3 > shapiro_alpha else "No"
    else:
        normal_flag3 = "N/A"
    axes[2].text(
        0.5, 0.9, f"Normal? {normal_flag3}", ha="center", va="top",
        transform=axes[2].transAxes, color="red", fontsize=10
    )

    #   Improve layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    #   Show figure then close
    plt.show()
    plt.close(fig)

# --------------------- Calculate Metrics ---------------------
#   This function computes the metrics for a subset of data.
def calc_metrics(dfsub, obs_col, mod_col, unc_col, depth, model, group_label):
    #   Drop rows missing obs or mod
    d = dfsub.dropna(subset=[obs_col, mod_col])
    if d.empty:
        return None

    #   Extract arrays
    obs = d[obs_col].values
    mod = d[mod_col].values

    #   Plot distributions
    plot_distributions(obs, mod, depth, model, group_label)

    #   Compute errors
    errors = obs - mod

    #   Basic stats
    medae = float(np.median(np.abs(errors)))
    rmse = float(np.sqrt(mean_squared_error(obs, mod)))
    r2 = float(r2_score(obs, mod))
    mean_obs = float(np.mean(obs))
    sd_obs = float(np.std(obs, ddof=1))

    #   Normalized metrics
    nmedae = (medae / mean_obs) if mean_obs != 0 else np.nan
    nrmse = (rmse / mean_obs) if mean_obs != 0 else np.nan
    nrmse_sd = rmse / sd_obs if sd_obs != 0 else np.nan

    #   One-sample t-test for bias
    t_stat, p_val = stats.ttest_1samp(errors, 0, nan_policy="omit")
    err_not_diff_label = "Pass" if p_val > alpha_vcs else "Fail"

    #   Coverage calculation
    if unc_col is not None and unc_col in d.columns:
        unc_vals = d[unc_col].values
        in_interval = (obs >= (mod - unc_vals)) & (obs <= (mod + unc_vals))
        coverage_pct = (np.sum(in_interval) / len(obs)) * 100
    else:
        coverage_pct = np.nan
    if np.isnan(coverage_pct):
        coverage_label = "N/A"
    else:
        coverage_label = "Pass" if coverage_pct >= 90 else "Fail"

    #   Bias & MoE
    bias_val = float(np.mean(mod) - np.mean(obs))
    n = len(obs)
    if n > 1:
        t_crit_90 = stats.t.ppf(1 - alpha_gs_bias/2, n - 1)
        MoE_val = t_crit_90 * sd_obs / np.sqrt(n)
    else:
        MoE_val = np.nan
    bias_label = "Pass" if abs(bias_val) < MoE_val else "Fail"

    #   Pass/fail checks
    r2_label = "Pass" if r2 > 0 else "Fail"
    rmse_stddev_label = "Pass" if rmse < sd_obs else "Fail"

    #   Return dictionary of results
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
        "T-Statistic": t_stat,
        "P-Value": p_val,
        "R2>0": r2_label,
        "RMSE<StdDevObs": rmse_stddev_label,
        "Err not stat diff from 0": err_not_diff_label,
        "Coverage>90%": coverage_label,
        "Bias<MoE": bias_label,
    }

# --------------------- Process Data ---------------------
#   This function handles grouping and calls calc_metrics.
def process_data(dfsub, group_tuple):
    #   Build group label
    if group_tuple:
        group_label = "-".join([str(g) for g in group_tuple])
    else:
        group_label = "All"

    #   20 cm isda
    res_20 = calc_metrics(
        dfsub, obs_20, mod_20, isda_unc_field, "20cm",
        "isda", group_label
    )
    if res_20:
        row_20 = {"Depth": "20cm", "Model": "isda"}
        for i, f in enumerate(grouping_fields):
            row_20[f] = group_tuple[i] if len(group_tuple) > i else None
        row_20.update(res_20)
        rows.append(row_20)

    #   30 cm for each model
    for mk, mc in mod_30_dict.items():
        if mk == "soilGrids":
            unc_col = soilgrids_unc_field
        else:
            unc_col = None
        res_30 = calc_metrics(
            dfsub, obs_30, mc, unc_col, "30cm",
            mk, group_label
        )
        if res_30:
            row_30 = {"Depth": "30cm", "Model": mk}
            for i, f in enumerate(grouping_fields):
                row_30[f] = (group_tuple[i] if len(group_tuple) > i
                             else None)
            row_30.update(res_30)
            rows.append(row_30)

# --------------------- Main Execution ---------------------
#   Store rows for results
rows = []

#   Read CSV
df = pd.read_csv(csv_in)
df.columns = df.columns.str.strip()

#   Process with or without grouping
if len(grouping_fields) == 0:
    process_data(df, ())
else:
    grouped = df.groupby(grouping_fields)
    for grp_vals, grp_df in grouped:
        if not isinstance(grp_vals, tuple):
            grp_vals = (grp_vals,)
        process_data(grp_df, grp_vals)

#   Create output DataFrame
df_out = pd.DataFrame(rows)

#   Create suffix for grouping
grp_suffix = ("_".join(grouping_fields)
              if grouping_fields else "noGrouping")

#   Build output CSV name
out_name = f"metrics_{grp_suffix}.csv"
out_path = os.path.join(out_dir, out_name)

#   Round numeric columns
df_out = df_out.round(2)

#   Save CSV
df_out.to_csv(out_path, index=False)
print(f"Saved metrics CSV at: {out_path}")

# ----------------- Print summary of which combos passed GS vs. VCS vs. both -----------------

# For convenience, let's define your grouping column name
# (Assuming you only have 1 grouping_field = ["site"])
group_col = grouping_fields[0] if grouping_fields else None

# 1) GS passing: R2>0, RMSE<StdDevObs, Bias<MoE
df_gs = df_out[
    (df_out["R2>0"] == "Pass")
    & (df_out["RMSE<StdDevObs"] == "Pass")
    & (df_out["Bias<MoE"] == "Pass")
]

print("\n--- GS Criteria Pass ---")
print(f"Number of model+site combos passing GS = {len(df_gs)}")

if not df_gs.empty:
    print("List of combos (Model, Site, Depth):")
    for idx, row in df_gs.iterrows():
        site_val = row[group_col] if group_col else "N/A"
        print(f"  Model={row['Model']}, {group_col}={site_val}, Depth={row['Depth']}")

# 2) VCS passing: R2>0, Err not stat diff from 0 (p>0.05), Coverage>90%
df_vcs = df_out[
    (df_out["R2>0"] == "Pass")
    & (df_out["Err not stat diff from 0"] == "Pass")
    & (df_out["Coverage>90%"] == "Pass")
]

print("\n--- VCS Criteria Pass ---")
print(f"Number of model+site combos passing VCS = {len(df_vcs)}")

if not df_vcs.empty:
    print("List of combos (Model, Site, Depth):")
    for idx, row in df_vcs.iterrows():
        site_val = row[group_col] if group_col else "N/A"
        print(f"  Model={row['Model']}, {group_col}={site_val}, Depth={row['Depth']}")

# 3) Both GS + VCS passing
df_both = df_out[
    (df_out["R2>0"] == "Pass")
    & (df_out["RMSE<StdDevObs"] == "Pass")
    & (df_out["Bias<MoE"] == "Pass")
    & (df_out["Err not stat diff from 0"] == "Pass")
    & (df_out["Coverage>90%"] == "Pass")
]

print("\n--- Both GS & VCS Criteria Pass ---")
print(f"Number of model+site combos passing BOTH = {len(df_both)}")

if not df_both.empty:
    print("List of combos (Model, Site, Depth):")
    for idx, row in df_both.iterrows():
        site_val = row[group_col] if group_col else "N/A"
        print(f"  Model={row['Model']}, {group_col}={site_val}, Depth={row['Depth']}")


# ----------------------- Build & Save Metrics CSV (Simple) -----------------------

# We'll assume you already have df_out from your calculations
group_col = grouping_fields[0] if len(grouping_fields) >= 1 else "Group"

# A lookup table to rename the "Model" entries
model_map = {
    "isda":      "iSDA",
    "soilGrids": "SoilGrids",
    "OLM2017":   "OLM",
    "SA":        "SA"
}

table_csv_rows = []
for idx, row in df_out.iterrows():
    # Extract needed numeric & pass/fail data
    r2_val   = row.get("R2",            np.nan)
    rmse_val = row.get("RMSE",          np.nan)
    sd_val   = row.get("Std Dev (obs)", np.nan)
    bias_val = row.get("Bias",          np.nan)
    moe_val  = row.get("MoE",           np.nan)
    p_val    = row.get("P-Value",       np.nan)
    cov_val  = row.get("Coverage (%)",  np.nan)

    # Build strings
    fit_str = f"{r2_val:.2f}" if not np.isnan(r2_val) else "N/A"
    error_str = (
        f"RMSE: {rmse_val:.2f}, SD: {sd_val:.2f}"
        if not np.isnan(rmse_val) and not np.isnan(sd_val)
        else "N/A"
    )
    bias_str = (
        f"Bias: {bias_val:.2f}, MoE: {moe_val:.2f}"
        if not np.isnan(bias_val) and not np.isnan(moe_val)
        else "N/A"
    )
    p_str = f"{p_val:.3f}" if not np.isnan(p_val) else "N/A"
    coverage_str = f"{cov_val:.2f}%" if not np.isnan(cov_val) else "N/A"

    # Rename the model if in model_map
    original_model = row.get("Model", "N/A")
    renamed_model = model_map.get(original_model, original_model)

    # Build row data
    row_dict = {
        "Model":  renamed_model,
        "Group":  row.get(group_col, "N/A"),
        "Fit, R2>0":         fit_str,
        "Error, RMSE<SD":    error_str,
        "Bias<MoE":          bias_str,
        "Bias, P-value>0.05": p_str,
        "Coverage>90%":          coverage_str
    }
    table_csv_rows.append(row_dict)

# Create a new DataFrame for these columns
df_table = pd.DataFrame(table_csv_rows, columns=[
    "Model", "Group", "Fit, R2>0", "Error, RMSE<SD",
    "Bias<MoE", "Bias, P-value>0.05", "Coverage>90%"
])

# Sort by Model in your custom order: iSDA, SoilGrids, OLM, SA
# We'll use a pandas Categorical to define the order:
import pandas as pd
model_cat = pd.CategoricalDtype(
    categories=["iSDA", "SoilGrids", "OLM", "SA"], 
    ordered=True
)
df_table["Model"] = df_table["Model"].astype(model_cat)

# Now sort by Model (in that custom order) and then by Group
df_table.sort_values(by=["Group", "Model"], inplace=True, na_position="last")

# Save to CSV
simple_csv_name = f"metrics_simple_{grp_suffix}.csv"
simple_csv_path = os.path.join(out_dir, simple_csv_name)
df_table.to_csv(simple_csv_path, index=False, encoding="utf-8")

print(f"Saved simple CSV at: {simple_csv_path}")




# --------------------- Script Completion ---------------------
#   End time
end_time = time.time()

#   Compute runtime
runtime_sec = end_time - start_time
hours = int(runtime_sec // 3600)
minutes = int((runtime_sec % 3600) // 60)
seconds = int(runtime_sec % 60)

#   Print final message
print("Script done.")
print(f"Total run time: {hours}h:{minutes}m:{seconds}s")
