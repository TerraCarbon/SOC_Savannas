# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 14:10:00 2025

This script computes R2, MedAE, RMSE, NMedAE, NRMSE, NRMSEsd for each scenario,
then produces four line+point subplots (R2, NMedAE, NRMSE, NRMSEsd).

Features:
- Adjustable figure width and height via variables.
- R2 plot: thick black horizontal line at y=0.
- NRMSE, NRMSEsd plots: thick black horizontal line at y=1.0.
- Light gray background highlighting to differentiate between classes
- Top row: metric titles, no x-tick labels.
- Bottom row: metric names as x-axis labels, no titles.

@author: André
"""
# -------------------------------------------------------------------------
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
# -------------------------------------------------------------------------

# ----------------------------- User Config -------------------------------
csv_in = (r"D:\TerraCarbon\2025_projects\SoilOrganicCarbon\QGIS\data\ICRAF"
          r"\ldsf_and_DSM_soc_stocks.csv")
out_dir = (r"D:\TerraCarbon\2025_projects\SoilOrganicCarbon\QGIS\data\ICRAF"
           r"\metrics")
os.makedirs(out_dir, exist_ok=True)
out_plot = os.path.join(out_dir, "scenarios_plot.png")

# Set save_individual to True for individual PNGs; False for one combined PNG
save_individual = True

# Control figure size
fig_width = 20    # Increase for wider plots
fig_height = 14   # Increase for taller plots

obs_20 = "predsoc_stock_20cm"
mod_20 = "isda SOC stock 20cm"
obs_30 = "predsoc_stock_30cm"
mod_30_dict = {
    "soilGrids": "soilGrids SOC stock 30cm",
    "OLM": "OLM 2017 SOC stock 30cm",
    "SA": "SA SOC stock 30cm"
}

label_replacements = {
    "wooded_grassland": "W grassland",  # Replace this label
    # Add more replacements if needed:
    # "grassland": "G grassland",
    # "Other LC": "Other Land Cover"
}
    
highlight_ranges = [
    (0.5, 1.5),  # Low 
    (2.5, 3.5),  # Low 
    (4.5, 5.5),  # Low 
]


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# ----------------------------- Scenario Filters ------------------------------


###################   UPDATE DEPENDING ON SCENARIO   ##########################


# # Scenario: All Groupings
# def no_filter(df): return df
# def filter_low(df): return df[df["LDSF_quantity_SOCstock"]=="Low"]
# def filter_high(df): return df[df["LDSF_quantity_SOCstock"]=="High"]
# def filter_cult(df): return df[df["GPW"]=="Cultivated"]
# def filter_nat(df): return df[df["GPW"]=="Natural"]
# def filter_other(df): return df[df["GPW"]=="Other LC"]
# def filter_grass(df): return df[df["Vegetation"]=="grassland"]
# def filter_wooded(df): return df[df["Vegetation"]=="wooded_grassland"]
# def filter_lesotho(df): return df[df["ADMIN"]=="Lesotho"]
# def filter_esw(df): return df[df["ADMIN"]=="eSwatini"]
# def filter_kenya(df): return df[df["ADMIN"]=="Kenya"]
# def filter_rwanda(df): return df[df["ADMIN"]=="Rwanda"]

# scenarios = [
#     ("All data", no_filter),
#     ("Low", filter_low),
#     ("High", filter_high),
#     ("Cultivated", filter_cult),
#     ("Natural", filter_nat),
#     ("Other LC", filter_other),
#     ("grassland", filter_grass),
#     ("wooded_grassland", filter_wooded),
#     ("Lesotho", filter_lesotho),
#     ("eSwatini", filter_esw),
#     ("Kenya", filter_kenya),
#     ("Rwanda", filter_rwanda)
# ]

# # Gray background spans to differentiate classes
# highlight_ranges = [
#     (0.5, 2.5),  # Low & High 
#     (5.5, 7.5),  # Vegetation
# ]

# # Black line vertical dividers:
# vertical_dividers = [0.5,2.5,5.5,7.5]  


#------------------------------------------------------------------------------
# Scenario:Landcover and Vegetation by Quantity
def no_filter(df): return df
def filter_low_cult(df): 
    return df[(df["LDSF_quantity_SOCstock"]=="Low") & (df["GPW"]=="Cultivated")]
def filter_high_cult(df): 
    return df[(df["LDSF_quantity_SOCstock"]=="High") & (df["GPW"]=="Cultivated")]
def filter_low_nat(df): 
    return df[(df["LDSF_quantity_SOCstock"]=="Low") & (df["GPW"]=="Natural")]
def filter_high_nat(df): 
    return df[(df["LDSF_quantity_SOCstock"]=="High") & (df["GPW"]=="Natural")]
def filter_low_other(df): 
    return df[(df["LDSF_quantity_SOCstock"]=="Low") & (df["GPW"]=="Other")]
def filter_high_other(df): 
    return df[(df["LDSF_quantity_SOCstock"]=="High") & (df["GPW"]=="Other")]
def filter_low_grass(df): 
    return df[(df["LDSF_quantity_SOCstock"]=="Low") & (df["Vegetation"]=="grassland")]
def filter_high_grass(df): 
    return df[(df["LDSF_quantity_SOCstock"]=="High") & (df["Vegetation"]=="grassland")]
def filter_low_wooded(df): 
    return df[(df["LDSF_quantity_SOCstock"]=="Low") & (df["Vegetation"]=="wooded_grassland")]
def filter_high_wooded(df): 
    return df[(df["LDSF_quantity_SOCstock"]=="High") & (df["Vegetation"]=="wooded_grassland")]

scenarios = [
    ("All data", no_filter),
    ("Low Grassland", filter_low_grass),
    ("High Grassland", filter_high_grass),
    ("Low W Grassland", filter_low_wooded),
    ("High W Grassland", filter_high_wooded),
    ("Low Other LC", filter_low_other),
    ("High Other LC", filter_high_other),
    ("Low Natural", filter_low_nat),
    ("High Natural", filter_high_nat),
    ("Low Cultivated", filter_low_cult),
    ("High Cultivated", filter_high_cult),

]

# Gray background spans to differentiate classes
highlight_ranges = [
    (0.5, 1.5),  # Low Grassland
    (2.5, 3.5),  # Low W Grassland
    (4.5, 5.5),  # Low Other LC
    (6.5, 7.5),  # Low Natural
    (8.5, 9.5)   # Low Cultivated
]

# Black line vertical dividers:
vertical_dividers = [0.5,4.5] #low/high LC

# -----------------------------------------------------------------------------
# # Scenario: Quantity and Quartiles
# def no_filter(df): return df

# def Low(df): 
#     return df[(df["LDSF_quantity_SOCstock"]=="Low")]
# def High(df): 
#     return df[(df["LDSF_quantity_SOCstock"]=="High")]

# def Q1(df): 
#     return df[(df["quartiles"]=="Q1")]
# def Q2(df): 
#     return df[(df["quartiles"]=="Q2")]
# def Q3(df): 
#     return df[(df["quartiles"]=="Q3")]
# def Q4(df): 
#     return df[(df["quartiles"]=="Q4")]

# scenarios = [
#     ("All data", no_filter),
#     ("Low", Low),
#     ("High", High),
#     ("Q1", Q1),
#     ("Q2", Q2),
#     ("Q3", Q3),
#     ("Q4", Q4)
# ]

# highlight_ranges = [
#     (0.5, 1.5),  # Low 
#     (2.5, 3.5),  # Low 
#     (4.5, 5.5),  # Low 
# ]

# vertical_dividers = [0.5,2.5] #low, Q1, Q3


# -----------------------------------------------------------------------------
# ------------------------ END SCENARIO SECTION -------------------------------
# -----------------------------------------------------------------------------

# ------------------------- Metrics Calculation ---------------------------
def calc_metrics(dfsub, obs_col, mod_col):
    d = dfsub.dropna(subset=[obs_col, mod_col]).copy()
    if d.empty: return None
    obs = d[obs_col].values
    mod = d[mod_col].values
    medae = np.median(np.abs(mod - obs))
    rmse = np.sqrt(mean_squared_error(obs, mod))
    r2v = r2_score(obs, mod)
    mean_obs = np.mean(obs)
    sd_obs = np.std(obs)
    nmedae = medae/mean_obs if mean_obs != 0 else np.nan
    nrmse = rmse/mean_obs if mean_obs != 0 else np.nan
    nrmse_sd = rmse/sd_obs if sd_obs != 0 else np.nan
    return {"R2": r2v, "MedAE": medae, "RMSE": rmse, 
            "NMedAE": nmedae, "NRMSE": nrmse, "NRMSEsd": nrmse_sd}

# ------------------------- Build Metrics Table ---------------------------
df_in = pd.read_csv(csv_in)
df_in.columns = df_in.columns.str.strip()

rows = []
for label, fn_filter in scenarios:
    df_scen = fn_filter(df_in)
    # 20cm metrics
    m20 = calc_metrics(df_scen, obs_20, mod_20)
    if m20:
        rows.append({"Scenario": label, "Depth": "20cm", "Model": "iSDA", **m20})
    # 30cm metrics
    for mk, mc in mod_30_dict.items():
        m30 = calc_metrics(df_scen, obs_30, mc)
        if m30:
            rows.append({"Scenario": label, "Depth": "30cm", "Model": mk, **m30})

df_metrics = pd.DataFrame(rows).round(3)

# ------------------------- Plotting --------------------------------------
metrics_to_plot = ["R2", "NMedAE", "NRMSE", "NRMSEsd"]
scenario_list = [s[0] for s in scenarios]
scen2x = {sc: i for i, sc in enumerate(scenario_list)}

if not save_individual:
    fig, axes = plt.subplots(2, 2, figsize=(fig_width, fig_height))
    axes = axes.ravel()
    for i, (ax, metric) in enumerate(zip(axes, metrics_to_plot)):
        for start, end in highlight_ranges:
            ax.axvspan(start, end, color="lightgray", alpha=0.2)
        for model, dfg in df_metrics.groupby("Model"):
            xvals = [scen2x[s] for s in dfg["Scenario"]]
            yvals = dfg[metric].values
            ax.plot(xvals, yvals, marker="o", label=model)
        if metric == "R2":
            ax.axhline(0, color="black", linewidth=2.5, linestyle="--", zorder=0)
        if metric in ["NRMSE", "NRMSEsd"]:
            ax.axhline(1.0, color="black", linewidth=2.5, linestyle="--", zorder=0)
        if vertical_dividers:
            for v in vertical_dividers:
                ax.axvline(v, color="black", linewidth=1.5)
        ax.set_xticks(range(len(scenario_list)))
        if i < 2:
            ax.set_title(metric, fontsize=16, fontweight="bold")
            ax.set_xticklabels([])
        else:
            ax.set_xlabel(metric, fontsize=16, fontweight="bold")
            custom_labels = [label_replacements.get(lbl, lbl) for lbl in scenario_list]
            ax.set_xticklabels(custom_labels, rotation=45, ha="right", fontsize=16)
        ax.tick_params(axis="y", labelsize=16)
    axes[0].legend(loc="best", fontsize=16)
    plt.subplots_adjust(hspace=0.1)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_plot, dpi=150)
    plt.show()
    print(f"Combined plot saved: {out_plot}")
else:
    # Individual plots: one png per metric
    for metric in metrics_to_plot:
        fig, ax = plt.subplots(figsize=(fig_width/2, fig_height/2))
        for start, end in highlight_ranges:
            ax.axvspan(start, end, color="lightgray", alpha=0.2)
        for model, dfg in df_metrics.groupby("Model"):
            xvals = [scen2x[s] for s in dfg["Scenario"]]
            yvals = dfg[metric].values
            ax.plot(xvals, yvals, marker="o", label=model)
        if metric == "R2":
            ax.axhline(0, color="black", linewidth=2.5, linestyle="--", zorder=0)
        if metric in ["NRMSE", "NRMSEsd"]:
            ax.axhline(1.0, color="black", linewidth=2.5, linestyle="--", zorder=0)
        if vertical_dividers:
            for v in vertical_dividers:
                ax.axvline(v, color="black", linewidth=1.5)
        ax.set_xticks(range(len(scenario_list)))
        custom_labels = [label_replacements.get(lbl, lbl) for lbl in scenario_list]
        ax.set_xticklabels(custom_labels, rotation=45, ha="right", fontsize=16)
        ax.set_xlabel("")
        ax.set_title(metric, fontsize=16, fontweight="bold")
        ax.tick_params(axis="y", labelsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"scatter_{metric}.png"), dpi=150)
        plt.show()
        print(f"Individual plot for {metric} saved.")
        plt.close(fig)
        
# -------------------------------------------------------------------------
