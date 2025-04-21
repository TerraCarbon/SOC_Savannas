# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 14:30:00 2025

This script reads a CSV file with SOC stock data and creates scatter plots 
for each combination of grouping fields (if any). For 20cm data, observed =
"predsoc_stock_20cm" and modeled = "isda SOC stock 20cm". For 30cm data, 
observed = "predsoc_stock_30cm" and modeled columns are defined in mod_dict_30.
Each scatter plot includes a 1:1 red dashed line and displays the following 
metrics (computed using observed data):
  • MedAE, RMSE, R2, n,
  • Std Dev of model predictions,
  • NMedAE = MedAE/mean(obs),
  • NRMSE = RMSE/mean(obs),
  • NRMSEsd = RMSE/std(obs).
  
The metrics are printed as text on the plot.

@author: André
"""
# --------------------------------------------------------------------------
import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
# --------------------------------------------------------------------------

# ----------------------------- User Config --------------------------------
# e.g., ["GPW", "LDSF_quant_SOCstock", "ADMIN", "Vegetation", "site"] or [] for no grouping
grouping_fields = []         ###  <-- UPDATE  ###

# If set, both x and y axes fixed from 0 to this value; else auto (use None)
max_axis_val = 215  # 100 for 20cm depth, 215 for 30cm depth

csv_input = (r"D:\TerraCarbon\2025_projects\SoilOrganicCarbon\QGIS\data\ICRAF"
             r"\ldsf_and_DSM_soc_stocks.csv")
out_dir = (r"D:\TerraCarbon\2025_projects\SoilOrganicCarbon\QGIS\data\ICRAF"
           r"\plots\scatter_customGrouping")
os.makedirs(out_dir, exist_ok=True)

# For 20cm data
obs_20 = "predsoc_stock_20cm"
mod_dict_20 = {"iSDAsoils": "isda SOC stock 20cm"}

# For 30cm data
obs_30 = "predsoc_stock_30cm"
mod_dict_30 = {
    "South Africa": "SA SOC stock 30cm",
    "OLM": "OLM 2017 SOC stock 30cm",
    "soilGrids": "soilGrids SOC stock 30cm"
}
# --------------------------------------------------------------------------

start_time = time.time()

df = pd.read_csv(csv_input)
df.columns = df.columns.str.strip()

def create_scatter(dfsub, xcol, ycol, model, title, path):
    dfsub = dfsub.copy()
    dfsub[xcol] = pd.to_numeric(dfsub[xcol], errors="coerce")
    dfsub[ycol] = pd.to_numeric(dfsub[ycol], errors="coerce")
    dfsub.dropna(subset=[xcol, ycol], inplace=True)
    if dfsub.empty:
        print(f"No data for plot: {title}")
        return

    x = dfsub[xcol].values
    y = dfsub[ycol].values
    n = len(x)
    r2 = r2_score(x, y) if n > 1 else np.nan
    rmse = np.sqrt(np.mean((y - x)**2))
    mean_obs = np.mean(x) if n else np.nan
    nrmse = rmse / mean_obs if mean_obs != 0 else np.nan

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, c="blue", alpha=0.6, edgecolors="none")

    # Optionally add best-fit line
    if n > 1:
        slope, intercept = np.polyfit(x, y, 1)
        fit_x = np.array(
            [0, max_axis_val]) if max_axis_val else np.linspace(
                0, x.max()*1.05, 100)
        fit_y = slope * fit_x + intercept
        plt.plot(fit_x, fit_y, "g--", label="Regression line")

    # 1:1 line
    if max_axis_val is not None:
        ax_lim = max_axis_val
    else:
        ax_lim = max(np.nanmax(x), np.nanmax(y)) * 1.05
    plt.plot([0, ax_lim], [0, ax_lim], "r--", label="1:1 line")

    plt.xlim(0, ax_lim)
    plt.ylim(0, ax_lim)

    # Make the title bold and slightly larger
    plt.title(f"{model} - {title}", fontsize=16, fontweight="bold")

    plt.xlabel("LDSF SOC stock samples", fontsize=16)
    plt.ylabel("Modeled SOC stock", fontsize=16)

    # Build metrics text in bold math style, with R^2 as an exponent, and using colons
    met_txt = (
        f"$\\boldsymbol{{NRMSE: {nrmse:.2f}}}$\n"
        f"$\\boldsymbol{{R^{{2}}: {r2:.2f}}}$\n"
        f"$\\boldsymbol{{n: {n}}}$"
    )
    plt.text(0.03 * ax_lim, 0.82 * ax_lim,
             met_txt,
             fontsize=16,  # bigger than title
             color="black")

    # Change tick label font sizes:
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.show()
    plt.close()
    print(f"Saved scatter plot: {path}")



def process_depth(depth="20cm"):
    if depth == "20cm":
        obs = obs_20
        mods = mod_dict_20
    else:
        obs = obs_30
        mods = mod_dict_30

    if len(grouping_fields) == 0:
        yield ((), df.copy())
    else:
        gp = df.groupby(grouping_fields)
        for keyvals, df_grp in gp:
            if not isinstance(keyvals, tuple):
                keyvals = (keyvals,)
            yield (keyvals, df_grp)

for dpth in ["20cm", "30cm"]:
    for keyvals, dfg in process_depth(dpth):
        sub_title = " - ".join(str(k) for k in keyvals) if keyvals else "All"
        if dpth == "20cm":
            for mk, mc in mod_dict_20.items():
                t = f"{dpth} - {sub_title}"
                fn = f"scatter_{dpth}_{mk}_{'_'.join(keyvals)}.png"
                pth = os.path.join(out_dir, fn)
                create_scatter(dfg, obs_20, mc, mk, t, pth)
        else:
            for mk, mc in mod_dict_30.items():
                t = f"{dpth} - {sub_title}"
                fn = f"scatter_{dpth}_{mk}_{'_'.join(keyvals)}.png"
                pth = os.path.join(out_dir, fn)
                create_scatter(dfg, obs_30, mc, mk, t, pth)

end_time = time.time()
secs = end_time - start_time
hrs = int(secs // 3600)
mins = int((secs % 3600) // 60)
s = int(secs % 60)
print(f"\nScript done in {hrs}h:{mins}m:{s}s.")
