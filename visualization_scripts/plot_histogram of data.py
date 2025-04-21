# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 12:30:00 2025

This script reads a CSV file with SOC stock data and creates histograms for
an arbitrary number of SOC fields provided in a list. For each field, the script
calculates and prints statistics (mean, median, std dev, min, max, range),
fits a 2-component GMM to compute a decision threshold, and draws a vertical red
dashed line at that threshold with the split value printed below the x-axis.
Each histogram is drawn with a unique pastel color. The plots are both displayed
and saved to the output directory.

User-adjustable variables at the top control:
  - The location of the stats text (in axis fraction coordinates).
  - The offset (in axis fraction coordinates) for placing the threshold label
    below the x-axis.

If only one histogram or set of stats appears, ensure the other field names
exist in your CSV and contain enough valid (non-NaN) data.

High-compression raster settings for QGIS are noted below.

@author: André
"""
# ----------------------------------------------------------------------------
# Import required libraries
import time                     # For timing script execution
import os                       # For file and directory operations
import pandas as pd             # For CSV data manipulation
import numpy as np              # For numerical computations
import matplotlib.pyplot as plt # For creating visual plots
from sklearn.mixture import GaussianMixture  # For fitting the GMM model
# ----------------------------------------------------------------------------

# ------------------------------- User Config -------------------------------
# Location of stats text in axis fraction coords (0.0 -> 1.0).
# e.g. (0.03, 0.90) near top-left, (0.75, 0.90) near top-right, etc.
stats_text_x = 0.77
stats_text_y = 0.95
fontsize= 12

# Threshold label offset below the x-axis, in axis fraction coordinates.
# Negative value places it below the axis line.
threshold_label_y_offset = -0.07

# Number of bins for histogram plots
bins_count = 50

# Max value of x Axis
max_axis_val = 215  # 100 for 20cm depth, 215 for 30cm depth

# Padding between x-axis label and tick labels
x_label_pad = 12                

# ---------------------------------------------------------------------------
# Record the script's start time
start_time = time.time()

# ----------------------------------------------------------------------------
# Define the CSV file path (update as needed)
csv_path = (
    r"D:\TerraCarbon\2025_projects\SoilOrganicCarbon\QGIS\data\ICRAF"
    r"\ldsf_soc_stocks_updated.csv"
)

# Define the output directory for histogram plots
output_dir = (
    r"D:\TerraCarbon\2025_projects\SoilOrganicCarbon\QGIS\data\ICRAF"
    r"\plots\histograms"
)
os.makedirs(output_dir, exist_ok=True)

# Define the list of SOC fields to analyze (must match CSV exactly!)
# New: dictionary to map original field names to desired display names
fields = {
    "predsoc_stock_20cm": "LDSF observed SOC stock (20cm)",
    "predsoc_stock_30cm": "LDSF observed SOC stock (30cm)",
    "isda SOC stock 20cm": "iSDA predicted SOC stock (20cm)",
    "soilGrids SOC stock 30cm": "SoilGrids predicted SOC stock (30cm)",
    "OLM 2017 SOC stock 30cm": "OLM predicted SOC stock (30cm)",
    "SA SOC stock 30cm": "SA predicted SOC stock (30cm)m"
}


# ----------------------------------------------------------------------------
def find_gmm_threshold(data_array):
    """
    Fit a 2-component GMM to data_array and return the decision threshold,
    defined as the x-value where the probability of the lower-mode component
    is closest to 0.5. Returns np.nan if not enough data to compute.
    """
    data_array = data_array[~np.isnan(data_array)]
    # Need at least 2 data points to fit a GMM
    if len(data_array) < 2:
        return np.nan

    X = data_array.reshape(-1, 1)  # shape for GMM
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(X)

    # Sort components by ascending mean
    means = gmm.means_.flatten()
    order = np.argsort(means)

    # Create a dense grid over [min, max]
    x_grid = np.linspace(data_array.min(), data_array.max(), 1000)
    # Predict posterior probabilities on the grid
    proba = gmm.predict_proba(x_grid.reshape(-1, 1))

    # Reorder columns so first col is the "lower" mode
    proba_lower = proba[:, order[0]]
    # Find x in x_grid where lower-mode prob is closest to 0.5
    diff = np.abs(proba_lower - 0.5)
    idx = np.argmin(diff)
    return x_grid[idx]

# ----------------------------------------------------------------------------
# Define a list of pastel colors to assign uniquely to each field
# 
pastel_colors = [
    "lightgreen", "lightblue", "lightyellow", "lightsalmon", "lightcoral", 
    "lavender", "lightpink", "lightcyan", "lightseagreen"
]

# Read the CSV file into a DataFrame and strip whitespace from column names
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()

# ----------------------------------------------------------------------------
# Loop through each field to create histogram, compute stats, and add GMM split
for i, field in enumerate(fields):
    if field not in df.columns:
        print(f"\nField '{field}' not found in CSV.")
        print("Available columns:", df.columns.tolist())
        continue

    # Filter out NaN values
    data_vals = df[field].dropna().values
    if len(data_vals) == 0:
        print(f"\nField '{field}' has no valid data (all NaN). Skipping.")
        continue

    # Assign a unique pastel color for this field
    hist_color = pastel_colors[i % len(pastel_colors)]

    # Create a figure and axis for the histogram
    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot the histogram for the current field
    ax.hist(data_vals, bins=bins_count, color=hist_color, edgecolor="black")
    
    # Set x-axis limit based on max_axis_val
    if max_axis_val is not None:
        ax.set_xlim(0, max_axis_val)  # Set fixed x-axis limit from 0 to max_axis_val
    else:
        ax.set_xlim(0, np.nanmax(data_vals) * 1.05)  # Auto-scale with 5% padding


    # Calculate basic statistics
    mean_val = np.mean(data_vals)
    median_val = np.median(data_vals)
    std_val = np.std(data_vals)
    min_val = np.min(data_vals)
    max_val = np.max(data_vals)
    rng_val = max_val - min_val

    # Print summary stats to console
    print(f"\n--- {field} Statistics ---")
    print(f"Mean:   {mean_val:.2f}")
    print(f"Median: {median_val:.2f}")
    print(f"Std:    {std_val:.2f}")
    print(f"Min:    {min_val:.2f}")
    print(f"Max:    {max_val:.2f}")
    print(f"Range:  {rng_val:.2f}")

    # Build a stats text string
    stats_text = (
        f"Mean:   {int(round(mean_val))}\n"
        f"Median: {int(round(median_val))}\n"
        f"Std:    {int(round(std_val))}\n"
        f"Min:    {int(round(min_val))}\n"
        f"Max:    {int(round(max_val))}\n"
        f"Range:  {int(round(rng_val))}"
    )



    # Place stats text on the plot (in fraction coords)
    ax.text(
        stats_text_x, stats_text_y, stats_text,
        transform=ax.transAxes,
        color="black",
        ha="left",
        va="top",
        fontsize=fontsize
    )

    # Compute GMM threshold for splitting the data
    # Only compute and draw the threshold for LDSF plots
    if field in ["predsoc_stock_20cm", "predsoc_stock_30cm"]:
        threshold_val = find_gmm_threshold(data_vals)
        if np.isnan(threshold_val):
            print("Not enough data to fit a GMM for threshold.\n")
        else:
            # Draw a vertical red dashed line at the threshold
            ax.axvline(threshold_val, color="red", linestyle="--", linewidth=2)
            # Place the split threshold text below x-axis
            ax.text(
                threshold_val, threshold_label_y_offset,
                f"{threshold_val:.2f}",
                transform=ax.get_xaxis_transform(),
                ha="center", va="top", color="red"
            )
            print(f"Threshold (GMM) at {threshold_val:.2f}\n")


    # Set title and axis labels
    ax.set_title(f"Distribution of {fields.get(field, field)}", fontsize=14)
    ax.set_xlabel("SOC stock (tC/ha)", labelpad=x_label_pad, fontsize=12)
    # plt.tight_layout(rect=[0, 0, 1, 0.95])
    # ax.set_ylabel("Frequency")

    # Construct a safe output file name by replacing spaces with underscores
    safe_field = field.replace(" ", "_")
    hist_output_path = os.path.join(output_dir, f"{safe_field}_hist.png")

    # Save the figure
    fig.savefig(hist_output_path, dpi=300)
    print(f"Histogram saved: {hist_output_path}")

    # Display the plot
    plt.show()
    # Close the figure to free memory resources
    plt.close(fig)

# ----------------------------------------------------------------------------
# Record the script's end time and report total runtime
end_time = time.time()
elapsed_seconds = end_time - start_time
hours = int(elapsed_seconds // 3600)
minutes = int((elapsed_seconds % 3600) // 60)
seconds = int(elapsed_seconds % 60)
print("\nScript done.")
print(f"Time taken: {hours}h:{minutes}m:{seconds}s")
# ----------------------------------------------------------------------------
# QGIS Raster compression settings (if exporting rasters):
# "-co", "COMPRESS=DEFLATE",
# "-co", "PREDICTOR=2",
# "-co", "ZLEVEL=9"
# ----------------------------------------------------------------------------
