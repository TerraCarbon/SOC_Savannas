# -*- coding: utf-8 -*-
"""
Created on Tue Apr 01 11:05:00 2025

This script reads a CSV file with SOC stock data and creates vertical whisker
(box) plots for a custom-defined list of SOC fields in a specified order:
  1) LDSF 20cm (observed),
  2) LDSF 30cm (observed),
  3) iSDAsoils 20cm (prediction),
  4) SoilGrids 30cm (prediction),
  5) OLM 2017 30cm (prediction),
  6) SA SOC stock 30cm (prediction).

Observations use a more saturated color for 20cm (green) or 30cm (blue).
Predictions use a lighter/desaturated color from the same gradient (green for
20cm, blue for 30cm). A vertical black line separates observation boxes from
prediction boxes.

Labels have two lines (e.g., "LDSF \n 20cm depth") and a bold group label
("SOC stock observations" or "SOC stock predictions") is placed *underneath*
each group of boxes. The user can specify the vertical and horizontal offsets of
these group labels at the top of the script.

The user can also specify grouping fields. If 'grouping_fields' is empty, one
figure is generated for all data. If non-empty, one figure per group is generated.
Each figure saves to the specified output directory.

The script prints the five-number summary to the left of each box and ends with
a total runtime report.

@author: André
"""
# ----------------------------------------------------------------------------
# Import required libraries
import time                # for timing script execution
import os                  # for file and directory operations
import pandas as pd        # for CSV data manipulation
import numpy as np         # for numerical computations
import matplotlib.pyplot as plt  # for creating visual plots
# ----------------------------------------------------------------------------

# ------------------------------ User Config ---------------------------------
# List of grouping fields. If empty, one plot is generated for the whole data.
# e.g., ["GPW", "LDSF_quant_SOCstock", "ADMIN", "Vegetation", "site"] or 
# [] for no grouping
grouping_fields = []    ###  <-- UPDATE  ###

# Define the CSV file path (update as needed)
csv_path = (
    r"D:\TerraCarbon\2025_projects\SoilOrganicCarbon\QGIS\data\ICRAF"
    r"\ldsf_and_DSM_soc_stocks.csv"
)

# Define the output directory for whisker (box) plots
output_dir = (
    r"D:\TerraCarbon\2025_projects\SoilOrganicCarbon\QGIS\data\ICRAF"
    r"\plots\whisker_plots"
)

# Define the exact fields in the desired order
fields_order = [
    "predsoc_stock_20cm",       # LDSF 20cm
    "predsoc_stock_30cm",       # LDSF 30cm
    "isda SOC stock 20cm",      # iSDAsoils 20cm
    "soilGrids SOC stock 30cm", # SoilGrids 30cm
    "OLM 2017 SOC stock 30cm",  # OLM 2017 30cm
    "SA SOC stock 30cm"         # SA 30cm
]

# Dictionary mapping each field to multiline x-axis labels
x_label_dict = {
    "predsoc_stock_20cm": "$\\mathbf{LDSF}$\n20cm depth",
    "predsoc_stock_30cm": "$\\mathbf{LDSF}$\n30cm depth",
    "isda SOC stock 20cm": "$\\mathbf{iSDA}$\n20cm depth",
    "soilGrids SOC stock 30cm": "$\\mathbf{SoilGrids}$\n30cm depth",
    "OLM 2017 SOC stock 30cm": "$\\mathbf{OLM}$\n30cm depth",
    "SA SOC stock 30cm": "$\\mathbf{SA}$\n30cm depth"
}


# Colors for observations (more saturated) and predictions (lighter),
# using similar gradients for 20cm (green) and 30cm (blue).
obs_20_color = "#2ecc71"  # saturated green for 20cm observations
obs_30_color = "#3498db"  # saturated blue for 30cm observations
pred_20_color = "#abebc6" # lighter green for 20cm predictions
pred_30_color = "#aed6f1" # lighter blue for 30cm predictions

# Define which fields are observations vs. predictions at each depth
observation_fields_20 = ["predsoc_stock_20cm"]
observation_fields_30 = ["predsoc_stock_30cm"]
prediction_fields_20 = ["isda SOC stock 20cm"]
prediction_fields_30 = [
    "soilGrids SOC stock 30cm",
    "OLM 2017 SOC stock 30cm",
    "SA SOC stock 30cm"
]

# Spacing between boxes on the x-axis
box_spacing = 1

# Box width (horizontal size of each box)
box_width = 0.6

# Horizontal offset for the five-number summary text
text_offset = 0.35

# Vertical offsets for the two group labels
observations_label_y = -0.20  # vertical offset for obs group label
predictions_label_y = -0.20   # vertical offset for pred group label

# Horizontal offsets for the two group labels (adjust as needed)
observations_label_x_offset = 0.07  # horizontal offset for obs group label
predictions_label_x_offset = -0.025   # horizontal offset for pred group label

# Labels for the two groups
observations_label = "SOC stock observations"
predictions_label = "SOC stock predictions"
# ----------------------------------------------------------------------------

# Record the script's start time
start_time = time.time()

# ----------------------------------------------------------------------------
# Create output directory if needed
os.makedirs(output_dir, exist_ok=True)

# ----------------------------------------------------------------------------
# Function to compute five-number summary for annotation
def compute_five_num(data):
    # Return (min, Q1, median, Q3, max) for the given data
    mn = np.min(data)
    q1 = np.percentile(data, 25)
    med = np.median(data)
    q3 = np.percentile(data, 75)
    mx = np.max(data)
    return mn, q1, med, q3, mx

# ----------------------------------------------------------------------------
# Function to convert x-position in data coords to axis fraction for text
def data_coord_to_axis_fraction(x_value, x_min, x_max):
    # Return fraction between 0 and 1 for axis placement
    return (x_value - x_min) / (x_max - x_min)

# ----------------------------------------------------------------------------
# Function to produce the boxplot given a subset DataFrame and a suffix
def produce_boxplot(dfsub, group_suffix=""):
    # Filter to keep only the fields in fields_order if they exist & have data
    final_fields = []
    for fld in fields_order:
        if fld in dfsub.columns:
            nonan = dfsub[fld].dropna()
            if len(nonan) > 0:
                final_fields.append(fld)
            else:
                print(f"Field '{fld}' has no data. Skipping.")
        else:
            print(f"Field '{fld}' not found in CSV. Skipping.")

    # If no fields remain, return
    if not final_fields:
        print("No valid fields found in this subset. Skipping plot.")
        return

    # Collect data for each field
    combined_data = []
    combined_labels = []
    combined_colors = []
    stats_list = []  # Store (min, Q1, median, Q3, max) for each box


    for fld in final_fields:
        arr = dfsub[fld].dropna().values
        combined_data.append(arr)
        combined_labels.append(x_label_dict.get(fld, fld))
        # Determine color based on observation/prediction + depth
        if fld in observation_fields_20:
            color = obs_20_color
        elif fld in observation_fields_30:
            color = obs_30_color
        elif fld in prediction_fields_20:
            color = pred_20_color
        else:
            color = pred_30_color
        combined_colors.append(color)
        
        # compute and store the quartile stats
        stats_list.append(compute_five_num(arr))

    # Determine positions for each box
    num_boxes = len(combined_data)
    positions = np.arange(1, num_boxes + 1) * box_spacing

    # Create a figure for the boxplot
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.subplots_adjust(left=0.15)  # Increase the left margin if needed

    # Create the boxplot with patch_artist=True to allow custom face colors
    bp = ax.boxplot(
        combined_data,
        patch_artist=True,
        vert=True,
        positions=positions,
        widths=box_width,
        labels=combined_labels
    )

    # Assign facecolor for each box
    for patch, c in zip(bp['boxes'], combined_colors):
        patch.set_facecolor(c)

    # Set median line color: red for prediction boxes, white for observation boxes
    for i, median_line in enumerate(bp['medians']):
        if final_fields[i] in prediction_fields_20 or final_fields[i] in prediction_fields_30:
            median_line.set_color("red")
        else:
            median_line.set_color("white")
    
    # Build the title text by appending the group suffix if it exists
    title_text = "SOC stock distribution in LDSF samples and models"
    if group_suffix:
        title_text = f"{group_suffix} {title_text}"

    # Set the main title
    ax.set_title(
        title_text,
        fontsize=18,
        fontweight='bold'
    )

    # Set y-axis label
    ax.set_ylabel("SOC stock (t/ha)", fontsize=16)
    
    ax.tick_params(axis='y', labelsize=12)


    # Set x-ticks and labels horizontally, center, fontsize=16
    ax.set_xticks(positions)
    ax.set_xticklabels(combined_labels, rotation=0, ha="center", fontsize=16)

    # Add vertical black line to separate observations (first 2) from
    # predictions (remaining). Only if at least 3 boxes.
    if num_boxes >= 3:
        # Line after the second box: between positions[1] and positions[2]
        line_xpos = positions[1] + ((positions[2] - positions[1]) / 2.0)
        ax.axvline(x=line_xpos, color="black", linewidth=1)

    # Place group label for observations if at least 2 boxes
    if num_boxes >= 2:
        obs_x = (positions[0] + positions[1]) / 2.0
        obs_x_frac = data_coord_to_axis_fraction(obs_x, positions[0], positions[-1])
        ax.text(
            obs_x_frac + observations_label_x_offset, observations_label_y,
            observations_label,
            transform=ax.transAxes,
            ha="center", va="center",
            fontstyle="italic", fontweight="bold",
            fontsize=16
        )

    # Place group label for predictions if more than 2 boxes
    if num_boxes > 2:
        # Average x for boxes 3..end
        pred_x = np.mean(positions[2:])
        pred_x_frac = data_coord_to_axis_fraction(pred_x, positions[0], positions[-1])
        ax.text(
            pred_x_frac + predictions_label_x_offset, predictions_label_y,
            predictions_label,
            transform=ax.transAxes,
            ha="center", va="center",
            fontstyle="italic", fontweight="bold",
            fontsize=16
        )

    # Tidy up layout
    plt.tight_layout()

      # Annotate each box with whisker endpoints (Q0 and Q4) plus Q1, median, Q3
    for i, pos in enumerate(positions):
        # For each box, retrieve lower and upper whisker endpoints
        lower_whisker_line = bp["whiskers"][2 * i]
        upper_whisker_line = bp["whiskers"][2 * i + 1]
        # Use the second point of each whisker line as the drawn endpoint
        q0 = int(round(lower_whisker_line.get_ydata()[1]))
        q4 = int(round(upper_whisker_line.get_ydata()[1]))
        # Use the computed stats for Q1, median, and Q3 from the earlier stats_list
        _, q1, med, q3, _ = stats_list[i]
        q1 = int(round(q1))
        med = int(round(med))
        q3 = int(round(q3))
        # Place the labels: Q0, Q1, median, Q3, and Q4
        ax.text(pos - text_offset, q0, f"{q0}", ha="right", va="center", 
                fontsize=12, color="black")
        ax.text(pos - text_offset, q1, f"{q1}", ha="right", va="center", 
                fontsize=12, color="black")
        ax.text(pos - text_offset, med, f"{med}", ha="right", va="center", 
                fontsize=12, color="black")
        ax.text(pos - text_offset, q3, f"{q3}", ha="right", va="center", 
                fontsize=12, color="black")
        ax.text(pos - text_offset, q4, f"{q4}", ha="right", va="center", 
                fontsize=12, color="black")


    # Build output filename
    if group_suffix:
        fn = f"SOC_whisker_plots_{group_suffix}.png"
    else:
        fn = "SOC_whisker_plots.png"
    output_path = os.path.join(output_dir, fn)

    # Save figure
    fig.savefig(output_path, dpi=300)
    plt.show()
    plt.close(fig)
    print(f"Saved plot: {output_path}")

# ----------------------------------------------------------------------------
# Read CSV file into DataFrame and strip whitespace from columns
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()

# ----------------------------------------------------------------------------
# Group or not, then call produce_boxplot
if not grouping_fields:
    # No grouping; single figure
    produce_boxplot(df, group_suffix="")
else:
    # Group by specified fields
    grouped = df.groupby(grouping_fields)
    for group_vals, df_grp in grouped:
        # Convert group_vals to tuple if needed
        if not isinstance(group_vals, tuple):
            group_vals = (group_vals,)
        # Create a label by joining group values with underscores
        suffix = "_".join(str(v) for v in group_vals)
        produce_boxplot(df_grp, group_suffix=suffix)

# ----------------------------------------------------------------------------
# Report total runtime
end_time = time.time()
elapsed_seconds = end_time - start_time
hours = int(elapsed_seconds // 3600)
minutes = int((elapsed_seconds % 3600) // 60)
seconds = int(elapsed_seconds % 60)
print(f"Script done. Time taken: {hours}h:{minutes}m:{seconds}s")
