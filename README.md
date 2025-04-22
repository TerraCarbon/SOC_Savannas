# SOC_Savannas
## Repository for data to be used in deliverable 4 dashboard of SOC Savanna project

### Deliverable 2
- ***LDSF_and_DSM_soc_stocks.csv*** : Table with SOC stock quantities for LDSF observations and DSM predictions. LDSF sample data provided by ICRAF. DSMs were publicly available for download.

- **visualization_scripts** : Folder containing scripts used to create visualizations for Deliverable 2 [write-up](https://docs.google.com/document/d/1DPCL_MbP-KHWClp3BZtj47DRd_q52aVi6-yFihE-WvI/edit?tab=t.0) and [slide deck](https://docs.google.com/presentation/d/1lVm03xCva5bzybsERM2RNNHQakpT8e1ArQoX1yOq8Bc/edit?usp=sharing)
  - *plot_histogram of data.py* : Create histograms showing the distribution of SOC stock in the LDSF observations and DSM predictions.
  - *box_plots_custom_grouping.py* : Create box plots showing the distribution of SOC stock in the LDSF observations and DSM predictions. Optional grouping of data by strata.
  - *scatter_plots_custom_groupings.py* : Create scatter plots showing SOC stock observations versus predictions by model. Optional grouping of data by strata.
  - *calc_ALL_metrics_and_validation_criteria.py* : Calculates evaluation metrics and determines if models pass or fail validation requirements. Exports as csv. Optional grouping of data by strata.
  - *calc_validation_criteria.py* : Calculates validation criteria for models. These are the summarized tables used in the write up. Exports as csv. Optional grouping of data by strata.

- DSM rasters for each LDSF sampling site are uploaded to the shared TC/ICRAF drive under the Deliverable 4 [folder](https://drive.google.com/drive/folders/1K4IaiV7A_20qXnASSA9VcV-liazGPecC?usp=sharing)

### Deliverable 3
- *Dev3_uncertainty_calculation.R* : Script for calculating uncertainty deductions for SOC stock change in a project.  

*repo created by André*
