########################################################################
# Script: SOC_Uncertainty_Deduction.R
#
# Adaptation of the "Appendix 6: Example uncertainty calculation" code in the 
# Verra VCS "Tool for Quantifying Organic Carbon Stocks Using Digital Soil 
# Mapping: Calibration, Validation and Uncertainty Estimation".
# https://verra.org/wp-content/uploads/2025/02/Appendix-6.html
# It calculates the variance of the mean SOC stock and stock changes following 
# the methods outlined in Wadoux and Heuvelink (2023)
#
# Workflow:
# 1. Load per‑site SOC stock DSM predictions and their 68% CI (σ) rasters.
# 2. Extract ground‑truth points, compute standardized prediction errors (SPE).
# 3. Fit and select an empirical variogram on SPE to characterize spatial autocorrelation.
# 4. Monte Carlo sample σ‑pairs + variogram to compute variance of the project‑mean SOC stock.
# 5. Simulate a second time‑point DSM (t + Δt), compute ΔSOĈ and its variance via Eq.5.
# 6. Convert to CO₂e units and apply the Verra “probability of exceedance” deduction 
#    (33.3% quantile) for 1,2,3 yr periods.
#
# Outputs:
#  - Best‐fit variogram model file
#  - Project‑mean variance estimates for SOC and SOC change
#  - Probability‐of‐exceedance plots with % deductions
########################################################################

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Section I ─ Setting up the R environment
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
setwd("D:/TerraCarbon/2025_projects/SoilOrganicCarbon/Uncertainty")   
set.seed(1)

required_pkgs <- c("sf", "terra", "gstat", "viridis", "sp", "stats",
                   "ggplot2", "dplyr")
to_install   <- required_pkgs[!(required_pkgs %in% rownames(installed.packages()))]
if (length(to_install)) install.packages(to_install, dependencies = TRUE)
invisible(lapply(required_pkgs, library, character.only = TRUE))

# ──────────────────────────────────────────────────────────────────────────────
# User‑specific paths & variables
# ──────────────────────────────────────────────────────────────────────────────
# Raster with a prediction of SOC stock values
pred_raster_dir  <- "D:/TerraCarbon/2025_projects/SoilOrganicCarbon/Uncertainty/data/isda_lesotho_sites_stock_and_uncertainty"
# Point layers with observed (ground truth) SOC stock values for each site
obs_point_dir    <- "D:/TerraCarbon/2025_projects/SoilOrganicCarbon/Uncertainty/data/points/observation_points"
# Polygons around sample 
site_poly_dir    <- "D:/TerraCarbon/2025_projects/SoilOrganicCarbon/Uncertainty/data/lesotho_sites"
# Directory to save outputs
output_dir       <- "D:/TerraCarbon/2025_projects/SoilOrganicCarbon/Uncertainty/output"
# Projection of layers
crs_project      <- "ESRI:102022"      # Africa Albers Equal‑Area Conic
# Field name of the SOC stock observations field in ground truth data
true_field_name  <- "SOCs20LDSF"       # attribute in point shapefiles

dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# ──────────────────────────────────────────────────────────────────────────────
# Helper to read prediction & uncertainty rasters for every site
# ──────────────────────────────────────────────────────────────────────────────
read_site_rasters <- function(site_name) {
  hat_file   <- file.path(pred_raster_dir, paste0(site_name, "_soc_stock.tif"))
  sigma_file <- file.path(pred_raster_dir, paste0(site_name, "_soc_stock_uncertainty.tif"))
  list(
    soc_stock_hat = rast(hat_file),
    sigma         = rast(sigma_file)
  )
}

# ──────────────────────────────────────────────────────────────────────────────
# Section II ─ Loading DSM predictions (replaces the old simulation block)
# ──────────────────────────────────────────────────────────────────────────────
# Step A – read the polygon layer that lists all sites
site_polys <- list.files(site_poly_dir, pattern = "\\.geojson$", full.names = TRUE) |>
  lapply(st_read) |>
  do.call(what = rbind)


# Step B – for every site, load the *predicted* SOC raster and its 68%‑CI σ
fields_pred <- lapply(site_polys$site, read_site_rasters)
names(fields_pred) <- site_polys$site



# ──────────────────────────────────────────────────────────────────────────────
# Section III – Empirical variogram of the STANDARDISED prediction error
# ──────────────────────────────────────────────────────────────────────────────

# (The standardized error is (Truth−Prediction)/σ.)
# A variogram shows how the spatial correlation decays with distance


set.seed(1)

# Step1 – gather points & extract hat + sigma
message("Extracting prediction/σ at every observed point")
pt_dfs <- vector("list", length = nrow(site_polys))
for(i in seq_len(nrow(site_polys))) {
  site_nm  <- site_polys$site[i]
  pts_sf   <- st_read(file.path(obs_point_dir, paste0(site_nm, "_points.shp")),
                      quiet=TRUE) |>
    st_transform(crs_project)
  hat  <- terra::extract(fields_pred[[site_nm]]$soc_stock_hat, pts_sf)[,2]
  sig  <- terra::extract(fields_pred[[site_nm]]$sigma,         pts_sf)[,2]
  df   <- data.frame(
    site  = site_nm, 
    x     = st_coordinates(pts_sf)[,1],
    y     = st_coordinates(pts_sf)[,2],
    error = pts_sf[[true_field_name]] - hat,
    sigma = sig
  )
  pt_dfs[[i]] <- na.omit(df)
}

# Step2 
# Combine the results across fields into a single data frame:
# The do.call function applies the function rbind to results_list. The result, 
# error_data is a data frame with columns error,sigma x, and y.
error_data      <- do.call(rbind, pt_dfs)

# Step 3
# Compute the standardized prediction error (spe):
# The standardized prediction error is derived by dividing the prediction error 
# at each sampled location by the predictive standard deviation.
error_data$spe  <- error_data$error / error_data$sigma


# Step 4 
# Compute the empirical variogram of SPE
# Fit variograms to the standardized prediction error:
# This step computes the empirical variogram and fits exponential, spherical, 
# and nugget-only models
# Set cutoff + bin width you prefer. Experiment with various values to reach an
# ideal fit, see step 5.
cutoff <- 6000     
width  <- 400      
message("Variogram: cutoff=", cutoff, "m  width=", width, "m")

# Empirical variogram
vg_emp <- variogram(spe~1,
                    data      = error_data,
                    locations = ~ x + y,
                    cutoff    = cutoff,
                    width     = width)


# Step 5 
# Check for proper sampling of variogram
# See section 5.5.1 of Verra tool, Sampling Guidelines for Variogram Calculations

# diagnostics: how many bins before / after
n_before <- nrow(vg_emp)

# drop any lag with <50 pairs (per Webster & Oliver)
valid  <- vg_emp$np >= 50
vg_emp <- vg_emp[valid,]
n_after  <- nrow(vg_emp)

cat("  * Bins before filtering:", n_before, "\n")
cat("  * Bins after  filtering:", n_after, "\n")
cat("  * Bins removed         :", n_before - n_after, "\n")

# initial range guess (we used cutoff/3)
initial_range_guess <- cutoff/3
cat("  * First bin centre (m):", round(min(vg_emp$dist),1),
    "  –  should be < initial range guess (", round(cutoff/3,1),"m", ")\n")

# compute actual maximum pairwise distance
cat("  * Last bin centre  (m):", round(max(vg_emp$dist),1),
    "  –  should be no more than half the maximum inter‐point distance (",
    round(max(dist(error_data[,c("x","y")]))),"m)\n")


# Step 6 
# Define theoretical semivariograms
sph_variogram <- function(h, psill, range, nugget){
  ifelse(h <= range,
         nugget + psill*((3*h)/(2*range) - (h^3)/(2*range^3)),
         nugget + psill)
}
exp_variogram <- function(h, psill, range, nugget){
  nugget + psill*(1 - exp(-h/range))
}
gau_variogram <- function(h, psill, range, nugget){
  nugget + psill*(1 - exp(-(h/range)^2))
}

# Initial guesses from empirical:
# Nugget: The value at distance 0. The variability at scales smaller than your 
#         closest measurements (plus any measurement error).
# Sill:   Where the curve levels off. The overall variability you’d see between 
#         completely unrelated locations.
# Range:  The distance where it reaches the sill. Beyond this, knowing the 
#         value at one spot gives you almost no information about the value at 
#         another.

nug0   <- vg_emp$gamma[1]
sill0  <- max(vg_emp$gamma)
psill0 <- sill0 - nug0
range0 <- cutoff/3

ini_sph <- vgm(psill0, "Sph", range0, nug0)
ini_exp <- vgm(psill0, "Exp", range0, nug0)
ini_gau <- vgm(psill0, "Gau", range0, nug0)
ini_nug <- vgm(nug0,    "Nug", 0,      0)

# Step 7
# Fit candidate models
vgm_sph <- fit.variogram(vg_emp, ini_sph)
vgm_exp <- fit.variogram(vg_emp, ini_exp)
vgm_gau <- fit.variogram(vg_emp, ini_gau)
vgm_nug <- fit.variogram(vg_emp, ini_nug)

# Step 8
# Compute Sum of Squared Errors (SSE) for each
h_vec     <- vg_emp$dist
g_emp     <- vg_emp$gamma
SSE <- c(
  Sph = sum((sph_variogram(h_vec, 
                           vgm_sph[2,"psill"], 
                           vgm_sph[2,"range"], 
                           vgm_sph[1,"psill"]) - g_emp)^2),
  Exp = sum((exp_variogram(h_vec, 
                           vgm_exp[2,"psill"],
                           vgm_exp[2,"range"],
                           vgm_exp[1,"psill"]) - g_emp)^2),
  Gau = sum((gau_variogram(h_vec,
                           vgm_gau[2,"psill"], 
                           vgm_gau[2,"range"], 
                           vgm_gau[1,"psill"]) - g_emp)^2),
  Nug = sum((vgm_nug[1,"psill"] - g_emp)^2)
)
print(round(SSE,6))
best_SSE <- names(SSE)[which.min(SSE)]
cat("Best by SSE:", best_SSE, "\n")

# Step 9
#Plot empirical + all candidates
png(file.path(output_dir,"variogram_candidates.png"),width=800,height=800)
plot(h_vec, g_emp,
     pch=16, cex=1.5,
     xlab="Distance (m)", ylab="Semivariance",
     ylim=c(0,max(g_emp)*1.2),
     cex.lab=1.5, cex.axis=1.3)
h_seq <- seq(0, cutoff, length=200)
cols  <- viridis(4)
lines(h_seq, exp_variogram(h_seq, 
                           vgm_exp[2,"psill"], 
                           vgm_exp[2,"range"], 
                           vgm_exp[1,"psill"]),
      col=cols[1], lwd=3)
lines(h_seq, sph_variogram(h_seq, 
                           vgm_sph[2,"psill"], 
                           vgm_sph[2,"range"], 
                           vgm_sph[1,"psill"]),
      col=cols[2], lwd=3)
lines(h_seq, gau_variogram(h_seq, 
                           vgm_gau[2,"psill"], 
                           vgm_gau[2,"range"], 
                           vgm_gau[1,"psill"]),
      col=cols[3], lwd=3)
abline(h=vgm_nug[1,"psill"], col=cols[4], lwd=3)
legend("topleft", 
       legend=c("Exp","Sph","Gau","Nug"), 
       col=cols, lwd=3, cex=1.3, bty="n")
dev.off()

# Save the best variogram model for later use (preserves its gstatVariogram class)
vario_dir <- file.path(output_dir, "variogram_models")
if (!dir.exists(vario_dir)) dir.create(vario_dir, recursive = TRUE)

# pick the best fitted model by name
vgm_best <- switch(best_SSE,
                   Sph = vgm_sph,
                   Exp = vgm_exp,
                   Gau = vgm_gau,
                   Nug = vgm_nug
)

# write it out under a filename that includes the fit type
vario_file <- file.path(
  vario_dir,
  paste0("vgm_best_t1_", tolower(best_SSE), ".rds")
)
saveRDS(vgm_best, vario_file)



# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Section IV – Variance of the project‑mean stock (Wadoux & Heuvelink)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Uses the σ‑rasters ONLY (truth values are not needed here).
# Pairs points from different sites

# This section describes the methods originally developed by Wadoux and 
# Heuvelink (2023) to compute the variance of a spatial average.

set.seed(1)

# Step 1 – Select the σ‑SpatRasters and bind into one data frame (original style)
# Extract the σ SpatRaster object from each site’s prediction
sampled_fields_df <- lapply(fields_pred, function(site_list) {
    # 1. pull out the sigma SpatRaster
    ras <- site_list$sigma
    # 2. turn to data.frame, drop NAs, keep x,y coords
    df <- as.data.frame(ras, xy = TRUE, na.rm = TRUE)
    # 3. ensure the third column is always named “sigma”
    names(df)[3] <- "sigma"
    
    df
  })
print(sampled_fields_df)

# Combine the list into a single data frame by row-binding them together.
sampled_fields_df <- do.call(rbind, sampled_fields_df)

cat("Sec IV, Step 1: sigma_df dimensions:", nrow(sampled_fields_df), 
    "×", ncol(sampled_fields_df), "\n")
cat("Sec IV, Step 1: sigma summary:\n"); print(summary(sampled_fields_df$sigma))


# Step 2 – Generate random sample pairs.
# This is a Monte‑Carlo sample of pixel pairs

# number of pairs
n_pairs <- 1000000

# Sample n_pairs of locations by randomly drawing row numbers within 
# sampled_fields_df.
sample_indices <- matrix(
  sample(seq_len(nrow(sampled_fields_df)), 
         n_pairs * 2,        # two indices per “pair”
         replace = TRUE),
  ncol = 2
)

# sample_indices contains two columns, each of which represents a row in 
# sigma_df that defines one point in the point pair. 
# Note that this uses pixel centers as sample locations, which means that 
# sample locations cannot be closer together than the ground_sample_distance.


# Step 3
# Compute pairwise distances:

# Randomly sample rows from sampled_fields_df at the locations indicated by 
# sample_indices[,1]
locations_u <- sampled_fields_df[sample_indices[,1], ]

# Randomly sample rows from sampled_fields_df at the locations indicated by 
# sample_indices[,2]
locations_s <- sampled_fields_df[sample_indices[,2], ]

cat("Sec IV, Step 2: sampled", nrow(locations_u), "pairs of pixels\n")

# Calculate the  Euclidean distance between the two locations using their 
# coordinates.
distances   <- sqrt((locations_u$x - locations_s$x)^2 +
                      (locations_u$y - locations_s$y)^2)

cat("Sec IV, Step 3: distance summary:\n"); print(summary(distances))


# Step 4
# Compute the semivariance for each point pair using the spherical variogram:
# sph_variogram and exp_variogram are custom functions, defined above, to 
# compute the semivariance under a spherical or exponential variogram model. 
# gamma is the predicted semivariance for each pair of locations using the 
# spherical variogram model.

# Compute semivariance for *those same* sampled pairs
# Can use exp_variogram if exponential variogram is best.See plot from sec III
# UPDATE TO BE BEST VARIOGRAM MODEL
gamma <- exp_variogram(
  distances,
  psill  = vgm_best[2,2], # partial sill of the spherical component
  range  = vgm_best[2,3], # range parameter of the spherical component
  nugget = vgm_best[1,2] # the nugget effect
)
cat(" gamma summary:\n"); print(summary(gamma))


# Step 5
# Calculate the correlation function of the standardized prediction error:
# The variable rho is the correlation function, described in step 4 of section 4
# of Wadoux and Heuvelink (2023) and in Equation 10 of the main text of the tool.
# Compute the spatial correlation function by fitting a variogram to the standardized
# prediction errors and transforming the variogram’s predictions into a correlation
# function. 

rho <- (sum(vgm_best[,2]) - gamma) / sum(vgm_best[,2])
cat(" rho summary:\n"); print(summary(rho)); cat("\n")

# Step 6
# Compute the variance of spatial average:
# Select the predictive standard deviation value corresponding to the first point
# in each sample pair and then multiply by the same for the second point.
# Then multiply by rho ->  multiplies these predictive standard deviation values
# by the spatial correlation function. 
vars <- sampled_fields_df$sigma[sample_indices[,1]] * sampled_fields_df$sigma[sample_indices[,2]] * rho
cat(" vars summary:\n"); print(summary(vars)); cat("\n")


# The resulting quantity represents the variance of the mean SOC stock.
mean(vars)
# result for lesotho fields: 0.7036484



# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Section V – Variance of the mean stock change & uncertainty deduction
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Step 1, load info for time 1 and 2

# --------------------------------------------------------------
# Time 1 variables:

# Average project variance for all sites in Lesotho in time 1 
avg_var_t1 <- mean(vars)   # 0.6578385
cat("Average project variance for time 1:", avg_var_t1,"tC/ha\n")

# list of prediction rasters per site at time 1
prediction_rasters_t1 <- lapply(fields_pred, function(x) x$soc_stock_hat)

# list of uncertainty rasters per site at time 1
sigma_rasters_t1 <- lapply(fields_pred, function(x) x$sigma)



# project‑wide mean SOC at time 1
# (Eq.4 from tool)
# project‑wide mean SOC at time 1 (Eq.4 from tool)
mean_SOCst_t1 <- mean(
  sapply(prediction_rasters_t1, function(rast) {
    # terra::global returns a data.frame with one row/one column
    terra::global(rast, mean, na.rm = TRUE)[1,1]
  })
)
cat("Mean SOC_st at time 1:", round(mean_SOCst_t1, 3), "tC/ha\n")
# Mean SOC st at time 1: 44.07 tC/ha 

# The fitted variogram model for time 1
vgm_t1 <- vgm_best

print("The fitted variogram model for time 1:")
print(vgm_t1)
# model     psill    range
# 1   Nug 0.4781086   0.0000
# 2   Exp 0.4128955 909.1337

# Variable that has error and standardized prediction error (SPE)
error_samples_t1 <- error_data
mean(error_samples_t1$spe)
#-0.486994

# --------------------------------------------------------------
# Time 2 variables:

# Average project variance for all sites in Lesotho in time 2
avg_var_t2 <- 0.6578385   
cat("Average project variance for time 2:", avg_var_t2,"tC/ha\n")

# Define folder that holds all simulated rasters for time 2 (t+Δt)
site_rasters_t2_dir <- file.path(
  "D:/TerraCarbon/2025_projects/SoilOrganicCarbon/Uncertainty",
  "output/t2_from_t1")


# collect prediction rasters (pattern “_gstat_simulation.tif”) into a named list
sim_hat_files        <- list.files(site_rasters_t2_dir,
                                   pattern = "_soc_stock\\.tif$",
                                   full.names = TRUE)
site_names_t2_hat    <- sub("_.*$", "", basename(sim_hat_files))
# List of prediction rasters at time 2
prediction_rasters_t2 <- setNames(lapply(sim_hat_files, terra::rast),
                                  site_names_t2_hat)

# collect σ rasters (pattern “_uncertainty.tif”) into a named list
sim_sigma_files <- list.files(site_rasters_t2_dir,
                              pattern = "_soc_stock_uncertainty\\.tif$",
                              full.names = TRUE)
site_names_t2    <- sub("_.*$", "", basename(sim_sigma_files))
# list of uncertainty rasters per site at time 2
sigma_rasters_t2 <- setNames(lapply(sim_sigma_files, terra::rast),
                             site_names_t2)

# project‑wide mean SOC at time 2 (Eq.4 from tool)
mean_SOCst_t2 <- mean(
  sapply(prediction_rasters_t2, function(r) {
    terra::global(r, mean, na.rm = TRUE)[1,1]
  })
)
cat("Mean SOC_st at time 2:", round(mean_SOCst_t2, 3), "tC/ha\n\n")
# Mean SOC at time 2: 42.269  tC/ha

# Load spherical variogram fitted for time 2
# define where the model was saved in the T2 script
vgm_file_t2 <- file.path(output_dir,
                         "variogram_models",
                         "vgm_best_t2_exp.rds")
# read the variogram model
vgm_t2 <- readRDS(vgm_file_t2)

print("The fitted variogram model for time 2:")
print(vgm_t2)
# model    psill    range
# 1   Nug 0.5348275   0.0000
# 2   Exp 0.3903652 923.4804
class(vgm_t1)
class(vgm_t2)

#### Create SOC stock error points for time 2. Use t1 observation points
message("Extracting t2 prediction/σ at every observed point …")
pt_dfs2 <- vector("list", length = nrow(site_polys))
for(i in seq_len(nrow(site_polys))) {
  site_nm <- site_polys$site[i]
  # load & reproject the obs points
  pts_sf <- st_read(file.path(obs_point_dir, paste0(site_nm, "_points.shp")),
                    quiet=TRUE) |>
    st_transform(crs_project)
  
  # extract t2 prediction & σ
  hat2 <- terra::extract(prediction_rasters_t2[[site_nm]], pts_sf)[,2]
  sig2 <- terra::extract(sigma_rasters_t2[[site_nm]],     pts_sf)[,2]
  
  # assemble
  df2 <- data.frame(
    site   = site_nm,
    x      = st_coordinates(pts_sf)[,1],
    y      = st_coordinates(pts_sf)[,2],
    error2 = pts_sf[[true_field_name]] - hat2,
    sigma2 = sig2
  )
  pt_dfs2[[i]] <- na.omit(df2)
}

# Step2b – combine into one big data.frame
error_samples_t2 <- do.call(rbind, pt_dfs2)

# Step3b – compute standardized prediction error for time2
error_samples_t2$spe2 <- error_samples_t2$error2 / error_samples_t2$sigma2
mean(error_samples_t2$spe)
#--0.5194228


# ───────────────────────────────────────────────────────────────────
# STEP 2 – compute ΔSOĈ, its variance, and PoE deduction

# 1) Compute the observed mean change (Eq.4 in main text):
delta_SOC_hat <- mean_SOCst_t2 - mean_SOCst_t1
#    ΔSOĈ = mean_SOCst_t+Δt  – mean_SOCst_t
cat(sprintf("ΔSOĈ (Eq.4): %.3f tC/ha\n\n", delta_SOC_hat))

# 2) Estimate the correlation ρ between the two time‑point estimates
#    via Pearson correlation of the paired standardized prediction errors (SPE).
#    This approximates the zero‑lag cross‑covariance term in Eq.5.
err12 <- merge(
  error_samples_t1, error_samples_t2,
  by = c("site","x","y"),  # align points by site and coordinates
  all = FALSE
)
rho <- cor(
  err12$spe,   # SPE at time t
  err12$spe2,  # SPE at time t+Δt
  use = "pairwise.complete.obs"
)
cat(sprintf("Estimated ρ (corr of SPE): %.3f\n\n", rho))

# 3) Compute the covariance term in Equation 5:
#    Cov_term = 2 * ρ * σ_t * σ_{t+Δt},
#    where σ = sqrt(var of the mean) from Section V (avg_var_t1, avg_var_t2).
cov_term <- 2 * rho * sqrt(avg_var_t1) * sqrt(avg_var_t2)
cat(sprintf("Covariance term (2ρ·σₜ·σₜ₊Δₜ): %.4f (tC/ha)^2\n\n", cov_term))

# 4) Compute the variance of the mean stock change (Eq.5):
#    var(ΔSOĈ) = var_t1 + var_t2 – Cov_term
var_mean_change <- avg_var_t1 + avg_var_t2 - cov_term
cat(sprintf("Var(ΔSOĈ) (Eq.5): %.4f (tC/ha)^2\n\n", var_mean_change))

# 5) Convert SOC change statistics into CO₂ e units (Eqs.6 & 7):
# mw is the “molecular weight” conversion factor that converts a mass of carbon 
# (C) into the equivalent mass of carbon dioxide (CO₂).
mw        <- 44/12                # 1 tC = 44/12 tCO₂
mean_co2e <- delta_SOC_hat * mw   # mean removal in tCO₂/ha (Eq.6)
var_co2e  <- var_mean_change * mw^2  # variance in (tCO₂/ha)^2 (Eq.7)
sd_co2e   <- sqrt(var_co2e)       # standard deviation in tCO₂/ha
cat(sprintf("Mean removal (CO₂e): %.3f tCO₂/ha\n", mean_co2e))
cat(sprintf("Var removal  (CO₂e)^2: %.4f\n",    var_co2e))
cat(sprintf("SD removal   (CO₂e): %.3f\n\n",    sd_co2e))

# ───────────────────────────────────────────────────────────────────
# STEP 3 – Plot probability of Exceedance and deduction

#    Plot Probability of Exceedance:
#    We simulate distributions for 1, 2, 3 year monitoring periods,
#    then find the 33.3% quantile (P(O>E)=1/3) and % deduction.

png("probability_of_exceedance.png",
    res    = 150,
    width  = 1500,
    # Maintain aspect ratio; increase top title space by bumping first height
    height = ceiling(1500 * ((12/2.54)/(23/2.54))))  

# Layout: title row (1), three density panels (2,3,4), footer row (5)
mat <- matrix(c(
  1,1,1,
  2,3,4,
  5,5,5
), nrow = 3, byrow = TRUE)
# Give extra height to title row for breathing room
layout(mat,
       widths  = lcm(c(7,7,7)),
       heights = lcm(c(0.5, 8, 2)))  

# — Title panel: empty plot with main title only
par(
  mar = c(0,1,1,1),  # bottom, left, top, right margins around title
  adj = 0.5          # center‐align the title
)
plot(
  seq(1,10), seq(1,10),  # dummy x/y to set up empty plot
  col      = NA,         # no points
  axes     = FALSE,      # no axes
  xlab     = "", ylab    = "",
  main     = "Probability of exceedance",
  cex.main = 2           # large title font
)

# — Three density panels for 1, 2, and 3‑year monitoring
years         <- c(1, 2, 3)
per_year_rate <- delta_SOC_hat / 3     # spread total ΔSOĈ across 3 years

for (i in seq_along(years)) {
  yr     <- years[i]
  mean_y <- yr * per_year_rate * mw    # expected mean in tCO₂e/ha
  draws  <- rnorm(1e6, mean = mean_y, sd = sd_co2e)  
  dens   <- density(draws)             # estimate density
  
  # Expand top margin to fit labels; allow drawing into the margin
  par(
    mar = c(1, 2, 5, 1),  # bottom, left, top, right margins
    xpd = NA              # allow text in the margins
  )
  plot(
    dens, main = "", xlab = "", ylab = "",
    lwd  = 2,                         # thicker line
    axes = FALSE,                     # draw axes manually
    xlim = range(dens$x),             # full x‐range
    ylim = c(0, max(dens$y) * 1.2)    # 20% headroom above curve
  )
  axis(1); axis(2)                   # add bottom & left axes
  
  # 33.3% quantile line (vertical dashed)
  q333 <- quantile(draws, 0.333)

  # draw dashed line WITHIN panel only:
  usr <- par("usr")  # user coords: c(xmin,xmax,ymin,ymax)
  segments(
    x0 = q333, y0 = usr[3], 
    x1 = q333, y1 = usr[4],
    lty = 2
  )
  
  # Label the dashed line to explain its meaning
  text(
    x     = q333+0.15,
    y     = max(dens$y) * 0.5,      # just above the headroom
    labels= "33.3% quantile",        # explain the line
    srt   = 90,                      # rotate text vertical
    adj   = c(0,0.5),
    cex   = 0.8
  )
  
  # Compute deduction fraction
  deduct_pct <- 1 - q333 / mean_y
  
  # Compute creditable amounts after the deduction
  cred_tC_ha  <- delta_SOC_hat * (i / 3) * (1 - deduct_pct)
  cred_co2e_ha <- cred_tC_ha * mw
  
  # Print a summary line to the console
  cat(sprintf(
    "%d‑year period: unadjusted = %.3f tC/ha, deduction = %.1f%% → creditable = %.3f tC/ha (%.3f tCO2e/ha)\n",
    i,
    mean_y / mw * mw,       # just mean_y converted back to tC/ha
    deduct_pct * 100,
    cred_tC_ha,
    cred_co2e_ha
  ))
  
  # Position "X year" label 5% above the top of the curve
  text(
    x      = mean_y,
    y      = max(dens$y) * 1.05,
    labels = paste0(yr, " year"),
    cex    = 1.2
  )
  
  # Position "% deduct." label 15% above the top of the curve
  text(
    x      = mean_y,
    y      = max(dens$y) * 1.15,
    labels = paste0(round(deduct_pct * 100, 1), "% deduct."),
    cex    = 1.2
  )
}

# Footer panel: display the equation label with a plain‐language caption underneath
par(
  mar = c(0, 0, 0, 0),  # no margins
  xpd = NA              # allow text to go into the figure region
)
plot.new()             # start a blank plot

# the math expression
text(
  x      = 0.5, y = 0.6,     # place it in the upper half of this footer panel
  labels = expression(
    paste(
      "(",
      bar(Delta)[t], " → ",
      bar(Delta)[t + Delta * t], "   ",
      CO[2], "[soil]",
      ")"
    )
  ),
  cex = 1.5                  # moderate text size for the formula
)

# explanatory text beneath the formula
text(
  x      = 0.5, y = 0.35,    # below the formula
  labels = expression(
    "Removal amount in tonnes of " * CO[2] * 
      " equivalent per hectare (t " * CO[2] * "e/ha)"
  ),
  cex = 1.2                  # slightly smaller text
)

dev.off()
