The sample data is available at https://doi.org/10.5281/zenodo.24123723 
# ##############################################################################################################################
#  A hybrid Bayesian Machine Learning and Geostatistics method for Improved Spatial Prediction of Soil Heavy Metals
#   Bayesian Additive Regression Trees (BART) section -- complete analysis workflow
#creator Abiot Molla
#date November 2026
#  WHAT THIS SCRIPT DOES
#    1  Aligns the predictor rasters on one 200 m grid and builds the training table
#    2  Describes spatial structure (variograms) and builds spatial block CV folds;
#       quantifies information leakage from the interpolated soil layers
#    3  Defines the BART (dbarts) wrappers and the scoring functions
#    4  BART Model Calibration and Hyperparameter sensitivity analysis : 24 prior settings 
#    5  Nested CV: is tuning better than the default priors?
#    6  Predictive uncertainty: coverage, interval score, CRPS, calibration, PIT
#    7  Final 4-chain models, convergence (split R-hat) and posterior maps and publication map 
#    8  Variable relevance against a permutation null
#    9  Spatial structure of out-of-fold residuals (Moran's I, variograms)
#   10  Sensitivity of the CV results to the validation block size
#   11  Uncertainty summary figures
#   12  Observed vs predicted plots
#
#  HOW TO RUN
#    1  Put the input data in data/ (repository folder).
#    2  Install the necessary packages
#    3  Set the working directory to the repository folder and run
#       Set RUN_MODE <- "quick".
#
#  REPRODUCIBILITY
#    * Every random step has a fixed seed; parallel jobs carry their own seeds, so
#      results do not depend on the number of cores.
#    * Long steps are cached as .rds files in the results folder. Delete a cache file
#      to recompute that step. 
#    * Developed with R 4.4.1, dbarts 0.9-32, terra 1.8-86, sf 1.0-24, gstat 2.1-4.
#
#  NOTE  terra is always called as terra:: because dbarts masks terra::extract().


suppressPackageStartupMessages({
  library(dbarts)                         # Chipman etal (2010)
  library(dplyr); library(readr); library(ggplot2)
  library(sf); library(gstat); library(parallel)
})
# CSV files 
#write_csv <- readr::write_excel_csv
#t0 <- Sys.time()
#section <- function(title) message(sprintf("\n[%s | %5.1f min] %s", format(Sys.time(), "%H:%M"),
                                           #as.numeric(difftime(Sys.time(), t0, units = "mins")), title))


# 0. CONFIGURATION
####################################################################################################################################
# Run mode 
# "full"  : settings used in the manuscript
# "quick" : 2 CV repeats and short MCMC chains, written to results_quick/ (code check only;
#           the numbers are NOT the published results)
RUN_MODE <- "full"

# ---- project directors, Files, output directors and folders 
proj_dir   <- "C:/Xiamen/XiamenBART_EBKR_Workflow" # project folder
sample_csv <- file.path(proj_dir, "Soil Samples and extract predictors values.csv")
raster_dir <- file.path(proj_dir, "Predictors")   # one GeoTIFF per predictor
out_dir    <- file.path(proj_dir, if (RUN_MODE == "quick") "results_quick" else "results")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
op     <- function(f) file.path(out_dir, f)                # path of an output file
cached <- function(f, expr) if (file.exists(op(f))) readRDS(op(f)) else { x <- expr; saveRDS(x, op(f)); x }

## ---- Study area, responses and predictors 
EPSG    <- "EPSG:32650"                                     # WGS 84 / UTM zone 50N (metres)
METALS  <- c("As", "Cd", "Pb", "Cu", "Zn")# soil heavy metals analysed from 78 soil samples 
# Soil layers interpolated (EBK, ArcGIS) from the same samples: their raster values at the 78 soil sampling, leaks information in cross-validation (Section 2)
INTERP  <- c("TN", "TP", "TK", "OC", "pH")
# Rasters NOT used as predictors (names matched ignoring case):
#   OM = 1.724 x OC (redundant)(removed from the final predictor set)
DROP    <- c("OM")
CATEGORICAL <- c("LULC", "Texture", "SoilType")             
ASPECT  <- c("Northness", "Eastness")                       
MAX_SNAP <- 400                                             
# ---- Response scale 
TRANSFORM <- "log"

## ---- Predictor sets used in cross-validation 
# all         : all predictors, soil layers as extracted from the rasters (leaky upper bound)
# infold      : at test sites the soil layers are re-kriged from the training samples only
# oof         : as infold, and at training sites the soil layers are also out-of-fold kriged, so the
#               model is trained on the same kind of input it receives at test sites and map cells
#               (MAIN ANALYSIS)
# independent : soil layers removed
EVAL_SET   <- "oof"     # set used for single-set summaries (residuals, block size, figures)
MAP_SET    <- "oof"     # set of the final mapping model: "oof", "independent" or "all"
OOF_SCHEME <- "loo"     # out-of-fold soil values at training sites: "loo" = leave-one-out kriging,
                        # "block" = kriged with the site's own block left out (stricter)
OOF_BLOCK_KM  <- 5
INTERP_METHOD <- "ok"   # in-fold re-interpolation: "ok" ordinary kriging 

## ---- Cross-validation design 
DESIGNS    <- c("block5km", "block10km")    # spatial block CV: primary design, sensitivity design
DIAG_SIZES <- c(2.5, 5, 10, 15)             # km: block sizes compared in the design table and Section 10
K_FOLD     <- 5
N_REP      <- 10                            # repeats of the outer CV (new random block grid each time)
N_REP_NESTED <- 5                           # repeats of the nested CV (Section 5)

## ---- MCMC settings 
MCMC_CV    <- list(nskip = 500L,  ndpost = 1000L)              # outer CV fits
MCMC_INNER <- list(nskip = 250L,  ndpost = 500L)               # inner (tuning) fits of the nested CV
MCMC_FINAL <- list(nskip = 1000L, ndpost = 1000L, nchain = 4L) # mapping models
N_PERM     <- 100                                              # permutations for variable relevance
N_CORES    <- max(1L, detectCores() - 1L)

if (RUN_MODE == "quick") {
  N_REP <- 2; N_REP_NESTED <- 1; N_PERM <- 10
  MCMC_CV <- list(nskip = 100L, ndpost = 200L); MCMC_INNER <- list(nskip = 50L, ndpost = 100L)
  MCMC_FINAL <- list(nskip = 200L, ndpost = 200L, nchain = 2L)
}

## ---- BART priors (Chipman et al, 2010) 
# Tree prior: alpha = 0.95, beta = 2 (fixed). Grid: number of trees m, leaf prior k, error SD prior.
SIG <- data.frame(prior    = c("default", "aggressive", "conservative"),
                  sigdf    = c(3,         3,            10),            # nu
                  sigquant = c(0.90,      0.99,         0.75))          # q
GRID <- merge(expand.grid(ntree = c(50L, 200L), k = c(1, 2, 3, 5)), SIG, by = NULL)   # 24 settings
GRID$combo <- with(GRID, sprintf("m%d_k%g_%s", ntree, k, prior))
DEFAULT <- "m200_k2_default"; G_DEF <- which(GRID$combo == DEFAULT)

## ---- Derived settings 
stopifnot(MAP_SET %in% c("oof", "independent", "all"), EVAL_SET %in% c("oof", "infold", "independent", "all"),
          TRANSFORM %in% c("log", "none"))
TF <- if (TRANSFORM == "log") log else identity       # response transform
BT <- if (TRANSFORM == "log") exp else identity       # back-transform to mg/kg
SCALE_LAB <- if (TRANSFORM == "log") "log scale" else "mg/kg"
TAG_BASE  <- paste(c("db", "infold", DESIGNS, TRANSFORM), collapse = "_")   # prefix of the cache names
DES_NAME  <- function(d) paste(sub("^block([0-9.]+)km$", "\\1", d), "km blocks")  # "block5km" -> "5 km blocks"
PAL_M   <- c(As = "#0072B2", Cd = "#E69F00", Pb = "#009E73", Cu = "#CC79A7", Zn = "#56B4E9")  # Okabe-Ito
SHAPE_M <- c(As = 16, Cd = 17, Pb = 15, Cu = 18, Zn = 8)
swap_sig <- function(x) gsub("@@", "aggressive", gsub("aggressive", "conservative", gsub("conservative", "@@", x)))
fix_sig_labels <- function(x) {
  if (isTRUE(attr(x, "sig_labels_v1"))) return(x)
  fx <- function(d) { if ("combo" %in% names(d)) d$combo <- swap_sig(d$combo); d }
  x <- if (is.data.frame(x)) fx(x) else lapply(x, fx)
  attr(x, "sig_labels_v1") <- TRUE; x
}
#####################################################################################################################################


# 1. PREDICTORS ON ONE GRID AND TRAINING DATA
######################################################################################################################################
section("1. Predictors and training data")
f <- list.files(raster_dir, "\\.tif$", full.names = TRUE, ignore.case = TRUE)
names(f) <- tools::file_path_sans_ext(basename(f))
if (!all(c("OC", "Aspect") %in% names(f)))
  stop("OC.tif and Aspect.tif not found in raster_dir = ", normalizePath(raster_dir, mustWork = FALSE),
       "\n  Files found: ", if (length(f)) paste(basename(f), collapse = ", ") else "none",
       "\n  Set proj_dir, sample_csv and raster_dir in Section 0 (working directory: ", getwd(), ").")
tmpl <- terra::rast(terra::rast(f["OC"]))                       # template: 200 m grid of the OC raster
asp  <- terra::rast(f["Aspect"])
lyr  <- lapply(setNames(setdiff(names(f), "Aspect"), setdiff(names(f), "Aspect")), function(nm)
  terra::resample(terra::rast(f[nm]), tmpl, method = if (nm %in% CATEGORICAL) "near" else "bilinear"))
lyr$Northness <- terra::resample(terra::ifel(asp < 0, 0, cos(asp * pi / 180)), tmpl, "bilinear")  # flat (-1) -> 0
lyr$Eastness  <- terra::resample(terra::ifel(asp < 0, 0, sin(asp * pi / 180)), tmpl, "bilinear")
stk_all <- terra::rast(lyr); names(stk_all) <- names(lyr)
dom_all <- sum(is.na(stk_all)) == 0                             # cells with every layer present

PRED    <- names(stk_all)[!tolower(names(stk_all)) %in% tolower(DROP)]
FACTORS <- intersect(CATEGORICAL, PRED)                         # categorical predictors in use (none by default)
message("Predictors (", length(PRED), "): ", paste(PRED, collapse = ", "),
        "\nNot used: ", paste(setdiff(names(stk_all), PRED), collapse = ", "))
# Short check sum of the predictor list, added to the cache names
PRED_ID <- paste0("p", length(PRED), "_", sum(utf8ToInt(paste(sort(PRED), collapse = "|"))) %% 100000)
TAG     <- paste(TAG_BASE, PRED_ID, sep = "_")
writeLines(PRED, op(paste0("predictor_list_", PRED_ID, ".txt")))
stk <- stk_all[[PRED]]
terra::writeRaster(stk, op("predictors.tif"), overwrite = TRUE)

# Sample table: NUM (ID), Lon/Lat (UTM 50N easting/northing in metres), metals (mg/kg), measured soil
# properties. Predictor values are extracted at the sites; NoData values are filled from the nearest
# complete cell within MAX_SNAP m; samples still incomplete are dropped.
smp <- read_csv(sample_csv, show_col_types = FALSE)
need <- c("NUM", "Lon", "Lat", METALS, INTERP)
if (length(miss <- setdiff(need, names(smp)))) stop("Missing columns in ", sample_csv, ": ", paste(miss, collapse = ", "))
pts <- terra::vect(as.data.frame(smp), geom = c("Lon", "Lat"), crs = EPSG)
ex  <- terra::extract(stk_all, pts, ID = FALSE)
na_rows <- which(!complete.cases(ex[, PRED]))
if (length(na_rows)) {
  dom_pts <- terra::as.points(terra::classify(dom_all, cbind(0, NA)))
  nr   <- terra::nearest(pts[na_rows], dom_pts)
  fill <- terra::extract(stk_all, dom_pts[nr$to_id], ID = FALSE)
  for (k in seq_along(na_rows)) if (nr$distance[k] <= MAX_SNAP) {
    miss <- PRED[is.na(unlist(ex[na_rows[k], PRED]))]
    ex[na_rows[k], miss] <- fill[k, miss]
  }
}
md <- bind_cols(smp[, c("NUM", "Lon", "Lat", METALS)], ex[, PRED],
                setNames(smp[, INTERP], paste0("lab_", INTERP)))   # measured values, used for in-fold kriging
dropped <- md$NUM[!complete.cases(md[, PRED])]
md <- md[complete.cases(md[, PRED]), ]
message("Samples used: ", nrow(md), if (length(dropped)) paste0(" (dropped, no predictor data: NUM ",
                                                                 paste(dropped, collapse = ", "), ")"))
LEVELS <- lapply(setNames(FACTORS, FACTORS), function(v) sort(unique(terra::values(stk[[v]], na.rm = TRUE)[, 1])))
for (v in FACTORS) md[[v]] <- factor(md[[v]], levels = LEVELS[[v]])
write_csv(md, op("model_data.csv"))

# Numeric design matrix (full indicator coding for factors, if any)
make_X <- function(df, vars) {
  df <- as.data.frame(df[, vars]); fac <- vars[sapply(df, is.factor)]
  ca <- if (length(fac)) setNames(lapply(fac, function(v) contrasts(df[[v]], contrasts = FALSE)), fac) else NULL
  X <- model.matrix(~ . - 1, df, contrasts.arg = ca)
  attr(X, "assign") <- NULL; attr(X, "contrasts") <- NULL; X
}
PRED_SETS <- list(all = PRED, infold = PRED, oof = PRED, independent = setdiff(PRED, INTERP))
SET_MODE  <- c(all = "none", infold = "infold", oof = "oof", independent = "none")
X_list <- lapply(PRED_SETS, function(v) make_X(md, v))
Y   <- sapply(METALS, function(m) TF(md[[m]]))                  # responses on the modelling scale
NUM <- md$NUM; n <- nrow(md)
LAB <- as.matrix(md[, paste0("lab_", INTERP)]); colnames(LAB) <- INTERP

# Prediction cells and applicability: a cell is "applicable" when every continuous mapping predictor
# lies inside the range seen at the samples (aspect excluded)
MAP_PRED <- PRED_SETS[[MAP_SET]]
MAP_TAG  <- paste(MAP_SET, TRANSFORM, sep = "_")                # suffix of all map outputs
vals  <- terra::values(stk[[MAP_PRED]], dataframe = TRUE)
cells <- which(complete.cases(vals))
newdf <- vals[cells, , drop = FALSE]
for (v in FACTORS) newdf[[v]] <- factor(newdf[[v]], levels = LEVELS[[v]])
X_new <- make_X(newdf, MAP_PRED); stopifnot(identical(colnames(X_new), colnames(X_list[[MAP_SET]])))
app <- Reduce(`&`, lapply(setdiff(MAP_PRED, c(FACTORS, ASPECT)), function(v)
  newdf[[v]] >= min(md[[v]]) & newdf[[v]] <= max(md[[v]])))
if (length(FACTORS)) app <- app & Reduce(`&`, lapply(FACTORS, function(v) newdf[[v]] %in% unique(md[[v]])))
to_rast <- function(mat, nm) {                                  # cell values -> multi-band raster
  full <- matrix(NA_real_, terra::ncell(stk), ncol(mat)); full[cells, ] <- mat
  r <- terra::setValues(terra::rast(stk[[1]], nlyrs = ncol(mat)), full); names(r) <- nm; r
}
app_r <- to_rast(cbind(as.numeric(app)), "applicable")
terra::writeRaster(app_r, op(paste0("applicability_", MAP_SET, ".tif")), overwrite = TRUE, datatype = "INT1U")
########################################################################################################################################


# 2. SPATIAL STRUCTURE, CROSS-VALIDATION DESIGN AND LEAKAGE ANALYSIS
##########################################################################################################################################
section("2. Variograms, CV folds and leakage")
xy <- as.matrix(md[, c("Lon", "Lat")]); D <- as.matrix(dist(xy)); cutoff <- max(D) / 2

# 2a. Variogram helpers 
# Fitted on standardized values with coordinates in km (raw semivariances such as Cd ~1e-3 against
# distances in meters are badly scaled for fit.variogram()).
CUT_KM <- cutoff / 1000
to_km_sf <- function(d, col) st_as_sf(data.frame(x = d$Lon / 1000, y = d$Lat / 1000,
                                                 z = (d[[col]] - mean(d[[col]])) / sd(d[[col]])), coords = c("x", "y"))
fit_vg <- function(ev, s2 = 1, co = CUT_KM) {                   # best of exponential / spherical, or NULL
  best <- NULL; sse <- Inf
  for (mt in c("Exp", "Sph")) {
    f <- tryCatch(suppressWarnings(fit.variogram(ev, vgm(0.7 * s2, mt, co / 3, 0.3 * s2))), error = function(e) NULL)
    e <- if (is.null(f) || isTRUE(attr(f, "singular"))) NA else attr(f, "SSErr")
    if (length(e) == 1 && is.finite(e) && e < sse) { best <- f; sse <- attr(f, "SSErr") }
  }
  best
}
describe_vg <- function(fit, co = CUT_KM) {                     # model, nugget ratio, practical range (m)
  if (is.null(fit)) return(data.frame(model = NA, nugget_ratio = NA, range_m = NA, status = "not converged"))
  st <- fit[fit$model != "Nug" & fit$psill > 0 & fit$range > 0, ]
  nr <- sum(fit$psill[fit$model == "Nug"]) / sum(fit$psill)
  if (!nrow(st)) return(data.frame(model = "Nug", nugget_ratio = 1, range_m = NA, status = "pure nugget"))
  rg <- if (st$model[1] == "Exp") 3 * st$range[1] else st$range[1]
  data.frame(model = as.character(st$model[1]), nugget_ratio = nr, range_m = round(rg * 1000),
             status = if (rg > co) "range beyond cutoff" else "structured")
}

## ---2b. Leakage-free values of the interpolated soil layers 
# interp_one: values at test sites te kriged from the MEASURED values at training sites tr only
interp_one <- function(z, tr, te) {
  mu <- mean(z[tr]); s <- sd(z[tr])
  s_tr <- st_as_sf(data.frame(x = xy[tr, 1] / 1000, y = xy[tr, 2] / 1000, z = (z[tr] - mu) / s), coords = c("x", "y"))
  s_te <- st_as_sf(data.frame(x = xy[te, 1] / 1000, y = xy[te, 2] / 1000), coords = c("x", "y"))
  fit <- if (INTERP_METHOD == "ok") fit_vg(variogram(z ~ 1, s_tr, cutoff = CUT_KM, width = 2)) else NULL
  p <- if (!is.null(fit)) krige(z ~ 1, s_tr, s_te, model = fit, debug.level = 0)$var1.pred
       else idw(z ~ 1, s_tr, s_te, idp = 2, debug.level = 0)$var1.pred
  mu + s * p
}
# interp_loo: out-of-fold value at every site in idx, kriged from the OTHER sites in idx only
interp_loo <- function(z, idx) {
  if (OOF_SCHEME == "block") {
    inner <- block_folds(xy[idx, , drop = FALSE], OOF_BLOCK_KM * 1000, K_FOLD, 1L); p <- numeric(length(idx))
    for (f in unique(inner)) p[inner == f] <- interp_one(z, idx[inner != f], idx[inner == f])
    return(p)
  }
  mu <- mean(z[idx]); s <- sd(z[idx])
  s_tr <- st_as_sf(data.frame(x = xy[idx, 1] / 1000, y = xy[idx, 2] / 1000, z = (z[idx] - mu) / s), coords = c("x", "y"))
  fit <- if (INTERP_METHOD == "ok") fit_vg(variogram(z ~ 1, s_tr, cutoff = CUT_KM, width = 2)) else NULL
  p <- suppressWarnings(krige.cv(z ~ 1, s_tr, model = fit, nfold = length(idx), verbose = FALSE,
                                 debug.level = 0)$var1.pred)        # model = NULL -> inverse distance (idp 2)
  mu + s * p
}
infold_X <- function(Xte, tr, te) { for (v in INTERP) Xte[, v] <- interp_one(LAB[, v], tr, te); Xte }
oof_X    <- function(Xtr, idx)    { for (v in INTERP) Xtr[, v] <- interp_loo(LAB[, v], idx); Xtr }
# Training and test design matrices of one fold; `pre` = precomputed interpolated values (FOLD_INTERP)
fold_X <- function(X, mode, tr, te, pre = NULL) {
  Xtr <- X[tr, , drop = FALSE]; Xte <- X[te, , drop = FALSE]
  if (mode %in% c("infold", "oof")) {
    if (!is.null(pre)) Xte[, INTERP] <- pre$te else Xte <- infold_X(Xte, tr, te) }
  if (mode == "oof") {
    if (!is.null(pre)) Xtr[, INTERP] <- pre$tr else Xtr <- oof_X(Xtr, tr) }
  list(tr = Xtr, te = Xte)
}

## ---- 2c. Distribution and spatial structure of the responses 
skew <- function(x) mean((x - mean(x))^3) / sd(x)^3
write_csv(bind_rows(lapply(METALS, function(m) data.frame(
  metal = m, skew_raw = skew(md[[m]]), shapiro_p_raw = shapiro.test(md[[m]])$p.value,
  skew_log = skew(log(md[[m]])), shapiro_p_log = shapiro.test(log(md[[m]]))$p.value))),
  op("response_normality.csv"))
variograms <- function(df, col) bind_rows(lapply(METALS, function(m) {
  ev <- variogram(z ~ 1, to_km_sf(df[df$metal == m, ], col), cutoff = CUT_KM, width = 2)
  cbind(metal = m, describe_vg(fit_vg(ev)))
}))
long_y <- bind_rows(lapply(METALS, function(m) data.frame(metal = m, Lon = md$Lon, Lat = md$Lat, y = Y[, m])))
write_csv(variograms(long_y, "y"), op("variogram_metals.csv"))

## ---- 2d. Spatial block folds 
# Square blocks on a randomly offset grid; whole blocks go to folds
# currently smallest fold (balanced fold sizes). A new offset is drawn for every repeat.
block_m <- function(design) as.numeric(sub("^block([0-9.]+)km$", "\\1", design)) * 1000
block_folds <- function(xy, size, k, seed) {
  set.seed(seed)
  off <- runif(2, 0, size)
  bid <- paste(floor((xy[, 1] - off[1]) / size), floor((xy[, 2] - off[2]) / size))
  blk <- unique(bid); cnt <- as.numeric(table(bid)[blk]); k <- min(k, length(blk))
  load <- numeric(k); fb <- integer(length(blk))
  for (i in order(-cnt, runif(length(blk)))) {
    f <- which(load == min(load)); f <- f[sample.int(length(f), 1)]
    fb[i] <- f; load[f] <- load[f] + cnt[i]
  }
  fb[match(bid, blk)]
}
make_folds <- function(design, seed) {
  if (design == "random_k5") { set.seed(seed); return(sample(rep(1:K_FOLD, length.out = n))) }
  block_folds(xy, block_m(design), K_FOLD, seed)
}
CAND <- c("random_k5", paste0("block", DIAG_SIZES, "km"))       # candidate designs
stopifnot(all(DESIGNS %in% CAND))
FOLDS <- lapply(setNames(CAND, CAND), function(d) sapply(1000 + seq_len(N_REP), function(s) make_folds(d, s)))

## ---- 2e. Design table: CV distances vs mapping distances (Milà et al., 2022) 
set.seed(42)
pc <- terra::spatSample(app_r, 20000, "random", na.rm = TRUE, xy = TRUE)
pc <- as.matrix(pc[pc$applicable == 1, c("x", "y")])
pred_nnd <- apply(pc, 1, function(p) min(sqrt((xy[, 1] - p[1])^2 + (xy[, 2] - p[2])^2)))   # map cell -> nearest sample
cv_nnd <- function(fold) unlist(lapply(unique(fold), function(f) apply(D[fold == f, fold != f, drop = FALSE], 1, min)))
design_tab <- bind_rows(lapply(CAND, function(d) bind_rows(lapply(seq_len(N_REP), function(r) {
  fo <- FOLDS[[d]][, r]; nn <- cv_nnd(fo)
  data.frame(design = d, rep = r, median_nnd = median(nn), KS = suppressWarnings(ks.test(nn, pred_nnd)$statistic),
             min_fold = min(table(fo)), max_fold = max(table(fo)))
})))) |> group_by(design) |>
  summarise(median_cv_nnd_m = mean(median_nnd), mean_KS = mean(KS), min_fold = min(min_fold),
            max_fold = max(max_fold), .groups = "drop") |>
  mutate(median_pred_nnd_m = median(pred_nnd))
write_csv(design_tab, op("cv_design.csv"))                     # random_k5: reference row only, not used for modelling
for (d in DESIGNS) write_csv(bind_cols(NUM = NUM, as.data.frame(FOLDS[[d]])), op(paste0("folds_", d, ".csv")))
ggsave(op("fig_block_folds.png"), width = 11, height = 5.5, dpi = 300,
       ggplot(bind_rows(lapply(DESIGNS, function(d) data.frame(design = d, Lon = xy[, 1], Lat = xy[, 2],
                                                               fold = factor(FOLDS[[d]][, 1])))),
              aes(Lon, Lat, colour = fold, shape = fold)) + geom_point(size = 2.2) + coord_equal() +
         facet_wrap(~ factor(design, DESIGNS), labeller = as_labeller(DES_NAME)) +
         scale_colour_manual(values = unname(PAL_M)) +
         labs(x = "Easting (m)", y = "Northing (m)", colour = "Fold", shape = "Fold") + theme_bw())

## ---- 2f. Interpolated soil layers for every outer fold (computed once, cached) 
# te = held-out sites kriged from the training sites; tr = training sites, out-of-fold within training
FOLD_INTERP <- cached(paste0("fold_interp_", OOF_SCHEME, "_", paste(DESIGNS, collapse = "_"), ".rds"),
  lapply(setNames(DESIGNS, DESIGNS), function(d) lapply(seq_len(N_REP), function(r) {
    fo <- FOLDS[[d]][, r]
    setNames(lapply(sort(unique(fo)), function(f) { te <- which(fo == f); tr <- which(fo != f)
      list(tr = sapply(INTERP, function(v) interp_loo(LAB[, v], tr)),
           te = sapply(INTERP, function(v) interp_one(LAB[, v], tr, te))) }), sort(unique(fo)))
  })))

## ---- 2g. Leakage table 
# R2 against the measured values of: the raster values at the sites ("all"), in-fold kriged values at
# held-out sites (test input of "infold"/"oof") and out-of-fold kriged values at training sites
# (training input of "oof")
r2 <- function(o, p) 1 - sum((o - p)^2) / sum((o - mean(o))^2)
leak_tab <- bind_rows(lapply(DESIGNS, function(d) bind_rows(lapply(seq_len(N_REP), function(r) {
  fo <- FOLDS[[d]][, r]; est <- matrix(NA_real_, n, length(INTERP), dimnames = list(NULL, INTERP))
  r2_tr <- matrix(NA_real_, 0, length(INTERP))
  for (f in sort(unique(fo))) { te <- which(fo == f); tr <- which(fo != f); pre <- FOLD_INTERP[[d]][[r]][[as.character(f)]]
    est[te, ] <- pre$te
    r2_tr <- rbind(r2_tr, sapply(seq_along(INTERP), function(j) r2(LAB[tr, j], pre$tr[, j]))) }
  data.frame(design = d, rep = r, variable = INTERP,
             R2_raster_at_site = sapply(INTERP, function(v) r2(LAB[, v], md[[v]])),
             R2_infold_interp  = sapply(INTERP, function(v) r2(LAB[, v], est[, v])),
             R2_oof_train      = colMeans(r2_tr))
})))) |> group_by(design, variable) |>
  summarise(R2_infold_sd = sd(R2_infold_interp), R2_raster_at_site = mean(R2_raster_at_site),
            R2_infold_interp = mean(R2_infold_interp), R2_oof_train = mean(R2_oof_train),
            .groups = "drop")                                   # sd computed before the mean overwrites the column
write_csv(leak_tab, op("table_leakage_interpolated_layers.csv"))


# ###########################################################################################################################
# 3. BART WRAPPERS AND SCORING FUNCTIONS
#############################################################################################################################
section("3. BART helpers")
# X is the numeric design matrix (column order fixed by make_X). Priors: base (alpha) = 0.95,
# power (beta) = 2, plus k, ntree and the sigma prior (sigdf = nu, sigquant = q) from one GRID row h.
# sp = per-column split probabilities (NULL = uniform).
db_args <- function(Xtr, ytr, h, mcmc, seed, sp, keep, Xte = NULL) {
  a <- list(x.train = Xtr, y.train = ytr, ntree = as.integer(h$ntree), k = h$k, power = 2, base = 0.95,
            sigdf = h$sigdf, sigquant = h$sigquant, nskip = mcmc$nskip, ndpost = mcmc$ndpost,
            keeptrees = keep, verbose = FALSE, nchain = 1L, nthread = 1L, seed = as.integer(seed))
  if (!is.null(Xte)) a$x.test <- Xte
  if (!is.null(sp)) a$splitprobs <- sp
  a
}
# bart_fit   : fit and keep the trees (for later prediction on map cells)
# bart_draws : posterior draws of f(x), draws x n_new
# bart_sigma : posterior draws of the error SD
# fit_bart   : fit and predict test rows in one pass; returns ev (draws x n_test) and sigma
bart_fit    <- function(Xtr, ytr, h, mcmc, seed, sp = NULL) do.call(dbarts::bart, db_args(Xtr, ytr, h, mcmc, seed, sp, TRUE))
bart_draws  <- function(fit, Xnew) predict(fit, newdata = Xnew, type = "ev")
bart_sigma  <- function(fit, ndpost) tail(as.vector(fit$sigma), ndpost)
split_share <- function(fit) { vc <- fit$varcount; colMeans(vc / pmax(rowSums(vc), 1)) }   # mean split share per column
fit_bart <- function(Xtr, ytr, Xte, h, mcmc, seed, sp = NULL) {
  fit <- do.call(dbarts::bart, db_args(Xtr, ytr, h, mcmc, seed, sp, FALSE, Xte))
  list(ev = matrix(fit$yhat.test, nrow = mcmc$ndpost), sigma = tail(as.vector(fit$sigma), mcmc$ndpost))
}

# Posterior predictive summaries at test sites (modelling scale): draws = f(x) + N(0, sigma)
QP <- c(0.025, 0.05, 0.10, 0.25, 0.5, 0.75, 0.90, 0.95, 0.975)
summ_ppd <- function(ev, sigma, y, crps = FALSE) {
  ppd <- ev + matrix(rnorm(length(ev)), nrow(ev)) * sigma
  q <- apply(ppd, 2, quantile, QP)
  data.frame(y_obs = y, y_mean = colMeans(ev), y_median = q[5, ], sd_f = apply(ev, 2, sd),
             sd_pred = apply(ppd, 2, sd), pit = colMeans(sweep(ppd, 2, y, "<=")), pred_mean = colMeans(BT(ppd)),
             lo50 = q[4, ], hi50 = q[6, ], lo80 = q[3, ], hi80 = q[7, ], lo90 = q[2, ], hi90 = q[8, ],
             lo95 = q[1, ], hi95 = q[9, ],
             crps = if (crps) sapply(seq_along(y), function(i) {          # sample-based CRPS estimator
               x <- ppd[, i]; mean(abs(x - y[i])) - 0.5 * mean(abs(x - sample(x))) }) else NA_real_)
}

# Null model: Gaussian with the training-fold mean and SD x sqrt(1 + 1/n) (reference for CRPSS)
crps_norm <- function(y, mu, s) { z <- (y - mu) / s; s * (z * (2 * pnorm(z) - 1) + 2 * dnorm(z) - 1 / sqrt(pi)) }
null_fold <- function(d, r, m, fold, f) {
  te <- which(fold == f); tr <- which(fold != f); y <- Y[, m]
  mu <- mean(y[tr]); s <- sd(y[tr]) * sqrt(1 + 1 / length(tr)); q <- qnorm(QP[-5], mu, s)
  data.frame(design = d, rep = r, metal = m, combo = "NULL", fold = f, NUM = NUM[te], y_obs = y[te],
             y_mean = mu, y_median = mu, sd_f = NA_real_, sd_pred = s, pit = pnorm(y[te], mu, s),
             pred_mean = if (TRANSFORM == "log") exp(mu + s^2 / 2) else mu, lo50 = q[4], hi50 = q[5], lo80 = q[3], hi80 = q[6],
             lo90 = q[2], hi90 = q[7], lo95 = q[1], hi95 = q[8], crps = crps_norm(y[te], mu, s))
}
null_cv <- function(designs, reps, sets = names(PRED_SETS)) {
  g <- expand.grid(d = designs, r = reps, m = METALS, stringsAsFactors = FALSE)
  nul <- bind_rows(lapply(seq_len(nrow(g)), function(i) {
    fo <- FOLDS[[g$d[i]]][, g$r[i]]
    bind_rows(lapply(sort(unique(fo)), function(f) null_fold(g$d[i], g$r[i], g$m[i], fo, f)))
  }))
  bind_rows(lapply(sets, function(s) mutate(nul, set = s)))
}

# Cached CV results by predictor set: sets already in the cache are reused, missing sets are run and appended
cached_sets <- function(f, run, sets = names(PRED_SETS)) {
  old <- if (file.exists(op(f))) fix_sig_labels(readRDS(op(f))) else NULL
  have <- if (is.null(old)) character(0) else unique((if (is.data.frame(old)) old else old[[1]])$set)
  todo <- setdiff(sets, have)
  if (length(todo)) {
    new <- run(todo)
    old <- if (is.null(old)) new else if (is.data.frame(old)) bind_rows(old, new) else
      setNames(lapply(names(old), function(k) bind_rows(old[[k]], new[[k]])), names(old))
    attr(old, "sig_labels_v1") <- TRUE
    saveRDS(old, op(f))
  }
  old
}

# Scores of one set of CV predictions.
#   RMSE, MAE, Bias, R2 : mg/kg (back-transformed posterior median)
#   RMSE_model, R2_model: modelling scale (log scale when TRANSFORM = "log"; reported as R2 in the paper)
#   cov50..cov95        : empirical coverage of central prediction intervals
#   width90             : mean 90% interval width (mg/kg); IS90: interval score (modelling scale)
#   CRPS                : continuous ranked probability score; rho_sd_err: Spearman(SD, |error|)
score <- function(d) {
  o <- BT(d$y_obs); p <- BT(d$y_median); e <- d$y_obs - d$y_mean
  cv <- function(l) mean(d$y_obs >= d[[paste0("lo", l)]] & d$y_obs <= d[[paste0("hi", l)]])
  data.frame(RMSE = sqrt(mean((o - p)^2)), MAE = mean(abs(o - p)), Bias = mean(p - o),
             R2 = 1 - sum((o - p)^2) / sum((o - mean(o))^2),
             RMSE_model = sqrt(mean(e^2)), R2_model = 1 - sum(e^2) / sum((d$y_obs - mean(d$y_obs))^2),
             cov50 = cv(50), cov80 = cv(80), cov90 = cv(90), cov95 = cv(95),
             width90 = mean(BT(d$hi90) - BT(d$lo90)),
             IS90 = mean((d$hi90 - d$lo90) + 20 * pmax(d$lo90 - d$y_obs, 0) + 20 * pmax(d$y_obs - d$hi90, 0)),
             CRPS = mean(d$crps),
             rho_sd_err = if (all(is.na(d$sd_f))) NA_real_ else
               suppressWarnings(cor(d$sd_pred, abs(d$y_obs - d$y_median), method = "spearman")))
}
metrics <- function(df, ...) df |> group_by(...) |> group_modify(~ score(.x)) |> ungroup()
msd <- function(x) sprintf("%.3f ± %.3f", mean(x, na.rm = TRUE), sd(x, na.rm = TRUE))   # "mean ± SD"

# Parallel execution: one job = one row of `jobs`
par_run <- function(jobs, fun) {
  cl <- makeCluster(N_CORES); on.exit(stopCluster(cl))
  clusterEvalQ(cl, suppressPackageStartupMessages({ library(dbarts); library(sf); library(gstat) }))
  clusterExport(cl, intersect(c("X_list", "Y", "FOLDS", "GRID", "G_DEF", "NUM", "MCMC_CV", "MCMC_INNER", "QP",
                                "fit_bart", "bart_fit", "split_share", "db_args", "summ_ppd", "X_vs", "col_vs", "PRED",
                                "xy", "K_FOLD", "block_m", "block_folds", "BT", "TRANSFORM",
                                "infold_X", "interp_one", "LAB", "INTERP", "INTERP_METHOD", "fit_vg", "CUT_KM",
                                "interp_loo", "oof_X", "fold_X", "FOLD_INTERP", "SET_MODE", "OOF_SCHEME",
                                "OOF_BLOCK_KM"),
                              ls(globalenv())))
  parLapplyLB(cl, split(jobs, seq_len(nrow(jobs))), fun)
}
# One CV job: (predictor set, design, repeat, metal, prior setting g), all outer folds
job_cv <- function(job) {
  X <- X_list[[job$set]]; mode <- SET_MODE[[job$set]]
  y <- Y[, job$metal]; fold <- FOLDS[[job$design]][, job$rep]; h <- GRID[job$g, ]
  set.seed(job$seed)
  do.call(rbind, lapply(sort(unique(fold)), function(f) {
    te <- which(fold == f); tr <- which(fold != f)
    pre <- if (job$design %in% names(FOLD_INTERP)) FOLD_INTERP[[job$design]][[job$rep]][[as.character(f)]]
    fx <- fold_X(X, mode, tr, te, pre)                         # pre = NULL: interpolated inside the job
    fb <- fit_bart(fx$tr, y[tr], fx$te, h, MCMC_CV, job$seed * 100L + f)
    cbind(job[setdiff(names(job), c("g", "seed", "crps"))], combo = h$combo, fold = f, NUM = NUM[te],
          summ_ppd(fb$ev, fb$sigma, y[te], crps = job$crps), row.names = NULL)
  }))
}

# Check that the installed dbarts returns outputs of the expected shape
local({
  chk <- fit_bart(X_list$all[-(1:5), ], Y[-(1:5), 1], X_list$all[1:5, , drop = FALSE], GRID[G_DEF, ],
                  list(nskip = 50L, ndpost = 100L), seed = 1L)
  stopifnot(identical(dim(chk$ev), c(100L, 5L)), length(chk$sigma) == 100L, all(chk$sigma > 0))
  fit <- bart_fit(X_list$all[-(1:5), ], Y[-(1:5), 1], GRID[G_DEF, ], list(nskip = 50L, ndpost = 100L), seed = 1L)
  stopifnot(identical(dim(bart_draws(fit, X_list$all[1:5, , drop = FALSE])), c(100L, 5L)),
            length(split_share(fit)) == ncol(X_list$all))
  writeLines(paste("dbarts", as.character(packageVersion("dbarts"))), op("dbarts_api_check.txt"))
})


# ############################################################################################################################
# 4. HYPERPARAMETER SENSITIVITY: 24 PRIOR SETTINGS IN THE SAME OUTER FOLDS
# ###########################################################################################################################
section("4. Hyperparameter grid")
grid_cv <- cached_sets(paste0("cv_grid_", TAG, ".rds"), function(sets) {
  jobs <- expand.grid(g = seq_len(nrow(GRID)), set = sets, design = DESIGNS, rep = seq_len(N_REP),
                      metal = METALS, stringsAsFactors = FALSE)
  jobs$seed <- seq_len(nrow(jobs)) + 100000L * match(jobs$set, names(PRED_SETS)); jobs$crps <- FALSE
  bind_rows(bind_rows(par_run(jobs, job_cv)), null_cv(DESIGNS, seq_len(N_REP), sets))
})
m_grid <- metrics(grid_cv, set, design, metal, combo, rep)
write_csv(m_grid, op("grid_metrics_by_repeat.csv"))
write_csv(m_grid |> filter(combo %in% c(DEFAULT, "NULL")) |> group_by(set, design, metal, combo) |>
            summarise(across(c(R2_model, RMSE_model, R2, RMSE, MAE, Bias), msd), .groups = "drop"),
          op("table_accuracy_default_vs_null.csv"))
sens <- m_grid |> filter(combo != "NULL") |> group_by(set, design, metal, combo) |>
  summarise(R2_model = mean(R2_model), .groups = "drop")
write_csv(sens |> group_by(set, design, metal) |>
            summarise(default = R2_model[combo == DEFAULT], best = max(R2_model), worst = min(R2_model),
                      best_combo = combo[which.max(R2_model)], default_rank = rank(-R2_model)[combo == DEFAULT],
                      .groups = "drop"), op("table_tuning_sensitivity.csv"))
write_csv(sens, op("fig_tuning_heatmap_data.csv"))
PRIOR_ORDER <- c("conservative", "aggressive", "default")          # row order of the heatmap
for (s in names(PRED_SETS)) {
  hm <- filter(sens, set == s) |> left_join(GRID, by = "combo") |>
    mutate(combo = factor(combo, levels = GRID$combo[order(match(GRID$prior, PRIOR_ORDER), GRID$ntree, GRID$k)]))
  ggsave(op(paste0("fig_tuning_heatmap_", s, ".png")), width = 9, height = 9, dpi = 300,
         ggplot(hm, aes(factor(metal, METALS), combo, fill = R2_model)) + geom_tile() +
           geom_text(aes(label = sprintf("%.2f", R2_model)), size = 2.5) +
           geom_tile(data = filter(hm, combo == DEFAULT), fill = NA, colour = "black", linewidth = 0.9) +
           scale_fill_gradient2(low = "firebrick", mid = "white", high = "steelblue") +
           facet_wrap(~ factor(design, DESIGNS), labeller = as_labeller(DES_NAME)) +
           labs(x = NULL, y = NULL, fill = paste0("CV R² (", SCALE_LAB, ")"),
                caption = "Black outline: default configuration (m = 200, k = 2, default σ prior)") + theme_bw())
}


# ############################################################################################################################
# 5. NESTED CV: TUNED VS DEFAULT PRIORS
# ########################################################################################################################
# Inside each outer training set, the 24 settings are compared with blocked inner 5-fold CV (same block
# size as the outer design); the best one (lowest inner RMSE) is refitted and scored on the outer test fold.
section("5. Nested CV")
job_nested <- function(job) {
  X <- X_list[[job$set]]; y <- Y[, job$metal]; fold <- FOLDS[[job$design]][, job$rep]
  te <- which(fold == job$ofold); tr <- which(fold != job$ofold)
  mode <- SET_MODE[[job$set]]
  inner <- block_folds(xy[tr, , drop = FALSE], block_m(job$design), K_FOLD, job$seed)
  ifs <- sort(unique(inner))
  Xin <- lapply(ifs, function(f) { ii <- which(inner == f)      # inner interpolation uses inner training sites only
    fold_X(X, mode, tr[-ii], tr[ii]) })
  oof <- sapply(seq_len(nrow(GRID)), function(g) {
    p <- numeric(length(tr))
    for (j in seq_along(ifs)) { ii <- which(inner == ifs[j])
      p[ii] <- colMeans(fit_bart(Xin[[j]]$tr, y[tr[-ii]], Xin[[j]]$te, GRID[g, ],
                                 MCMC_INNER, job$seed * 1000L + g * 10L + ifs[j])$ev) }
    p
  })
  gb <- which.min(sqrt(colMeans((y[tr] - oof)^2)))
  fx <- fold_X(X, mode, tr, te, FOLD_INTERP[[job$design]][[job$rep]][[as.character(job$ofold)]])
  fb <- fit_bart(fx$tr, y[tr], fx$te, GRID[gb, ], MCMC_CV, job$seed)
  id <- job[c("set", "design", "rep", "metal", "ofold")]
  list(test = cbind(id, combo = GRID$combo[gb], NUM = NUM[te], summ_ppd(fb$ev, fb$sigma, y[te]), row.names = NULL),
       oof  = data.frame(id, NUM = NUM[tr], y_obs = y[tr], oof_default = oof[, G_DEF], row.names = NULL))
}
nested <- cached_sets(paste0("cv_nested_", TAG, ".rds"), function(sets) {
  base <- expand.grid(set = sets, design = DESIGNS, rep = seq_len(N_REP_NESTED), metal = METALS,
                      stringsAsFactors = FALSE)
  jobs <- do.call(rbind, lapply(seq_len(nrow(base)), function(i)
    data.frame(base[i, ], ofold = sort(unique(FOLDS[[base$design[i]]][, base$rep[i]])), row.names = NULL)))
  jobs$seed <- seq_len(nrow(jobs)) + 100000L * match(jobs$set, names(PRED_SETS))
  res <- par_run(jobs, job_nested)
  list(test = bind_rows(lapply(res, `[[`, "test")), oof = bind_rows(lapply(res, `[[`, "oof")))
})
write_csv(nested$oof, op("bart_default_inner_oof.csv"))       # out-of-fold residuals for the hybrid (EBKR) step
write_csv(bind_rows(metrics(nested$test, set, design, metal, rep) |> mutate(model = "tuned"),
                    m_grid |> filter(combo %in% c(DEFAULT, "NULL"), rep <= N_REP_NESTED) |>
                      mutate(model = ifelse(combo == "NULL", "null", "default")) |> select(-combo)) |>
            group_by(set, design, metal, model) |> summarise(across(c(R2_model, RMSE, cov90), msd), .groups = "drop"),
          op("table_tuning_nested.csv"))
write_csv(nested$test |> distinct(set, design, metal, rep, ofold, combo) |> count(set, design, metal, combo),
          op("table_tuning_selection_frequency.csv"))


###################################################################################################################################
# 6. PREDICTIVE UNCERTAINTY OF DEFAULT BART
#################################################################################################################################
section("6. Uncertainty assessment")
unc_cv <- cached_sets(paste0("cv_uncertainty_", TAG, ".rds"), function(sets) {
  jobs <- expand.grid(set = sets, design = DESIGNS, rep = seq_len(N_REP), metal = METALS,
                      stringsAsFactors = FALSE)
  jobs$seed <- seq_len(nrow(jobs)) + 100000L * match(jobs$set, names(PRED_SETS)); jobs$g <- G_DEF; jobs$crps <- TRUE
  bind_rows(bind_rows(par_run(jobs, job_cv)) |> mutate(combo = "BART"), null_cv(DESIGNS, seq_len(N_REP), sets))
})
m_unc <- metrics(unc_cv, set, design, metal, combo, rep)
crpss <- m_unc |> group_by(set, design, metal, rep) |>                       # CRPS skill vs the null model
  summarise(CRPSS = 1 - CRPS[combo == "BART"] / CRPS[combo == "NULL"], .groups = "drop")
write_csv(m_unc |> group_by(set, design, metal, combo) |>
            summarise(across(c(cov50, cov80, cov90, cov95, width90, IS90, CRPS, rho_sd_err), msd), .groups = "drop") |>
            left_join(crpss |> group_by(set, design, metal) |> summarise(CRPSS = msd(CRPSS), .groups = "drop") |>
                        mutate(combo = "BART"), by = c("set", "design", "metal", "combo")),
          op("table_uncertainty.csv"))

# Calibration curves: empirical vs nominal coverage of central intervals (from the PIT values)
nominal <- seq(0.05, 0.95, 0.05)
calib <- unc_cv |> filter(combo == "BART") |> group_by(set, design, metal) |>
  group_modify(~ data.frame(nominal = nominal, empirical = sapply(nominal, function(l) mean(abs(.x$pit - 0.5) <= l / 2)))) |>
  ungroup()
write_csv(calib, op("fig_calibration_data.csv"))
CAL_SET <- c(oof = "Out-of-fold soil layers", infold = "In-fold interpolation",
             independent = "Independent predictors", all = "Original rasters (leaky)")
calib <- calib |> mutate(set = factor(set, names(CAL_SET), CAL_SET), design = factor(design, DESIGNS, DES_NAME(DESIGNS)))
ax01 <- function(f) f(limits = c(0, 1), breaks = seq(0, 1, 0.25), labels = c("0", "0.25", "0.5", "0.75", "1"),
                      expand = c(0, 0))
ggsave(op("fig_calibration.png"), width = 6, height = 10, dpi = 600,
       ggplot(calib, aes(nominal, empirical, colour = factor(metal, METALS), shape = factor(metal, METALS))) +
         geom_abline(linetype = 2, colour = "grey50") + geom_line(linewidth = 0.4) + geom_point(size = 1.1) +
         facet_grid(set ~ design) + coord_equal() +
         scale_colour_manual(values = PAL_M) + scale_shape_manual(values = SHAPE_M) +
         ax01(scale_x_continuous) + ax01(scale_y_continuous) +
         labs(x = "Nominal coverage", y = "Empirical coverage", colour = NULL, shape = NULL) +
         theme_bw(base_size = 9) + theme(panel.grid.minor = element_blank(), strip.background = element_blank(),
                                         strip.text = element_text(face = "bold"), legend.position = "top",
                                         panel.spacing = unit(3, "mm"), plot.margin = margin(4, 8, 4, 4)))
write_csv(unc_cv |> filter(combo == "BART", set == EVAL_SET) |> select(design, rep, metal, fold, NUM, pit),
          op("fig_PIT_data.csv"))
ggsave(op("fig_PIT.png"), width = 11, height = 5, dpi = 300,
       ggplot(filter(unc_cv, combo == "BART", set == EVAL_SET), aes(pit)) +
         geom_histogram(aes(y = after_stat(density)), breaks = seq(0, 1, 0.1), fill = "steelblue", colour = "white") +
         geom_hline(yintercept = 1, linetype = 2) + facet_grid(design ~ factor(metal, METALS)) +
         labs(x = "PIT", y = "Density") + theme_bw())


##########################################################################################################################
# 7. FINAL MODELS: CONVERGENCE AND POSTERIOR MAPS
# ####################################################################################################################
section("7. Final models and posterior maps")
rhat <- function(m) {                                           # split-R-hat, m = chains x draws
  h <- floor(ncol(m) / 2); m <- rbind(m[, 1:h], m[, (h + 1):(2 * h)])
  nn <- ncol(m); B <- nn * var(rowMeans(m)); W <- mean(apply(m, 1, var)); sqrt(((nn - 1) / nn * W + B / nn) / W)
}
# Bands of BART_<metal>_posterior_<MAP_TAG>.tif (sd_* on the modelling scale)
LAYERS <- c("median_mgkg", "mean_mgkg", "lo90_mgkg", "hi90_mgkg", "lo95_mgkg", "hi95_mgkg",
            "pi90_width_mgkg", "pi90_relwidth", "mean_model", "sd_model", "sd_total", "applicable")
CENSOR0 <- TRANSFORM == "none"
# Training matrix of the mapping model; for "oof" the soil layers at the samples are leave-one-out kriged
X_map_tr <- if (MAP_SET == "oof") oof_X(X_list$oof, seq_len(n)) else X_list[[MAP_SET]]
if (MAP_SET == "oof") write_csv(bind_cols(NUM = NUM, as.data.frame(X_map_tr[, INTERP]),
                                          as.data.frame(LAB) |> rename_with(~ paste0("lab_", .x))),
                                op(paste0("map_training_soil_values_", OOF_SCHEME, ".csv")))
set.seed(2026)
final_tab <- bind_rows(lapply(seq_along(METALS), function(mi) {
  m <- METALS[mi]
  fits <- lapply(seq_len(MCMC_FINAL$nchain), function(ch)       # independent chains = separately seeded fits
    bart_fit(X_map_tr, Y[, mi], GRID[G_DEF, ], MCMC_FINAL, seed = 1000L * mi + ch))
  sig_m  <- t(sapply(fits, bart_sigma, ndpost = MCMC_FINAL$ndpost))
  fit_tr <- lapply(fits, bart_draws, Xnew = X_map_tr)            # in-sample draws for R-hat of f(x)
  rh_f   <- sapply(seq_len(n), function(i) rhat(t(sapply(fit_tr, function(x) x[, i]))))
  sig    <- as.vector(t(sig_m))
  res <- do.call(rbind, lapply(split(seq_along(cells), ceiling(seq_along(cells) / 2000)), function(idx) {
    ev  <- do.call(rbind, lapply(fits, bart_draws, Xnew = X_new[idx, , drop = FALSE]))
    ppd <- ev + matrix(rnorm(length(ev)), nrow(ev)) * sig
    sd_tot <- apply(ppd, 2, sd)
    q_raw  <- apply(ppd, 2, quantile, c(0.05, 0.5))
    neg    <- cbind(BT(q_raw[2, ]) < 0, BT(q_raw[1, ]) < 0)     # median / lower 90% bound below 0 (raw scale only)
    if (CENSOR0) ppd <- pmax(ppd, 0)
    q   <- BT(apply(ppd, 2, quantile, c(0.025, 0.05, 0.5, 0.95, 0.975)))
    cbind(q[3, ], colMeans(BT(ppd)), q[2, ], q[4, ], q[1, ], q[5, ], q[4, ] - q[2, ],
          ifelse(q[3, ] > 0, (q[4, ] - q[2, ]) / q[3, ], NA_real_),
          colMeans(ev), apply(ev, 2, sd), sd_tot, neg)
  }))
  r <- to_rast(cbind(res[, 1:11], as.numeric(app)), LAYERS)
  terra::writeRaster(r, op(paste0("BART_", m, "_posterior_", MAP_TAG, ".tif")), overwrite = TRUE)
  png(op(paste0("map_BART_", m, "_", MAP_TAG, ".png")), width = 2400, height = 1300, res = 150)   
  par(mfrow = c(2, 4), mar = c(2, 2, 3, 5))
  for (l in c("median_mgkg", "lo90_mgkg", "hi90_mgkg", "pi90_relwidth", "sd_model", "sd_total", "applicable"))
    terra::plot(r[[l]], main = paste(m, "—", l), axes = FALSE)
  dev.off()
  data.frame(metal = m, map_set = MAP_SET, transform = TRANSFORM, censored_at_0 = CENSOR0,
             Rhat_sigma = rhat(sig_m), Rhat_f_max = max(rh_f), sigma_mean = mean(sig_m),
             map_median = median(res[, 1]), map_p05 = quantile(res[, 1], 0.05), map_p95 = quantile(res[, 1], 0.95),
             share_neg_median = mean(res[, 12]), share_neg_lo90 = mean(res[, 13]),
             median_relwidth90 = median(res[, 8], na.rm = TRUE), median_width90 = median(res[, 7]),
             share_lo90_at0 = mean(res[, 3] <= 0), model_var_share = mean(res[, 10]^2 / res[, 11]^2),
             pct_applicable = 100 * mean(app), row.names = NULL)
}))
write_csv(final_tab, op(paste0("table_final_model_", MAP_TAG, ".csv")))
if (CENSOR0 && any(final_tab$share_neg_lo90 > 0.01))
  message("Raw scale: lower 90% bounds fell below 0 mg/kg in >1% of cells for ",
          paste(final_tab$metal[final_tab$share_neg_lo90 > 0.01], collapse = ", "), " (censored at 0 in the maps).")

## ---- 7a. Publication maps 
#   fig_map_median     posterior median (mg/kg)
#   fig_map_pi90width  width of the 90% prediction interval (mg/kg)
#   fig_map_modelcv    coefficient of variation (%) of the model (trend) component
section("7a. Publication maps")
if (all(sapply(c("ggspatial", "patchwork"), requireNamespace, quietly = TRUE))) {
  library(patchwork)
  MAP_W <- 180; FONT <- 8                                        # figure width (mm, double column), font (pt)
  rd <- lapply(setNames(METALS, METALS), function(m) terra::rast(op(paste0("BART_", m, "_posterior_", MAP_TAG, ".tif"))))
  map_df <- function(r, lyr) {
    d <- terra::as.data.frame(r[[c(lyr, "applicable")]], xy = TRUE, na.rm = FALSE)
    names(d)[3:4] <- c("v", "app"); d[!is.na(d$app), ]
  }
  map_panel <- function(d, title, leg, pal = "viridis", pts = FALSE) {
    lim <- quantile(d$v[d$app == 1], c(0.02, 0.98), na.rm = TRUE)
    g <- ggplot() +
      geom_raster(data = d[d$app == 0, ], aes(x, y), fill = "grey82") +
      geom_raster(data = d[d$app == 1, ], aes(x, y, fill = v)) +
      scale_fill_viridis_c(option = pal, limits = lim, oob = scales::squish, name = leg,
                           breaks = scales::breaks_extended(n = 4), labels = scales::label_number(drop0trailing = TRUE),
                           guide = guide_colourbar(barwidth = unit(30, "mm"), barheight = unit(2.2, "mm"),
                                                   title.position = "top", title.hjust = 0.5)) +
      coord_sf(crs = sf::st_crs(EPSG), datum = sf::st_crs(4326), expand = FALSE) +
      labs(title = title, x = NULL, y = NULL) +
      theme_bw(base_size = FONT) +
      theme(legend.position = "bottom", legend.margin = margin(0, 0, 0, 0),
            plot.title = element_text(face = "bold", size = FONT + 1, hjust = 0),
            axis.text = element_blank(), axis.ticks = element_blank(), panel.grid = element_blank()) +
      ggspatial::annotation_scale(location = "br", width_hint = 0.18, height = unit(0.9, "mm"),
                                  text_cex = 0.4, line_width = 0.25, bar_cols = c("black", "white")) +
      ggspatial::annotation_north_arrow(location = "tl", height = unit(4.5, "mm"), width = unit(3.2, "mm"),
                                        style = ggspatial::north_arrow_orienteering(text_size = 3.5, line_width = 0.5))
    if (pts) g <- g + geom_point(data = md, aes(Lon, Lat), shape = 21, size = 0.7, stroke = 0.25,
                                 fill = "white", colour = "black")
    g
  }
  # Model CV (%) from the model SD: log scale (lognormal) 100 * sqrt(exp(sd^2) - 1); raw scale 100 * sd / mean
  model_cv <- function(r, d) {
    if (TRANSFORM == "log") 100 * sqrt(exp(d$v^2) - 1)
    else 100 * d$v / pmax(map_df(r, "mean_model")$v, .Machine$double.eps)
  }
  map_fig <- function(lyr, leg, pal, file, pts = FALSE, fun = NULL) {
    p <- lapply(METALS, function(m) {
      d <- map_df(rd[[m]], lyr); if (!is.null(fun)) d$v <- fun(rd[[m]], d)
      map_panel(d, m, leg, pal = pal, pts = pts) })
    f <- wrap_plots(p, ncol = 3) + plot_annotation(tag_levels = "a", tag_prefix = "(", tag_suffix = ")") &
      theme(plot.tag = element_text(size = FONT + 1, face = "bold"))
    ggsave(op(paste0(file, "_", MAP_TAG, ".png")), f, width = MAP_W, height = 130, units = "mm", dpi = 600)
    ggsave(op(paste0(file, "_", MAP_TAG, ".pdf")), f, width = MAP_W, height = 130, units = "mm")
  }
  map_fig("median_mgkg", "Posterior median (mg/kg)", "viridis", "fig_map_median", pts = TRUE)
  map_fig("pi90_width_mgkg", "90% PI width (mg/kg)", "magma", "fig_map_pi90width")
  map_fig("sd_model", "Model CV (%)", "cividis", "fig_map_modelcv", fun = model_cv)
} else message("Section 7a skipped: install.packages(c(\"ggspatial\", \"patchwork\")) for the publication maps.")

## ---- 7b. Single-band GeoTIFFs for ArcGIS mapp processing
#   BART_<metal>_median_*     band 1   posterior median (mg/kg)
#   BART_<metal>_pi90width_*  band 7   90% prediction-interval width (mg/kg)
#   BART_<metal>_modelsd_*    band 10  model SD (log scale if TRANSFORM = "log", else mg/kg)
#   BART_<metal>_modelcv_*    model CV (%)
section("7b. Single-band GeoTIFFs")
ebkr_dir <- op("ebkr_input"); dir.create(ebkr_dir, showWarnings = FALSE)
for (m in METALS) {
  f <- op(paste0("BART_", m, "_posterior_", MAP_TAG, ".tif"))
  if (!file.exists(f)) { message("Section 7c: not found, run Section 7 first: ", f); next }
  r <- terra::rast(f)
  if (terra::nlyr(r) != 12L) { message("Section 7c: ", basename(f), " has ", terra::nlyr(r), " bands, expected 12; skipped"); next }
  cv <- if (TRANSFORM == "log") 100 * sqrt(exp(r[[10]]^2) - 1) else 100 * r[[10]] / r[[9]]
  lyr <- list(median = r[[1]], pi90width = r[[7]], modelsd = r[[10]], modelcv = cv)
  for (k in names(lyr)) {
    names(lyr[[k]]) <- k
    terra::writeRaster(lyr[[k]], op(paste0("BART_", m, "_", k, "_", MAP_TAG, ".tif")), overwrite = TRUE)
  }
  trend <- r[[9]]; names(trend) <- "trend"; trend_sd <- r[[10]]; names(trend_sd) <- "trend_sd"
  terra::writeRaster(trend,    file.path(ebkr_dir, paste0("BART_", m, "_trend_",   TRANSFORM, ".tif")), overwrite = TRUE)
  terra::writeRaster(trend_sd, file.path(ebkr_dir, paste0("BART_", m, "_trendsd_", TRANSFORM, ".tif")), overwrite = TRUE)
}


##################################################################################################
# 8. VARIABLE RELEVANCE AGAINST A PERMUTATION NULL ACCORDING TO (Bleich et al., 2014)
###############################################################################################################################
# Split share of each predictor in small BART models (m = 20) fitted to the real response (5 runs)
# and to permuted responses (N_PERM runs). Local threshold: 95th percentile of the predictor's own null;
# global threshold: 95th percentile of the maximum (prior-scaled) null share.
section("8. Variable relevance")
X_vs   <- X_list$all
col_vs <- vapply(colnames(X_vs), function(cn) if (cn %in% PRED) cn else FACTORS[startsWith(cn, FACTORS)], character(1))
prior_share <- as.numeric(table(factor(col_vs, levels = PRED))) / ncol(X_vs); names(prior_share) <- PRED
job_vs <- function(job) {
  y <- Y[, job$metal]; if (job$type == "perm") { set.seed(job$seed); y <- sample(y) }
  fit  <- bart_fit(X_vs, y, transform(GRID[G_DEF, ], ntree = 20L), list(nskip = 1000L, ndpost = 1000L), job$seed)
  ss   <- split_share(fit)
  prop <- if (all(colnames(X_vs) %in% names(ss))) ss[colnames(X_vs)] else setNames(as.numeric(ss), colnames(X_vs))
  data.frame(metal = job$metal, type = job$type, run = job$run, predictor = PRED,
             share = sapply(PRED, function(v) sum(prop[col_vs == v])), row.names = NULL)
}
vs_runs <- cached(paste0("variable_relevance_runs_db_", TRANSFORM, "_", PRED_ID, ".rds"), {
  jobs <- bind_rows(expand.grid(metal = METALS, type = "real", run = 1:5,      stringsAsFactors = FALSE),
                    expand.grid(metal = METALS, type = "perm", run = 1:N_PERM, stringsAsFactors = FALSE))
  jobs$seed <- 5000L + seq_len(nrow(jobs))
  bind_rows(par_run(jobs, job_vs))
})
gmax <- vs_runs |> filter(type == "perm") |> group_by(metal, run) |>
  summarise(mx = max(share / prior_share[predictor]), .groups = "drop") |>
  group_by(metal) |> summarise(gmax_thr = quantile(mx, 0.95), .groups = "drop")
vs <- vs_runs |> group_by(metal, predictor) |>
  summarise(real = mean(share[type == "real"]), null_mean = mean(share[type == "perm"]),
            null_q95 = quantile(share[type == "perm"], 0.95), .groups = "drop") |>
  left_join(gmax, by = "metal") |>
  mutate(prior = prior_share[predictor], ratio = real / null_mean,
         relevant_local = real > null_q95, relevant_global = real / prior > gmax_thr)
write_csv(vs, op("table_variable_relevance.csv"))
ggsave(op("fig_variable_relevance.png"), width = 15, height = 6, dpi = 300,
       ggplot(vs, aes(y = reorder(predictor, ratio))) +
         geom_errorbarh(aes(xmin = null_mean, xmax = null_q95), height = 0, linewidth = 2.5, colour = "grey80") +
         geom_point(aes(x = real, colour = relevant_local), size = 2) +
         scale_colour_manual(values = c(`TRUE` = "firebrick", `FALSE` = "grey30")) +
         facet_wrap(~ factor(metal, METALS), nrow = 1, scales = "free_x") +
         labs(x = "Split share (ntree = 20); grey = permutation null mean to 95th percentile", y = NULL, colour = "Relevant") +
         theme_bw())


#######################################################################################################################################
# 9. SPATIAL STRUCTURE OF OUT-OF-FOLD RESIDUALS DIANGNOSTICS
# ######################################################################################################################################
# Residuals of default BART (primary design, averaged over repeats): variogram, leave-one-out kriging R2  of the residuals, and Moran's I in distance bands (permutation p-values)
section("9. Residual diagnostics")
oof <- grid_cv |> filter(set == EVAL_SET, design == DESIGNS[1], combo == DEFAULT) |>
  group_by(metal, NUM) |> summarise(y_obs = first(y_obs), pred = mean(y_mean), .groups = "drop") |>
  mutate(resid = y_obs - pred) |> left_join(md[, c("NUM", "Lon", "Lat")], by = "NUM")
write_csv(oof, op("oof_residuals.csv"))
# Residuals for EBKR in ArcGIS # load with "XY Table To Point", coordinate system WGS 84 / UTM zone 50N)
dir.create(op("ebkr_input"), showWarnings = FALSE)
for (m in METALS) write_csv(oof |> filter(metal == m) |> select(NUM, Lon, Lat, y_obs, pred, resid),
                            op(file.path("ebkr_input", paste0("residuals_", m, "_", TRANSFORM, ".csv"))))
set.seed(2026)
bands <- c(0, 2000, 4000, 8000, 16000)                          
moran_p <- function(x, lo, hi, nperm = 999) {
  W <- (D > lo & D <= hi) * 1; diag(W) <- 0; z <- x - mean(x); s0 <- sum(W)
  mi <- function(zz) (n / s0) * sum(W * outer(zz, zz)) / sum(zz^2)
  obs <- mi(z); c(I = obs, p = (1 + sum(replicate(nperm, mi(sample(z))) >= obs)) / (nperm + 1), pairs = s0 / 2)
}
resid_tab <- bind_rows(lapply(METALS, function(m) {
  d <- oof |> filter(metal == m) |> arrange(match(NUM, md$NUM)); s <- to_km_sf(d, "resid")
  fit <- fit_vg(variogram(z ~ 1, s, cutoff = CUT_KM, width = 2)); vg <- describe_vg(fit)
  mor <- sapply(seq_len(length(bands) - 1), function(b) moran_p(d$resid, bands[b], bands[b + 1]))
  kr <- if (!is.null(fit) && vg$status != "pure nugget") {
    k <- suppressWarnings(krige.cv(z ~ 1, s, model = fit, nfold = nrow(d), verbose = FALSE))
    1 - sum(k$residual^2) / sum((k$observed - mean(k$observed))^2) } else NA
  cbind(data.frame(metal = m, var_ratio = var(d$resid) / var(d$y_obs), resid_skew = skew(d$resid),
                   resid_shapiro_p = shapiro.test(d$resid)$p.value), vg, LOO_krig_R2 = kr,
        setNames(as.data.frame(t(mor["I", ])), paste0("I_", bands[-length(bands)] / 1000, "km")),
        setNames(as.data.frame(t(mor["p", ])), paste0("p_", bands[-length(bands)] / 1000, "km")))
}))
write_csv(resid_tab, op("table_residual_structure.csv"))


# ############################################################################################################################
# 10. SENSITIVITY OF THE CV RESULTS TO THE VALIDATION BLOCK SIZE
# #############################################################################################################################
section("10. Block-size sensitivity")
SIZE_DESIGNS <- c("random_k5", paste0("block", DIAG_SIZES, "km"))
size_cv <- cached(paste0("cv_blocksize_db_", EVAL_SET, "_", TRANSFORM, "_", PRED_ID, ".rds"), {
  jobs <- expand.grid(set = EVAL_SET, design = SIZE_DESIGNS, rep = seq_len(N_REP), metal = METALS,
                      stringsAsFactors = FALSE)
  jobs$seed <- 9000L + seq_len(nrow(jobs)); jobs$g <- G_DEF; jobs$crps <- FALSE
  bind_rows(par_run(jobs, job_cv))
})
nnd <- setNames(design_tab$median_cv_nnd_m / 1000, design_tab$design)
x_lab <- setNames(paste0(ifelse(SIZE_DESIGNS == "random_k5", "Random",
                                paste(sub("^block([0-9.]+)km$", "\\1", SIZE_DESIGNS), "km")),
                         ifelse(SIZE_DESIGNS %in% DESIGNS, " (main)", ""),
                         "\n", sprintf("%.1f km", nnd[SIZE_DESIGNS])), SIZE_DESIGNS)
size_tab <- metrics(size_cv, design, metal, rep) |> group_by(design, metal) |>
  summarise(R2_mean = mean(R2_model), R2_sd = sd(R2_model), cov90_mean = mean(cov90), cov90_sd = sd(cov90),
            RMSE_mean = mean(RMSE), .groups = "drop") |>
  mutate(median_test_train_km = nnd[design], design = factor(design, SIZE_DESIGNS))
write_csv(size_tab, op("table_blocksize_sensitivity.csv"))
panels <- c(r2 = paste0("Cross-validated R² (", SCALE_LAB, ")"), cov = "90% prediction-interval coverage")
long <- bind_rows(transmute(size_tab, design, metal, panel = panels[["r2"]], mean = R2_mean, sd = R2_sd),
                  transmute(size_tab, design, metal, panel = panels[["cov"]], mean = cov90_mean, sd = cov90_sd)) |>
  mutate(panel = factor(panel, panels), metal = factor(metal, METALS))
pd <- position_dodge(width = 0.4)
ggsave(op("fig_blocksize_sensitivity.png"), width = 9, height = 7.5, dpi = 300,
       ggplot(long, aes(design, mean, colour = metal, shape = metal, group = metal)) +
         geom_hline(data = data.frame(panel = factor(panels, panels), y = c(0, 0.9)), aes(yintercept = y),
                    linetype = 2, colour = "grey55") +
         geom_line(position = pd, linewidth = 0.6) +
         geom_errorbar(aes(ymin = mean - sd, ymax = mean + sd), width = 0, linewidth = 0.4, position = pd) +
         geom_point(position = pd, size = 2.6) +
         facet_wrap(~ panel, ncol = 1, scales = "free_y") +
         scale_x_discrete(labels = x_lab) +
         scale_colour_manual(values = PAL_M) + scale_shape_manual(values = SHAPE_M) +
         labs(x = "Validation design (median distance from test sample to nearest training sample)", y = NULL,
              colour = "Metal", shape = "Metal",
              subtitle = paste0("Default BART, predictor set '", EVAL_SET, "'; mean ± SD over ", N_REP,
                                " repeats; dashed lines: R² = 0 and nominal 90%")) +
         theme_bw(base_size = 11) + theme(panel.grid.minor = element_blank(), panel.grid.major.x = element_blank()))


# ###############################################################################################################
# 11. UNCERTAINTY SUMMARY FIGURES (from Section 6)
# #######################################################################################################################
section("11. Uncertainty summary figures")
PAL_SET <- c(oof = "#009E73", infold = "#0072B2", independent = "#E69F00", all = "#8a8a8a")
SET_LAB <- c(oof = "out-of-fold soil layers (train and test)", infold = "in-fold interpolation (test only)",
             independent = "independent predictors only", all = "original rasters, leaky upper bound")
DES_LAB <- setNames(DES_NAME(DESIGNS), DESIGNS)
mstat <- function(d, v) d |> summarise(mean = mean(.data[[v]]), sd = sd(.data[[v]]), .groups = "drop")
th <- theme_bw(base_size = 11) + theme(panel.grid.minor = element_blank(), panel.grid.major.y = element_blank(),
                                       legend.position = "top", strip.background = element_blank(),
                                       strip.text = element_text(face = "bold", hjust = 0))
# (a) Coverage of 50% and 90% intervals: BART vs null, primary design
cov_df <- m_unc |> filter(set == EVAL_SET, design == DESIGNS[1]) |>
  tidyr::pivot_longer(c(cov50, cov90), names_to = "level", values_to = "cov") |>
  group_by(metal, combo, level) |> mstat("cov") |>
  mutate(level = factor(level, c("cov50", "cov90"), c("Coverage of 50% intervals", "Coverage of 90% intervals")),
         nominal = ifelse(grepl("50", level), 0.5, 0.9), metal = factor(metal, rev(METALS)),
         model = ifelse(combo == "BART", paste0("BART: ", SET_LAB[[EVAL_SET]]), "Null model: training mean"))
write_csv(cov_df, op("fig_uncertainty_coverage_data.csv"))
ggsave(op("fig_uncertainty_coverage.png"), width = 10, height = 4.5, dpi = 300,
       ggplot(cov_df, aes(mean, metal, colour = model, shape = model)) +
         geom_vline(aes(xintercept = nominal), linetype = 2, colour = "grey45") +
         geom_errorbarh(aes(xmin = mean - sd, xmax = mean + sd), height = 0, linewidth = 0.6,
                        position = position_dodge(width = 0.5)) +
         geom_point(size = 2.8, fill = "white", position = position_dodge(width = 0.5)) +
         facet_wrap(~ level, scales = "free_x") +
         scale_colour_manual(values = c("#0072B2", "#8a8a8a")) + scale_shape_manual(values = c(16, 23)) +
         labs(x = "Empirical coverage of held-out samples (dashed = nominal)", y = NULL, colour = NULL, shape = NULL,
              caption = paste0("Spatial block CV (", DES_LAB[[1]], "); mean ± 1 SD over ", N_REP, " repeats")) + th)
# (b) CRPS skill and 90% interval width relative to the null, all predictor sets and both designs
skill_df <- bind_rows(
  crpss |> group_by(set, design, metal) |> mstat("CRPSS") |> mutate(panel = "CRPS skill score vs null (>0 = better)"),
  m_unc |> select(set, design, metal, rep, combo, width90) |>
    tidyr::pivot_wider(names_from = combo, values_from = width90) |>
    mutate(wr = 100 * (BART / `NULL` - 1)) |>                  # backticks: the column "NULL", not R's NULL
    group_by(set, design, metal) |> mstat("wr") |>
    mutate(panel = "90% interval width relative to null (%)")) |>
  mutate(metal = factor(metal, rev(METALS)), set = factor(set, names(SET_LAB)), design = factor(design, DESIGNS),
         panel = factor(panel, c("CRPS skill score vs null (>0 = better)", "90% interval width relative to null (%)")))
write_csv(skill_df, op("fig_uncertainty_skill_data.csv"))
ggsave(op("fig_uncertainty_skill.png"), width = 10, height = 4.8, dpi = 300,
       ggplot(skill_df, aes(mean, metal, colour = set, shape = design, group = interaction(set, design))) +
         geom_vline(xintercept = 0, colour = "grey35") +
         geom_errorbarh(aes(xmin = mean - sd, xmax = mean + sd), height = 0, linewidth = 0.6,
                        position = position_dodge(width = 0.7)) +
         geom_point(size = 2.6, fill = "white", position = position_dodge(width = 0.7)) +
         facet_wrap(~ panel, scales = "free_x") +
         scale_colour_manual(values = PAL_SET, labels = SET_LAB) +
         scale_shape_manual(values = c(16, 21), labels = DES_LAB) +
         guides(shape = guide_legend(order = 1), colour = guide_legend(order = 2, nrow = 2)) +
         theme(legend.box = "vertical", legend.spacing.y = unit(0, "pt")) +
         labs(x = "0 = null model", y = NULL, colour = NULL, shape = NULL,
              caption = paste0("Spatial block CV; mean ± 1 SD over ", N_REP, " repeats")) + th)

# ##################################################################################################################################
# 12. OBSERVED VS PREDICTED
# ##################################################################################################################################
# One panel per metal. Each point is a sample: observed concentration against its held-out posterior
# median, averaged over the CV repeats; vertical bars = 90% prediction interval (averaged).
# Panel label: CV R2 (modelling scale, mean ± SD over repeats, as in the tables), RMSE (mg/kg) and
# MAPE (%), the last computed from the plotted (repeat-averaged) predictions.
section("12. Observed vs predicted")
for (d in DESIGNS) {
  
  # ---- plotted points: one row per sample and metal (back-transformed to mg/kg) 
  sc <- unc_cv |> filter(combo == "BART", set == EVAL_SET, design == d) |>
    group_by(metal, NUM) |>
    summarise(obs  = BT(first(y_obs)),
              pred = BT(mean(y_median)),
              lo   = BT(mean(lo90)),
              hi   = BT(mean(hi90)),
              .groups = "drop") |>
    mutate(metal = factor(metal, METALS))
  
  # ---- metrics per metal 
  # R2 and RMSE: mean (and SD for R2) over the CV repeats, as in the tables
  st_cv <- m_unc |> filter(combo == "BART", set == EVAL_SET, design == d) |> group_by(metal) |>
    summarise(R2_mean = mean(R2_model), R2_sd = sd(R2_model), RMSE = mean(RMSE), .groups = "drop")
  
  # MAPE (and, for reference, pooled R2 / RMSE) from the plotted repeat-averaged predictions
  TRF <- if (TRANSFORM == "log") log else identity
  st_pt <- sc |> group_by(metal) |>
    summarise(n          = n(),
              MAPE       = 100 * mean(abs(obs - pred) / obs),
              MdAPE      = 100 * median(abs(obs - pred) / obs),      
              R2_pooled  = 1 - sum((TRF(obs) - TRF(pred))^2) / sum((TRF(obs) - mean(TRF(obs)))^2),
              RMSE_pooled = sqrt(mean((obs - pred)^2)),
              .groups = "drop")
  
  st <- left_join(st_cv, st_pt, by = "metal") |>
    mutate(lab = sprintf("R² = %.2f ± %.2f\nRMSE = %s mg/kg\nMAPE = %.1f%%\nn = %d",
                         R2_mean, R2_sd, format(signif(RMSE, 3)), MAPE, n),
           metal = factor(metal, METALS))
  
  # ---- same x and y range in each panel 
  rng <- sc |> group_by(metal) |>
    summarise(a = min(obs, pred, lo[lo > 0], na.rm = TRUE), b = max(obs, pred, hi), .groups = "drop")
  blank <- bind_rows(transmute(rng, metal, x = a, y = a), transmute(rng, metal, x = b, y = b))
  
  # ---- figure 
  g <- ggplot(sc, aes(obs, pred, colour = metal)) +
    geom_blank(data = blank, aes(x, y), inherit.aes = FALSE) +
    geom_abline(slope = 1, intercept = 0, linetype = 2, colour = "grey45", linewidth = 0.4) +
    geom_linerange(aes(ymin = pmax(lo, rng$a[match(metal, rng$metal)]), ymax = hi),
                   alpha = 0.25, linewidth = 0.35) +
    geom_point(size = 1.4, alpha = 0.85) +
    geom_text(data = st, aes(x = -Inf, y = Inf, label = lab), inherit.aes = FALSE,
              hjust = -0.06, vjust = 1.15, size = 2.5, lineheight = 0.95) +
    facet_wrap(~ metal, ncol = 3, scales = "free") +
    scale_colour_manual(values = PAL_M, guide = "none") +
    labs(x = "Observed concentration (mg/kg)", y = "Predicted concentration (mg/kg)") +
    theme_bw(base_size = 9) +
    theme(aspect.ratio = 1, panel.grid.minor = element_blank(), strip.background = element_blank(),
          strip.text = element_text(face = "bold", hjust = 0, size = 10))
  if (TRANSFORM == "log") g <- g + scale_x_log10() + scale_y_log10()
  
  # ---- outputs: figure (PNG, PDF)
  f <- paste0("fig_obs_vs_pred_", EVAL_SET, "_", d, "_", TRANSFORM)
  write_csv(sc, op(paste0(f, "_data.csv")))
  write_csv(select(st, metal, n, R2_mean, R2_sd, RMSE, MAPE, MdAPE, R2_pooled, RMSE_pooled),
            op(paste0(f, "_metrics.csv")))
  ggsave(op(paste0(f, ".png")), g, width = 180, height = 125, units = "mm", dpi = 600)
  ggsave(op(paste0(f, ".pdf")), g, width = 180, height = 125, units = "mm")
}

writeLines(capture.output(sessionInfo()), op("sessionInfo.txt"))
section(paste("Done. Outputs in", normalizePath(out_dir)))
###################################END###########################################################################################
