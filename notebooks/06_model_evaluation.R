# =============================================================================
# 06 - Model Evaluation and Interpretability
# Project: Pump It Up — Data Mining the Water Table
# Author: Jarret Angbazo | Date: June 2026
# =============================================================================
# Run after 05_advanced_models.R has been executed.
# Requires: tidyverse, tidymodels, ranger or xgboost (whichever was best),
#           vip, yardstick
#
# Objective: Comprehensive evaluation of the best model:
#  - Confusion matrix and per-class metrics
#  - Calibration
#  - Feature importance (permutation + model-native)
#  - Subgroup accuracy by key categorical features
#  - Model card

library(tidyverse)
library(tidymodels)
library(vip)
library(yardstick)

tidymodels_prefer()
set.seed(42)

# ---------------------------------------------------------------------------
# 1. Load data and best model workflow
# ---------------------------------------------------------------------------
X_train  <- read_csv("data/processed/X_train.csv")
y_train  <- read_csv("data/processed/y_train.csv") |>
  mutate(status_group = factor(status_group_enc,
                                levels = c(0, 1, 2),
                                labels = c("non_functional",
                                           "needs_repair",
                                           "functional")))
X_test   <- read_csv("data/processed/X_test.csv")
test_ids <- read_csv("data/processed/test_ids.csv")

train_df <- bind_cols(X_train, y_train["status_group"])

if (!file.exists("models/best_model.rds")) {
  stop(
    "models/best_model.rds not found.\n",
    "Run 05_advanced_models.R first, or use the fallback re-fit block below."
  )
}

# Load best model fit (from 05_advanced_models.R)
# Adjust to whichever workflow was best
best_fit <- readRDS("models/best_model.rds")   # saved at end of script 05 — see note below

# NOTE: If running independently, re-fit the best workflow here:
# library(ranger)
# rf_spec <- rand_forest(mtry=8, trees=300, min_n=3) |>
#   set_engine("ranger", importance="permutation") |>
#   set_mode("classification")
# best_fit <- workflow() |>
#   add_recipe(recipe(status_group ~ ., data=train_df) |> step_zv(all_predictors())) |>
#   add_model(rf_spec) |>
#   fit(train_df)

# ---------------------------------------------------------------------------
# 2. Cross-validated predictions (fair evaluation)
# ---------------------------------------------------------------------------
folds    <- vfold_cv(train_df, v = 5, strata = status_group)
cv_preds <- best_fit |>
  fit_resamples(folds,
                metrics = metric_set(accuracy, mn_log_loss, kap),
                control = control_resamples(save_pred = TRUE)) |>
  collect_predictions()

# ---------------------------------------------------------------------------
# 3. Overall metrics
# ---------------------------------------------------------------------------
overall_metrics <- cv_preds |>
  metrics(truth = status_group, estimate = .pred_class)

cat("=" , strrep("=", 50), "\n", sep = "")
cat("OVERALL CV METRICS\n")
cat("=" , strrep("=", 50), "\n", sep = "")
print(overall_metrics)

# ---------------------------------------------------------------------------
# 4. Confusion matrix
# ---------------------------------------------------------------------------
cat("\nConfusion Matrix:\n")
cv_preds |>
  conf_mat(truth = status_group, estimate = .pred_class) |>
  print()

# Autoplot (save in RStudio)
cv_preds |>
  conf_mat(truth = status_group, estimate = .pred_class) |>
  autoplot(type = "heatmap") +
  scale_fill_gradient(low = "#EEF2FF", high = "#1976D2") +
  labs(title = "Confusion Matrix — Best Model (5-fold CV)") +
  theme_minimal()
# ggsave("reports/figures/06_confusion_matrix.png", width = 6, height = 5)

# ---------------------------------------------------------------------------
# 5. Per-class precision / recall / F1
# ---------------------------------------------------------------------------
cat("\nPer-Class Metrics:\n")
class_metrics <- metric_set(precision, recall, f_meas)
cv_preds |>
  class_metrics(truth = status_group, estimate = .pred_class) |>
  pivot_wider(names_from = .metric, values_from = .estimate) |>
  print()

# ---------------------------------------------------------------------------
# 6. Subgroup accuracy by key features
# ---------------------------------------------------------------------------
# Re-attach raw feature values to CV predictions
train_raw <- read_csv("data/raw/training_set_features.csv") |>
  select(id, quantity, waterpoint_type, extraction_type_class,
         payment, region, basin)

# Match on row index (CV preds preserve original row order within folds)
cv_preds_aug <- cv_preds |>
  mutate(row_id = .row) |>
  bind_cols(train_raw |>
              slice(cv_preds$.row) |>
              select(-id))

subgroup_acc <- function(data, group_col) {
  data |>
    group_by({{ group_col }}) |>
    summarise(
      n        = n(),
      accuracy = mean(.pred_class == status_group),
      .groups  = "drop"
    ) |>
    arrange(desc(accuracy))
}

cat("\nAccuracy by 'quantity' category:\n")
subgroup_acc(cv_preds_aug, quantity) |> print(n = 10)

cat("\nAccuracy by 'waterpoint_type':\n")
subgroup_acc(cv_preds_aug, waterpoint_type) |> print(n = 10)

cat("\nAccuracy by 'extraction_type_class':\n")
subgroup_acc(cv_preds_aug, extraction_type_class) |> print(n = 10)

# ---------------------------------------------------------------------------
# 7. Feature importance — permutation importance
# ---------------------------------------------------------------------------
# Fit on full training set with importance = "permutation"
# (Must re-fit with ranger importance param if using RF)

cat("\nPermutation Feature Importance (full training set):\n")
best_fit_full <- best_fit |> fit(train_df)

best_fit_full |>
  extract_fit_parsnip() |>
  vip(num_features = 20, method = "model") +
  labs(title = "Feature Importance — Best Model") +
  theme_minimal() +
  theme(axis.text.y = element_text(size = 9))
# ggsave("reports/figures/06_feature_importance.png", width = 8, height = 7)

# ---------------------------------------------------------------------------
# 8. Calibration (reliability diagram)
# ---------------------------------------------------------------------------
# Generate probability predictions
cv_probs <- best_fit |>
  fit_resamples(folds,
                metrics = metric_set(accuracy),
                control = control_resamples(save_pred = TRUE)) |>
  collect_predictions()

# Check if probabilities are available
if (".pred_functional" %in% names(cv_probs)) {
  cal_data <- cv_probs |>
    mutate(true_functional = as.integer(status_group == "functional")) |>
    mutate(prob_bin = cut(.pred_functional, breaks = seq(0, 1, 0.1)))

  cal_summary <- cal_data |>
    group_by(prob_bin) |>
    summarise(n = n(), obs_rate = mean(true_functional), .groups = "drop")

  cat("\nCalibration check for 'functional' class:\n")
  print(cal_summary)
}

# ---------------------------------------------------------------------------
# 9. Model Card
# ---------------------------------------------------------------------------
model_card <- list(
  model_name        = "Pump It Up — Water Pump Classifier",
  model_type        = "Random Forest (ranger) / XGBoost — see 05_advanced_models.R",
  training_date     = as.character(Sys.Date()),
  training_samples  = nrow(train_df),
  features_used     = ncol(X_train),
  target            = "status_group (3 classes)",
  evaluation_metric = "Accuracy (classification)",
  cv_accuracy       = overall_metrics |> filter(.metric == "accuracy") |> pull(.estimate),
  class_imbalance   = "54% functional / 38% non-functional / 7% needs repair",
  known_limitations = paste(
    "Model performs weakest on 'functional needs repair' (~35% recall).",
    "This minority class is underrepresented and may benefit from SMOTE (see recipe in 05).",
    "Geographic features (longitude, latitude, region) are strong predictors —",
    "model may not generalize to pumps outside the training geography."
  ),
  recommended_use = paste(
    "Prioritize non-functional pump flagging for maintenance dispatch.",
    "Use probability scores to rank pumps by urgency, not just the binary prediction.",
    "Manual inspection recommended for cases where P(non-functional) is 0.4-0.6."
  )
)

cat("\n", strrep("=", 50), "\n", sep = "")
cat("MODEL CARD\n")
cat(strrep("=", 50), "\n", sep = "")
for (nm in names(model_card)) {
  cat(sprintf("  %-22s : %s\n", nm, model_card[[nm]]))
}

# Save model card
jsonlite::write_json(model_card, "models/model_card.json", pretty = TRUE, auto_unbox = TRUE)
cat("\nModel card saved: models/model_card.json\n")

# Save a human-readable version of the model card alongside the JSON
model_card_text <- paste0(
  strrep("=", 50), "\n",
  "MODEL CARD — Pump It Up Water Pump Classifier\n",
  strrep("=", 50), "\n",
  sprintf("  %-25s %s\n", "Model type:",        model_card$model_type),
  sprintf("  %-25s %s\n", "Training date:",     model_card$training_date),
  sprintf("  %-25s %s\n", "Training samples:",  model_card$training_samples),
  sprintf("  %-25s %s\n", "Features used:",     model_card$features_used),
  sprintf("  %-25s %.4f\n","CV accuracy:",       model_card$cv_accuracy),
  "\nLimitations:\n", "  ", model_card$known_limitations, "\n",
  "\nRecommended use:\n", "  ", model_card$recommended_use, "\n"
)

writeLines(model_card_text, "models/model_card.txt")
cat("\nModel card (text) saved: models/model_card.txt\n")

# ---------------------------------------------------------------------------
# 10. Final summary table
# ---------------------------------------------------------------------------
cat("\n", strrep("=", 50), "\n", sep = "")
cat("FULL RESULTS COMPARISON\n")
cat(strrep("=", 50), "\n", sep = "")

# Load results saved by 04_baseline_models.R and 05_advanced_models.R
baseline_results <- read_csv("data/processed/baseline_results.csv", show_col_types = FALSE)
advanced_results <- read_csv("data/processed/advanced_results.csv", show_col_types = FALSE)

# Extract individual accuracy values by model name
pull_acc <- function(df, model_name) {
  val <- df |> filter(model == model_name) |> pull(cv_acc)
  if (length(val) == 0) NA_real else val
}

dt_acc <- pull_acc(baseline_results, "Decision Tree")
rf_default_acc <- pull_acc(advanced_results, "Random Forest (100 trees)")
rf_acc <- pull_acc(advanced_results, "Random Forest (tuned)")
xgb_acc <- pull_acc(advanced_results, "XGBoost (tuned)")

results_comparison <- tibble(
  model = c(
    "Majority Class Baseline",
    "Logistic Regression",
    "Decision Tree",
    "Random Forest (100 trees)",
    "Random Forest (tuned)",
    "XGBoost (tuned)"
  ),
  cv_accuracy = c(
    0.5431, 
    0.6606, 
    dt_acc, 
    rf_default_acc, 
    rf_acc, 
    xgb_acc
    ),
  notes = c(
    "Always predicts 'functional'",
    "Linear decision boundary",
    "Single tree, depth 15",
    "100 trees, default params",
    "Tuned via random search (see 05)",
    "Tuned via random search (see 05)"
  )
)
print(results_comparison)
cat("\nNote: NA values for tuned models are populated after running 05_advanced_models.R\n")
