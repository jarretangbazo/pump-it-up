# =============================================================================
# 04 - Baseline Models
# Project: Pump It Up — Data Mining the Water Table
# Author: Jarret Angbazo
# =============================================================================
# Run after 03_feature_engineering.ipynb has been executed.
# Requires: tidyverse, tidymodels, vip, ranger
#
# install.packages(c("tidyverse", "tidymodels", "vip", "ranger"))
#
# Objective: Establish baseline classification accuracy with simple models
# All preprocessing is handled inside a recipes pipeline — no data leakage.

library(tidyverse)
library(tidymodels)
library(vip)

tidymodels_prefer()
set.seed(123)

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
X_train <- read_csv("data/processed/X_train.csv")
y_train <- read_csv("data/processed/y_train.csv") |>
  mutate(status_group = factor(status_group_enc,
                                levels = c(0, 1, 2),
                                labels = c("non_functional",
                                           "needs_repair",
                                           "functional")))

train_df <- bind_cols(X_train, y_train["status_group"])

cat("Training set:", nrow(train_df), "rows,", ncol(train_df) - 1, "features\n")
cat("Class distribution:\n")
print(count(train_df, status_group) |> mutate(pct = round(n / sum(n) * 100, 1)))

# ---------------------------------------------------------------------------
# 2. CV split — 5-fold stratified (mirrors Python setup)
# ---------------------------------------------------------------------------
folds <- vfold_cv(train_df, v = 5, strata = status_group)

# ---------------------------------------------------------------------------
# 3. Recipe — shared preprocessing for all models
# ---------------------------------------------------------------------------
base_recipe <- recipe(status_group ~ ., data = train_df) |>
  # All columns already numeric from Python feature engineering
  step_zv(all_predictors()) |>                    # remove zero-variance columns
  step_normalize(all_numeric_predictors())        # needed only for logistic reg

# ---------------------------------------------------------------------------
# 4. Model specifications
# ---------------------------------------------------------------------------

# 4a. Majority class (dummy baseline)
dummy_spec <- decision_tree(mode = "classification") |>
  set_engine("rpart") |>
  set_mode("classification")

# 4b. Logistic Regression (multinomial)
lr_spec <- multinom_reg(penalty = 0.001) |>
  set_engine("nnet") |>
  set_mode("classification")

# 4c. Decision Tree
dt_spec <- decision_tree(
  cost_complexity = 0.001,
  tree_depth      = 15,
  min_n           = 10
) |>
  set_engine("rpart") |>
  set_mode("classification")

# ---------------------------------------------------------------------------
# 5. Workflows
# ---------------------------------------------------------------------------
lr_wflow <- workflow() |>
  add_recipe(base_recipe) |>
  add_model(lr_spec)

dt_wflow <- workflow() |>
  add_recipe(recipe(status_group ~ ., data = train_df) |> step_zv(all_predictors())) |>
  add_model(dt_spec)

# ---------------------------------------------------------------------------
# 6. Fit and evaluate via cross-validation
# ---------------------------------------------------------------------------
metrics_set <- metric_set(accuracy, mn_log_loss)

cat("\nFitting Logistic Regression (5-fold CV)...\n")
lr_res <- lr_wflow |>
  fit_resamples(folds, metrics = metrics_set,
                control = control_resamples(save_pred = TRUE))

cat("Fitting Decision Tree (5-fold CV)...\n")
dt_res <- dt_wflow |>
  fit_resamples(folds, metrics = metrics_set,
                control = control_resamples(save_pred = TRUE))

# ---------------------------------------------------------------------------
# 7. Collect and compare results
# ---------------------------------------------------------------------------
lr_acc <- collect_metrics(lr_res) |> filter(.metric == "accuracy") |> pull(mean)
dt_acc <- collect_metrics(dt_res) |> filter(.metric == "accuracy") |> pull(mean)

results <- tibble(
  model    = c("Majority Class (54.3%)", "Logistic Regression", "Decision Tree"),
  cv_acc   = c(0.5431, round(lr_acc, 4), round(dt_acc, 4))
) |> arrange(desc(cv_acc))

cat("\n")
cat(str_pad("=" , 45, pad = "="), "\n")
cat("BASELINE MODEL RESULTS (5-fold CV Accuracy)\n")
cat(str_pad("=" , 45, pad = "="), "\n")
print(results)

# ---------------------------------------------------------------------------
# 8. Per-class performance — Decision Tree
# ---------------------------------------------------------------------------
dt_preds <- collect_predictions(dt_res)
cat("\nDecision Tree — Per-class performance:\n")
dt_preds |>
  group_by(status_group) |>
  summarise(
    n        = n(),
    accuracy = mean(.pred_class == status_group),
    .groups  = "drop"
  ) |>
  print()

# ---------------------------------------------------------------------------
# 9. Decision tree feature importance
# ---------------------------------------------------------------------------
# Fit on full training data
dt_final <- dt_wflow |> fit(train_df)

cat("\nDecision Tree — Top feature importances:\n")
dt_final |>
  extract_fit_parsnip() |>
  vip(num_features = 15) |>
  print()

# ---------------------------------------------------------------------------
# 10. Confusion matrix — Decision Tree
# ---------------------------------------------------------------------------
cat("\nDecision Tree — Confusion Matrix (CV):\n")
dt_preds |>
  conf_mat(truth = status_group, estimate = .pred_class) |>
  print()

# ---------------------------------------------------------------------------
# 11. Save results
# ---------------------------------------------------------------------------
write_csv(results, "data/processed/baseline_results.csv")
cat("\nBaseline results saved to data/processed/baseline_results.csv\n")
cat("\nNext: Run 05_advanced_models.R for Random Forest + XGBoost\n")
