# =============================================================================
# 05 - Advanced Models
# Project: Pump It Up — Data Mining the Water Table
# Author: Jarret Angbazo
# =============================================================================
# Run after 04_baseline_models.R has been executed.
# Requires: tidyverse, tidymodels, ranger, xgboost, vip, themis
#
# install.packages(c("tidyverse","tidymodels","ranger","xgboost","vip","themis"))
#
# Objective: Tune Random Forest and XGBoost using tidymodels + Optuna-style
# random search. Address class imbalance with SMOTE via {themis}.

library(tidyverse)
library(tidymodels)
library(ranger)       # fast Random Forest backend
library(xgboost)
library(vip)
library(themis)       # SMOTE oversampling for minority class

tidymodels_prefer()
set.seed(42)

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

# ---------------------------------------------------------------------------
# 2. CV strategy — 5-fold stratified
# ---------------------------------------------------------------------------
folds <- vfold_cv(train_df, v = 5, strata = status_group)

# ---------------------------------------------------------------------------
# 3. Recipes
# ---------------------------------------------------------------------------

# Base recipe — no scaling (trees don't need it)
base_recipe <- recipe(status_group ~ ., data = train_df) |>
  step_zv(all_predictors())

# SMOTE recipe — oversample the minority "needs_repair" class
smote_recipe <- base_recipe |>
  step_smote(status_group, over_ratio = 0.5, seed = 42)

# ---------------------------------------------------------------------------
# 4. Model specifications with tunable hyperparameters
# ---------------------------------------------------------------------------

# 4a. Random Forest (ranger backend)
rf_spec <- rand_forest(
  mtry           = tune(),
  trees          = tune(),
  min_n          = tune()
) |>
  set_engine("ranger",
             importance     = "impurity",
             num.threads    = parallel::detectCores()) |>
  set_mode("classification")

# 4b. XGBoost
xgb_spec <- boost_tree(
  trees           = tune(),
  tree_depth      = tune(),
  learn_rate      = tune(),
  loss_reduction  = tune(),
  sample_size     = tune(),
  mtry            = tune()
) |>
  set_engine("xgboost", nthread = parallel::detectCores()) |>
  set_mode("classification")

# ---------------------------------------------------------------------------
# 5. Workflows
# ---------------------------------------------------------------------------
rf_wflow <- workflow() |>
  add_recipe(base_recipe) |>
  add_model(rf_spec)

xgb_wflow <- workflow() |>
  add_recipe(base_recipe) |>
  add_model(xgb_spec)

# ---------------------------------------------------------------------------
# 5b. Default Random Forest — 100 trees, no tuning
# ---------------------------------------------------------------------------
rf_default_spec <- rand_forest(trees = 100) |>
  set_engine("ranger",
             importance  = "impurity",
             num.threads = parallel::detectCores()) |>
  set_mode("classification")

rf_default_wflow <- workflow() |>
  add_recipe(base_recipe) |>
  add_model(rf_default_spec)

cat("Fitting Random Forest (100 trees, default params, 5-fold CV)...\n")
rf_default_res <- rf_default_wflow |>
  fit_resamples(folds,
                metrics = metrics_set,
                control = control_resamples(save_pred = FALSE))

rf_default_acc <- show_best(rf_default_res, metric = "accuracy", n = 1) |> pull(mean)
cat(sprintf("RF (100 trees) CV Accuracy: %.4f\n", rf_default_acc))

# ---------------------------------------------------------------------------
# 6. Hyperparameter grids
# ---------------------------------------------------------------------------
rf_grid <- grid_random(
  mtry(range = c(3, 15)),
  trees(range = c(200, 500)),
  min_n(range = c(1, 10)),
  size = 15   # increase for production; 15 balances speed vs thoroughness
)

xgb_grid <- grid_random(
  trees(range = c(200, 500)),
  tree_depth(range = c(3, 8)),
  learn_rate(range = c(-3, -1)),  # log10 scale: 0.001 to 0.1
  loss_reduction(range = c(-5, 0)),
  sample_prop(range = c(0.6, 1.0)),
  mtry(range = c(5, 20)),
  size = 15
)

# ---------------------------------------------------------------------------
# 7. Tune (parallel recommended: doParallel::registerDoParallel())
# ---------------------------------------------------------------------------
metrics_set <- metric_set(accuracy, mn_log_loss)

cat("Tuning Random Forest (15 candidate configs × 5 folds)...\n")
rf_tune <- rf_wflow |>
  tune_grid(
    resamples = folds,
    grid      = rf_grid,
    metrics   = metrics_set,
    control   = control_grid(verbose = TRUE, save_pred = FALSE)
  )

cat("\nTuning XGBoost (15 candidate configs × 5 folds)...\n")
xgb_tune <- xgb_wflow |>
  tune_grid(
    resamples = folds,
    grid      = xgb_grid,
    metrics   = metrics_set,
    control   = control_grid(verbose = TRUE, save_pred = FALSE)
  )

# ---------------------------------------------------------------------------
# 8. Select best hyperparameters
# ---------------------------------------------------------------------------
best_rf  <- select_best(rf_tune,  metric = "accuracy")
best_xgb <- select_best(xgb_tune, metric = "accuracy")

cat("\nBest RF parameters:\n");  print(best_rf)
cat("\nBest XGBoost parameters:\n"); print(best_xgb)

# ---------------------------------------------------------------------------
# 9. Finalize workflows and fit on full training data
# ---------------------------------------------------------------------------
final_rf_wflow  <- finalize_workflow(rf_wflow,  best_rf)
final_xgb_wflow <- finalize_workflow(xgb_wflow, best_xgb)

final_rf_fit  <- final_rf_wflow  |> last_fit(split = initial_split(train_df, strata = status_group))
final_xgb_fit <- final_xgb_wflow |> last_fit(split = initial_split(train_df, strata = status_group))

# ---------------------------------------------------------------------------
# 10. Compare tuned models
# ---------------------------------------------------------------------------
rf_cv_acc  <- show_best(rf_tune,  metric = "accuracy", n = 1) |> pull(mean)
xgb_cv_acc <- show_best(xgb_tune, metric = "accuracy", n = 1) |> pull(mean)

results <- tibble(
  model  = c("Logistic Regression (baseline)", "Random Forest (100 trees)", "Random Forest (tuned)", "XGBoost (tuned)"),
  cv_acc = c(0.6606, round(rf_default_acc, 4), round(rf_cv_acc, 4), round(xgb_cv_acc, 4))
) |> arrange(desc(cv_acc))

cat("\n")
cat(str_pad("=", 50, pad = "="), "\n")
cat("ADVANCED MODEL RESULTS (5-fold CV Accuracy)\n")
cat(str_pad("=", 50, pad = "="), "\n")
print(results)

# ---------------------------------------------------------------------------
# 11. Variable importance — best model
# ---------------------------------------------------------------------------
best_wflow <- if (rf_cv_acc >= xgb_cv_acc) final_rf_wflow else final_xgb_wflow
best_label <- if (rf_cv_acc >= xgb_cv_acc) "Random Forest" else "XGBoost"

best_fit <- best_wflow |> fit(train_df)
dir.create("models", showWarnings = FALSE, recursive = TRUE)
saveRDS(best_fit, "models/best_model.rds")
cat("\nBest model saved: models/best_model.rds\n")
cat(glue::glue("\n{best_label} — Top 20 Feature Importances:\n"))
best_fit |>
  extract_fit_parsnip() |>
  vip(num_features = 20, geom = "point") +
  ggplot2::labs(title = glue::glue("{best_label} — Feature Importance")) +
  ggplot2::theme_minimal()

# Optional: save plot
ggsave("reports/figures/05_feature_importance.png", width = 8, height = 7)

# ---------------------------------------------------------------------------
# 12. Generate submission (best model)
# ---------------------------------------------------------------------------
X_test   <- read_csv("data/processed/X_test.csv")
test_ids <- read_csv("data/processed/test_ids.csv")

preds <- predict(best_fit, X_test)$.pred_class
label_recode <- c("non_functional" = "non functional",
                  "needs_repair"   = "functional needs repair",
                  "functional"     = "functional")
submission <- bind_cols(test_ids, status_group = label_recode[as.character(preds)])
dir.create("submissions", showWarnings = FALSE, recursive = TRUE)
write_csv(submission, "submissions/r_tuned_submission.csv")
cat("\nSubmission saved: submissions/r_tuned_submission.csv\n")

# ---------------------------------------------------------------------------
# 13. Save tuning results
# ---------------------------------------------------------------------------
write_csv(results, "data/processed/advanced_results.csv")
rf_tuning_summary  <- collect_metrics(rf_tune)  |> arrange(desc(mean))
xgb_tuning_summary <- collect_metrics(xgb_tune) |> arrange(desc(mean))

write_csv(rf_tuning_summary,  "data/processed/rf_tuning_results.csv")
write_csv(xgb_tuning_summary, "data/processed/xgb_tuning_results.csv")

cat("\nAll results saved. Next: Run 06_model_evaluation.R\n")
