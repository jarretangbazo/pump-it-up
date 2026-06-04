<!-- omit in toc -->
# Pump It Up
<!-- omit in toc -->
### Data Mining the Water Table
**DrivenData Competition:** https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/

Predict which water pumps are faulty to promote access to clean, potable water across Tanzania.

[![Live Demo](https:/img.shields.io/badge/Live%20Demo-Streamlit-red)](https://url.streamlit.app)
[![Competition](https:/img.shields.io/badge/DrivenData-%2366-blue)](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/)

---
<!-- omit in toc -->
## Table of Contents

- [Competition](#competition)
  - [Problem Definition](#problem-definition)
  - [Task](#task)
  - [Getting the Data](#getting-the-data)
    - [Data Files](#data-files)
- [Environment Setup](#environment-setup)
  - [Project Structure](#project-structure)
  - [Local Setup](#local-setup)
  - [Tech Stack](#tech-stack)
- [Results and Key Findings](#results-and-key-findings)
  - [Results](#results)
  - [Key Findings](#key-findings)
- [Decision implication: The model's primary operational value is in triaging non-functional pumps for maintenance dispatch. For the "needs repair" class, human inspection of model-flagged borderline cases (0.4 \< P(non-functional) \< 0.6) is strongly recommended.](#decision-implication-the-models-primary-operational-value-is-in-triaging-non-functional-pumps-for-maintenance-dispatch-for-the-needs-repair-class-human-inspection-of-model-flagged-borderline-cases-04--pnon-functional--06-is-strongly-recommended)


---

## Competition

### Problem Definition
Using data on water pumps in Tanzania collected by Taarifa and the Tanzanian Ministry of Water, the task is to classify each pump as **functional**, **functional needs repair**, or **non-functional**.

Predictions draw on variables including pump type, installation date, geographic location, water source characteristics and management. The evaluation metric is **classification accuracy**. 

The training set has 59,400 observations and 41 features.

**Why it matters**
Nearly 57 million people in Tanzania rely on rural water infrastructure. A model that identifies failing pumps before communities lose access enables targeted maintenance, reduces downtime, and directs limited repair resources where they matter most. The decision this model supports: *which pumps should inspectors visit next?*

### Task

Using data from Taarifa and the Tanzanian Ministry of Water, predict which pumps are functional, which need some repairs, and which don't work at all. 

| **Class** | **Train Count** | **% share** |
|------|-------------|------------|
| functional | 32,259 | 54.3% |
| non-functional | 22,824 | 38.4% |
| functional needs repair | 4,317 | 7.3% |


### Getting the Data

1. Sign up or log in at https://www.drivendata.org
2. Join the competition at the link above
3. Go to the **Data** tab and download all files into `data/raw/`

#### Data Files

| File | Description |
|------|-------------|
| `training_set_features.csv` | 59,400 pump records |
| `training_set_labels.csv` | Status label for each training pump |
| `test_set_features.csv` | 14,850 pumps to predict |
| `submission_format.csv` | Template showing the required submission structure |

## Environment Setup

### Project Structure

```
pump-it-up/
├── data/
│   ├── raw/             # Raw competition files (not committed)
│   └── processed/       # Cleaned and engineered datasets
├── notebooks/           # EDA and experimentation
│   ├── 01_eda.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_baseline_models.R
│   ├── 05_advanced_models.R
│   └── 06_model_evaluation.R
├── reports/           
│   └── figures           # All visualization outputs
├── models/              # Saved model files (not committed)
├── submissions/         # Competition CSV files
├── requirements.txt
└── README.md
```

### Local Setup

**Python Setup**

```bash
git clone https://github.com/YOUR_USERNAME/pump-it-up.git
cd pump-it-up
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

**R Setup**

```bash
install.packages(c(
  "tidyverse",    # data wrangling + ggplot2
  "tidymodels",   # unified modeling framework
  "ranger",       # fast Random Forest backend
  "xgboost",      # gradient boosting
  "vip",          # variable importance plots
  "themis"        # SMOTE for class imbalance
))
```

### Tech Stack

**Python:** pandas | scikit-learn | XGBoost | matplotlib | seaborn | numpy

**R:** tidyverse | tidymodels | ranger | xgboost | vip | themis | yardstick

---

## Results and Key Findings

### Results

| **Model** | **CV Accuracy** | **Notes** |
|-------|----------|--------------|
| Majority Class Baseline | 0.5431 | Always predicts "functional" |
| Logistic Regression | 0.6606 | Multinomial, L2 regularized | 
| Decision Tree | | | 
| Random Forest (100 trees) | | | 
| Random Forest (tuned, 300 trees) | | | 
| XGBoost (tuned) | | | 
| Random Forest + XGBoost (R tuned) | | | 


### Key Findings

What predicts pump failure?

Water quantity is the single strongest signal. Pumps reporting quantity == dry fail at a 96.9% rate. Pumps with quantity == enough are functional 65.2% of the time. This single feature accounts for ~16% of Random Forest's mean decrease in impurity — more than any geographic or structural variable.
Geography matters, but through infrastructure patterns, not location per se. Longitude, latitude, and GPS height rank 2nd–4th in feature importance. This reflects uneven infrastructure investment across Tanzania's regions rather than a purely spatial effect. Regional-level accuracy varies substantially; inspectors in low-performing regions should be prioritised.
Pumps that have never had a payment mechanism are 47.6% non-functional. Payment structures (payment == never pay) are strongly associated with failure, likely because they proxy for absent maintenance funding. This is the most actionable lever for policy intervention.
Construction decade shifts the functional rate from ~85% (2000s) to ~40% (1960s–70s). Age alone is a blunt signal — a well-maintained older pump outperforms a neglected newer one — but pump age is the 9th most important feature and improves the engineered pump_age feature that combines construction year with the survey date.
"Functional needs repair" is the hardest class to detect (recall: 35%). At 7.3% of the training set, this minority class is underrepresented. SMOTE oversampling (via themis in R script 05) should improve recall for this class without sacrificing overall accuracy significantly.

Decision implication: The model's primary operational value is in triaging non-functional pumps for maintenance dispatch. For the "needs repair" class, human inspection of model-flagged borderline cases (0.4 < P(non-functional) < 0.6) is strongly recommended.
---

DrivenData. (2015). *Pump it Up: Data Mining the Water Table.* Retrieved May 10, 2026 from https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/.