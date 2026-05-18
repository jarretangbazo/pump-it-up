<!-- omit in toc -->
# Flu Shot Learning
<!-- omit in toc -->
### Predict H1N1 and Seasonal Flu Vaccines
**DrivenData Competition:** https://www.drivendata.org/competitions/66/flu-shot-learning/

Predict the likelihood that a survey respondent received the H1N1 and seasonal flu vaccines, using background, opinion, and behavioral features from the United States Centers for Disease Control's 2009 National H1N1 Flu Survey.

[![Live Demo](https:/img.shields.io/badge/Live%20Demo-Streamlit-red)](https://url.streamlit.app)
[![Competition](https:/img.shields.io/badge/DrivenData-%2366-blue)](https://www.drivendata.org/competitions/66/flu-shot-learning/)

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


---

## Competition

### Problem Definition
Public health agencies need to understand what drives vaccine acceptance and hesitancy to design effective outreach campaigns. This project builds a multi-label classifier that predicts two independent binary outcomes per respondent: HINI vaccine uptake and seasonal flu vaccine uptake.

**Why it matters**
Understanding what drives vaccine uptake (and hesitancy) allows public health agencies to design better outreach campaigns, target under-vaccinated populations, and anticipate demand. This applies directly to future pandemics, including COVID-19.

### Task

Given survey responses about a person's background, opinions, and behaviors, predict two binary outcomes:
- Did they receive the **H1N1 vaccine**? (`h1n1_vaccine`: 0 or 1)
- Did they receive the **seasonal flu vaccine**? (`seasonal_vaccine`: 0 or 1)

### Getting the Data

1. Sign up or log in at https://www.drivendata.org
2. Join the competition at the link above
3. Go to the **Data** tab and download all files into `data/raw/`

#### Data Files

| File | Description |
|------|-------------|
| `training_set_features.csv` | Survey responses for ~26,700 training respondents |
| `training_set_labels.csv` | The two vaccine targets for each training respondent |
| `test_set_features.csv` | Survey responses for ~26,700 test respondents |
| `submission_format.csv` | Template showing the required submission structure |

## Environment Setup

### Project Structure

```
flu-shot-learning/
├── data/
│   ├── raw/             # Raw competition files (not committed)
│   ├── processed/       # Cleaned and engineered datasets
├── notebooks/           # EDA and experimentation
│   ├── 01_eda.ipynb
│   ├── 02_data_cleaning.ipynb
│   └── 03_feature_engineering.ipynb
│   └── 04_baseline_models.ipynb
│   └── 05_advanced_models.ipynb
│   └── 06_model_evaluation.ipynb
├── reports/           
│   ├── figures
├── src/
│   ├── features.py      # Feature engineering functions
│   ├── train.py         # Training and MLflow logging script
│   └── api.py           # FastAPI deployment
├── models/              # Saved model files (not committed)
├── submissions/         # Competition CSV files
├── logs/                # Prediction and monitoring logs
├── app.py               # Streamlit demo app
├── Dockerfile
├── requirements.txt
```

### Local Setup

```bash
git clone https://github.com/YOUR_USERNAME/flu-shot-learning.git
cd flu-shot-learning
python -m venv flu_env && source flu_env/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

To launch the MLflow experiment dashboard:
```bash
mlflow ui   # opens at http://localhost:5000
```

To run the prediction API:
```bash
uvicorn src.api:app --reload    #opens at http://localhost:8000
```

### Tech Stack

Python | pandas | scikit-learn | XGBoost | LightGBM | SHAP | Optuna | MLflow | FastAPI | Docker | Streamlit

---

## Results and Key Findings

### Results

| Model | H1N1 AUC | Seasonal AUC | Mean AUC |
|-------|----------|--------------|----------|
| Logistic Regression (baseline) | | | |
| XGBoost (tuned) | | | |
| XGBoost + LightGBM ensemble | | | |


### Key Findings

*Summary of key findings forthcoming*

---

DrivenData. (2015). *Flu Shot Learning: Predict H1N1 and Seasonal Flu Vaccines.* Retrieved May 10, 2026 from https://www.drivendata.org/competitions/66/flu-shot-learning/.