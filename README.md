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


---

## Competition

### Problem Definition
Using data on water pumps in Tanzania collected by Taarifa and the Tanzanian Ministry of Water, the task is to classify each pump as functional, functional needs repair, or non-functional. GitHub

Predictions are based on variables about pump type, installation date, and management. The evaluation metric is classification accuracy. The training set has 59,400 observations and 41 features.

**Why it matters**
A smart understanding of which waterpoints will fail can improve maintenance operations and ensure that clean, potable water is available to communities across Tanzania.

### Task

Using data from Taarifa and the Tanzanian Ministry of Water, predict which pumps are functional, which need some repairs, and which don't work at all. 

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
git clone https://github.com/YOUR_USERNAME/pump-it-up.git
cd pump-it-up
python -m venv venv && source venv/bin/activate
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

DrivenData. (2015). *Pump it Up: Data Mining the Water Table.* Retrieved May 10, 2026 from https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/.