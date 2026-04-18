# 🏥 Diabetes Early Readmission Prediction

A full end-to-end machine learning pipeline to predict whether a diabetic patient will be **readmitted to hospital within 30 days** of discharge — a critical quality indicator tracked by the Centers for Medicare & Medicaid Services (CMS).

---

## 👥 Team Members

| Name | Role |
|------|------|
| **Rishi Badhwar** | ML Engineer / Data Scientist |
| **Varun Sahu** | ML Engineer / Data Scientist |

---

## 📌 Problem Statement

Hospital readmissions within 30 days are costly for both patients and healthcare systems. CMS penalises hospitals with high readmission rates, making early risk prediction a high-value clinical tool. This project builds a binary classifier to identify at-risk diabetic patients before discharge, enabling proactive clinical intervention.

- **Target variable:** `early_readmission` — `1` if readmitted within 30 days, `0` otherwise
- **Key challenge:** Severe class imbalance (~11% positive class)

---

## 📂 Dataset

**UCI 130-US Hospitals for Diabetic Patients (1999–2008)**

| Attribute | Details |
|-----------|---------|
| Source | [Kaggle — brandao/diabetes](https://www.kaggle.com/datasets/brandao/diabetes) |
| Rows | ~101,766 patient encounters |
| Features | 55 (demographics, diagnoses, medications, lab results) |
| Target | Binary: readmitted within 30 days |
| Class ratio | ~89% negative / ~11% positive |

---

## 🔁 Pipeline Overview

```
Data Loading → Missing Value Analysis → Cleaning → Feature Engineering
→ EDA → Preprocessing → SMOTE → Model Training → Hyperparameter Tuning
→ Evaluation → SHAP Explainability → Model Saving
```

| Step | Description |
|------|-------------|
| 1 | Data Loading & Initial Exploration |
| 2 | Missing Value & Outlier Analysis |
| 3 | Data Cleaning & Preprocessing |
| 4 | Target Variable Engineering |
| 5 | Feature Engineering (10+ new features) |
| 6 | Descriptive Statistics & Data Quality |
| 7 | Outlier Analysis (IQR + Winsorisation) |
| 8 | Exploratory Data Analysis (15+ visualisations) |
| 9 | Preprocessing Pipeline (Encoding, Scaling, SMOTE) |
| 10 | Multi-Model Training (8 models) |
| 11 | Hyperparameter Tuning (RandomisedSearchCV) |
| 12 | Comprehensive Model Evaluation |
| 13 | Final Model Comparison |
| 14 | Overfitting & Learning Curve Analysis |
| 15 | Feature Importance & SHAP Explainability |
| 16 | Model Saving & Deployment Code |
| 17 | Business Interpretation & Clinical Relevance |
| 18 | Final Summary Dashboard |

---

## ⚙️ Feature Engineering

Domain-informed features were created to improve model signal:

| Feature | Description |
|---------|-------------|
| `age_midpoint` | Numeric midpoint of age range bracket |
| `total_visits` | Sum of outpatient + emergency + inpatient visits |
| `medication_change_flag` | Whether medication was changed during encounter |
| `num_active_medications` | Count of active diabetes medications (polypharmacy proxy) |
| `on_insulin` | Boolean flag for insulin use |
| `diag_N_group` | ICD-9 chapter grouping for all 3 diagnoses |
| `A1C_tested` / `A1C_high` | Glycaemic control indicators |
| `glucose_tested` | Whether max glucose serum was measured |
| `high_utiliser` | Flag for patients with >3 total prior visits |
| `chronic_condition_count` | Number of chronic disease chapters present across diagnoses |

---

## 🤖 Models Trained

| Model | Type |
|-------|------|
| Logistic Regression | Linear |
| K-Nearest Neighbours | Instance-based |
| Decision Tree | Tree |
| Random Forest | Ensemble (Bagging) |
| Gradient Boosting | Ensemble (Boosting) |
| XGBoost | Ensemble (Boosting) |
| LightGBM | Ensemble (Boosting) |
| MLP (Neural Network) | Deep Learning |

All models were evaluated using **5-Fold Stratified Cross-Validation** and tested on a held-out 20% test set. The top 3 models were further tuned using `RandomizedSearchCV`.

---

## 📊 Key Results

| Metric | Expected Range |
|--------|---------------|
| AUC-ROC | 0.68 – 0.72 |
| Best Model | XGBoost / LightGBM / Gradient Boosting variant |

> Best AUC of ~0.68–0.72 is consistent with published literature on this dataset (Strack et al., 2014).

---

## 🔍 Explainability (SHAP)

SHAP (SHapley Additive exPlanations) was used to provide both global and local model explanations, identifying the most clinically meaningful predictors:

1. **Number of inpatient visits** — strongest predictor; prior hospitalisations signal disease severity
2. **Time in hospital** — longer stays correlate with higher readmission risk
3. **Number of diagnoses** — comorbidity burden
4. **Discharge disposition** — patients discharged to care facilities have elevated risk
5. **Insulin use** — indicator of medical complexity
6. **A1C tested** — sicker patients are more likely to be tested

---

## 🛠️ Tech Stack

```
Python 3.x
├── pandas, numpy                 — Data manipulation
├── matplotlib, seaborn, missingno — Visualisation
├── scikit-learn                  — ML models, preprocessing, evaluation
├── xgboost, lightgbm, catboost  — Gradient boosting models
├── imbalanced-learn (SMOTE)      — Class imbalance handling
├── shap                          — Model explainability
└── joblib                        — Model serialisation
```

---

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install kagglehub xgboost lightgbm catboost shap imbalanced-learn missingno scikit-learn pandas numpy matplotlib seaborn joblib
```

### 2. Run the Notebook

Open `Diabetes_Classification.ipynb` in Jupyter Notebook or JupyterLab and run all cells sequentially.

```bash
jupyter notebook Diabetes_Classification.ipynb
```

The notebook will automatically download the dataset from Kaggle via `kagglehub`.

### 3. Run Inference on New Data

```python
import joblib
import pandas as pd

bundle = joblib.load('diabetes_readmission_model.pkl')

result = predict_readmission(new_patient_df)
print(result['probability'])    # e.g. [0.23, 0.67]
print(result['risk_category'])  # e.g. ['Medium', 'High']
```

---

## 📁 Output Files

| File | Description |
|------|-------------|
| `diabetes_readmission_model.pkl` | Trained model bundle (model + scaler + encoders) |
| `model_comparison_results.csv` | All model metrics in tabular format |
| `feature_importance.csv` | Feature importances from best tree model |
| `final_dashboard.png` | Summary dashboard visualisation |

---

## ⚠️ Limitations

- Dataset is from **1999–2008**; clinical practices have evolved significantly
- Uses **ICD-9 codes** (modern systems use ICD-10)
- Missing clinically important features: BMI, social determinants of health, HbA1c numeric values
- Model finds **correlations, not causal relationships**
- Trained on US hospital data — may not generalise internationally
- Default 0.5 threshold used; clinical deployment may benefit from a lower threshold to maximise recall

---

## 🚀 Future Work

- Temporal modelling with patient encounter sequences (RNN/LSTM)
- Incorporate social determinants of health (ZIP code, insurance type)
- Cost-sensitive threshold optimisation (Youden's J statistic)
- Ensemble stacking (meta-learner)
- ML fairness auditing across race and gender subgroups
- REST API deployment via FastAPI / Flask for EHR integration
- Retrain on modern ICD-10 coded clinical data

---

## 📚 References

- Strack, B. et al. (2014). *Impact of HbA1c Measurement on Hospital Readmission Rates.* BioMed Research International.
- Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* KDD '16.
- Lundberg, S., & Lee, S. (2017). *A Unified Approach to Interpreting Model Predictions (SHAP).* NeurIPS.
- UCI Machine Learning Repository — 130-US Hospitals Diabetes Dataset.

---

*Project completed: April 2025*
