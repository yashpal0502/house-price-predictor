# 🏠 House Price Prediction

A machine learning project that predicts median house values using the California Housing dataset, built end-to-end: exploratory data analysis, preprocessing, model comparison, cross-validation, a feature engineering experiment, hyperparameter tuning, and final test-set evaluation.

## 📌 Status

✅ ML pipeline complete — final Random Forest model trained, tuned, and evaluated on a held-out test set.
🚧 Web app conversion planned as a future step (not started yet).

## 📊 Results at a Glance

| Metric | Value |
|---|---|
| Final model | Random Forest Regressor (tuned) |
| Test RMSE | **$46,994.54** |
| Test MAE | **$31,295.90** |
| Test R² | **0.831** |

The final model explains about **83% of the variance** in median house value on data it never saw during training or tuning.

## 📂 Project Structure

```
house-price-prediction/
├── data/
│   ├── raw/
│       └── housing.csv
│   
├── models/
│   ├── final_model.joblib
│   └── preprocessing_pipeline.joblib
├── notebooks/
│   ├── 01_-_eda.ipynb
│   ├── 02_-_data_preprocessing.ipynb
│   ├── 03_-_model_pipelines.ipynb              
│   ├── 04_-_columnTransformer.ipynb     
│   ├── 05_-_model_training.ipynb
│   ├── 06_-_feature_engineering.ipynb
│   └── 07_-_final_model_evaluation.ipynb
├── src/
│   ├── preprocessing.py
│   └── train.py
├── docs/
│   └── REPORT.md
├── requirements.txt
├── .gitignore
└── README.md
```

## 🔄 Workflow

```
Raw data → EDA → Stratified train/test split → Preprocessing pipeline
  (median imputation + scaling for numeric, one-hot encoding for categorical)
→ Baseline model comparison → 5-fold cross-validation
→ Feature engineering experiment (kept separate — see report)
→ Hyperparameter tuning (GridSearchCV + RandomizedSearchCV)
→ Final Random Forest → Test set evaluation → Error analysis → Model saved
```

For the full methodology, findings, and interpretation, see **[docs/REPORT.md](docs/REPORT.md)**.

## 🤖 Models Compared

| Model | 5-Fold CV RMSE (original features) |
|---|---|
| Linear Regression | 69,218.45 ± 689.50 |
| Decision Tree | 69,593.04 ± 1,215.06 |
| **Random Forest** | **49,885.18 ± 852.12** |

Random Forest was selected and tuned further — see the report for the full hyperparameter search and why the feature-engineering experiment was ultimately left out of the final model.

## 🚀 Getting Started

1. Clone the repo and enter it:
   ```bash
   git clone https://github.com/<yashpal0502>/house-price-predictor.git
   cd house-price-prediction
   ```

2. Set up a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Place `housing.csv` in `data/raw/`.

## 🛠️ Technologies

Python · pandas · NumPy · scikit-learn · matplotlib · Jupyter · joblib

## 🔮 Future Work

- [ ] Add `src/predict.py` for inference on new, unseen houses
- [ ] Convert to a web app (Flask/FastAPI/Streamlit)
- [ ] Deploy the model

## 👤 Author

**Yashpal**
