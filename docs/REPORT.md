# House Price Prediction — Technical Report

## 1. Executive Summary

This project builds a regression model to predict the median house value of a California district from census-derived features (location, housing age, room counts, population, income, and proximity to the ocean). After comparing three baseline models, running a feature engineering experiment, and tuning hyperparameters, the final model — a tuned Random Forest Regressor — achieves an RMSE of **$46,994.54** and an R² of **0.831** on a held-out test set it never influenced during training or tuning.

## 2. Objective

Predict `median_house_value` for a California district using demographic and housing features. The intended use case is a learning project demonstrating a full, leakage-free ML workflow — not a production valuation tool (see Limitations, Section 10).

## 3. Dataset

The California Housing dataset, derived from the 1990 U.S. Census (one row per census block group / "district"). It contains 9 raw features plus the target:

| Feature | Description |
|---|---|
| `longitude`, `latitude` | Geographic coordinates |
| `housing_median_age` | Median age of houses in the district |
| `total_rooms`, `total_bedrooms` | Aggregate room counts for the district |
| `population`, `households` | District population and household counts |
| `median_income` | Median income (tens of thousands of USD) |
| `ocean_proximity` | Categorical: distance/relation to the ocean |
| `median_house_value` (target) | Median house value for the district |

## 4. Methodology

### 4.1 Exploratory Data Analysis

Initial inspection covered structure, dtypes, missing values, and distributions. Two dataset quirks are worth flagging explicitly, since they affect how results should be interpreted:

- **`total_bedrooms` contains missing values**, handled later via median imputation.
- **`median_house_value` is capped** at $500,001 — a real artifact of how the original census data was published, not noise. Districts at or near the cap are under-represented in terms of true value, and the model cannot learn values above the cap because none exist in training data. This caps the model's practical ceiling and is the most likely source of the largest errors in Section 9.

A geographic scatter plot (`longitude`/`latitude`, colored by price) showed clear clustering of high-value districts along the coast, consistent with `median_income` and `ocean_proximity` being the strongest predictors — confirmed by correlation analysis.

### 4.2 Train/Test Split

A plain random split risks skewing the income distribution between train and test sets, since `median_income` is not uniformly distributed. To avoid this, `median_income` was binned into 5 categories and `StratifiedShuffleSplit` (80/20, `random_state=42`) was used so both sets carry a representative income distribution.

- Training set: **16,512 rows**
- Test set: **4,128 rows**

The test set was set aside after this split and not used again until Section 8.

### 4.3 Preprocessing Pipeline

A `ColumnTransformer` combining two sub-pipelines was used, so the exact same transformation logic is guaranteed to apply consistently to training data, test data, and any future input:

- **Numeric columns** (8 features): median imputation → `StandardScaler`
- **Categorical column** (`ocean_proximity`, 5 categories): `OneHotEncoder`

This produces **13 processed features** (8 numeric + 5 one-hot columns) from the 9 raw input columns, confirmed by the transformed shape `(16512, 13)`.

## 5. Baseline Model Comparison

Three models were trained on the processed training set and evaluated two ways: naively (predict on the same data used for training) and honestly (5-fold cross-validation).

| Model | Training RMSE (in-sample) | 5-Fold CV RMSE | CV Std Dev |
|---|---:|---:|---:|
| Linear Regression | 69,050.56 | 69,218.45 | 689.50 |
| Decision Tree | **0.00** | 69,593.04 | 1,215.06 |
| Random Forest | 18,419.50 | **49,885.18** | 852.12 |

**Key finding:** the Decision Tree's in-sample RMSE of 0.00 is a textbook overfitting signature — the tree memorized the training data rather than learning a generalizable pattern. Its true performance, revealed only by cross-validation, is actually no better than Linear Regression. This is the clearest illustration in the project of why models must never be evaluated on the data they were trained on.

Random Forest was the clear winner by cross-validated RMSE and was carried forward for tuning.

## 6. Feature Engineering Experiment

Three ratio features were engineered from existing columns, on the hypothesis that ratios are more directly informative than raw counts:

- `rooms_per_household` = `total_rooms / households`
- `bedrooms_per_room` = `total_bedrooms / total_rooms`
- `population_per_household` = `population / households`

| Model | CV RMSE (original) | CV RMSE (with engineered features) | Change |
|---|---:|---:|---:|
| Linear Regression | 69,218.45 | 68,433.91 | **−784.54** (improved) |
| Decision Tree | 69,593.04 | 70,859.65 | +1,266.61 (worse) |
| Random Forest | 49,885.18 | 50,690.46 | **+805.28 (worse)** |

**Interpretation:** the engineered features helped Linear Regression but hurt the tree-based models, including the selected Random Forest. This is a reasonable outcome, not a bug — linear models can only combine features additively, so an explicit ratio gives them information they cannot otherwise construct. Tree-based models, by contrast, can already approximate a ratio's effect by splitting sequentially on the two raw columns, so the added features mostly contribute redundant, correlated signal — which can dilute split quality and add noise rather than information. Since Random Forest was the selected model, **these engineered features were not included in the final pipeline**, keeping it simpler with no loss in performance.

## 7. Hyperparameter Tuning

Tuning was performed on Random Forest using the **original (non-engineered) features**, matching the model configuration that will actually ship.

### 7.1 GridSearchCV

Searched `n_estimators` ∈ {3, 10, 30}, `max_features` ∈ {2, 4, 6, 8}, plus a second grid with `bootstrap=False`.

- Best parameters: `{max_features: 8, n_estimators: 30}`
- Best CV RMSE: **49,933.76**

### 7.2 RandomizedSearchCV

Searched a wider space (`n_estimators`, `max_features`, `max_depth`, `min_samples_split`) with `n_iter=10`, `cv=5`.

- Best parameters: `{n_estimators: 50, min_samples_split: 5, max_features: 8, max_depth: 20}`
- Best CV RMSE: **49,673.85**

RandomizedSearchCV found a marginally better configuration than GridSearchCV within a larger search space — these parameters were adopted for the final model.

## 8. Final Model

```
RandomForestRegressor(
    n_estimators=50,
    max_features=8,
    max_depth=20,
    min_samples_split=5,
    random_state=42,
)
```

Trained on the full training set (16,512 rows) using the original 8 numeric + 1 categorical features — no engineered features, per Section 6's finding.

## 9. Final Test Set Evaluation

The tuned model was evaluated exactly once against the test set held out since Section 4.2.

| Metric | Value | Interpretation |
|---|---:|---|
| RMSE | $46,994.54 | Typical prediction error magnitude, in the same units as house value, penalizing large errors more heavily |
| MAE | $31,295.90 | Average absolute error — less sensitive to outliers than RMSE |
| R² | 0.8305 | ~83% of the variance in house value is explained by the model |

The gap between RMSE ($46,994.54) and MAE ($31,295.90) indicates the error distribution has a meaningful tail of larger misses rather than uniformly moderate errors — expected given the price-cap artifact noted in Section 4.1 and natural regional pricing variance.

### 9.1 Error Analysis

Two diagnostics were produced from the test-set predictions:

- **Actual vs. Predicted scatter plot** — points cluster near the diagonal (good agreement) for most of the price range, with visible spread widening at higher actual values — the model tends to underpredict the most expensive districts, consistent with fewer high-value training examples and the $500,001 cap compressing the top of the distribution.
- **Error histogram** — roughly centered near zero, suggesting no strong systematic bias, though with a longer tail than a normal distribution would predict, reflecting the same high-value underprediction pattern.

A sample of individual errors (from `error_analysis.head()`) confirms this qualitatively: most errors are in the low thousands, with occasional errors exceeding $50,000 on individual districts.

## 10. Limitations

- **Price-cap artifact:** `median_house_value` is capped at $500,001 in the source data. The model cannot learn to predict above this ceiling, and districts near/at the cap likely contribute disproportionately to the model's largest errors. Neither the tutorial nor this project removed or flagged these rows before training — a natural next experiment.
- **No monetary confidence intervals:** the model returns a point estimate only; no prediction interval or uncertainty quantification is provided.
- **Census-block granularity:** predictions are for district medians (aggregates of many households), not individual homes — the model cannot account for individual-property features like square footage, condition, or amenities, since the dataset doesn't contain them.
- **1990 data:** the dataset reflects 1990 census-era prices and demographics; it is not representative of current housing markets and shouldn't be used to size real transactions.

## 11. Conclusion

A tuned Random Forest Regressor, trained on the original (non-engineered) feature set, was selected as the final model based on cross-validated performance, and achieved an R² of 0.83 on a genuinely held-out test set. The project's main technical lessons — the danger of in-sample evaluation (Section 5), and that feature engineering helps some model families more than others (Section 6) — are reflected directly in the final modeling choices rather than treated as side notes.

## 12. Future Work

- Update `src/train.py` to reproduce the exact tuned final model
- Add `src/predict.py` for inference on new, single-house input
- Investigate removing or separately modeling capped districts (`median_house_value >= 500001`)
- Try gradient-boosted models (e.g. `XGBoost`, `HistGradientBoostingRegressor`) as a stronger baseline than Random Forest
- Convert to a web app and deploy

## Appendix: Notebook Reference

| Notebook | Purpose |
|---|---|
| `01_-_eda.ipynb` | Exploratory data analysis |
| `02_-_data_preprocessing.ipynb` | Missing values, initial preprocessing |
| `03_-_model_pipelines.ipynb` | Building the numeric/categorical pipelines |
| `04_-_columnTransformer.ipynb` | Consolidating pipelines with `ColumnTransformer` |
| `05_-_model_training.ipynb` | Baseline models, cross-validation, hyperparameter tuning |
| `06_-_feature_engineering.ipynb` | Engineered-feature experiment |
| `07_-_final_model_evaluation.ipynb` | Final model, test evaluation, error analysis, model saving |
