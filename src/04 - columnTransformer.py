import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_csv("../data/raw/housing.csv")
print(df.head())

# Stratified test set
df["income_cat"] = pd.cut(
    df["median_income"], bins=[0, 1.5, 3.0, 4.5, 6.0, np.inf], labels=[1, 2, 3, 4, 5]
)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in split.split(df, df["income_cat"]):
    strat_train_set = df.loc[train_idx].drop("income_cat", axis=1)
    strat_test_set = df.loc[test_idx].drop("income_cat", axis=1)

housing = strat_train_set.copy()

# Sperate features and labels

housing_labels = housing["median_house_value"].copy()
housing = housing.drop("median_house_value", axis=1)

# Seperate numerical and categorical columns
housing_num = housing.drop("ocean_proximity", axis=1).columns.tolist()
housing_cat = ["ocean_proximity"]

# Pipeline for numerical cols

num_pipeline = Pipeline(
    [("impute", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

# Pipeline for categorical columns

cat_pipeline = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))])

# Construct the full pipeline

full_pipeline = ColumnTransformer(
    [("num", num_pipeline, housing_num), ("cat", cat_pipeline, housing_cat)]
)

# Transform the data

housing_processed = full_pipeline.fit_transform(housing)
