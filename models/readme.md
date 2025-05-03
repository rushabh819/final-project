# Models Directory

This directory contains serialized model artifacts and the scaler produced by the modeling pipeline for the Country Health and Economy Analysis project.

## Contents

| File                            | Description                                                                                                       |
|---------------------------------|-------------------------------------------------------------------------------------------------------------------|
| `scaler.pkl`                    | A `StandardScaler` object fitted on training features (`GDP_per_capita_log`, `Year`, and one-hot country codes`). |
| `linear_regression.pkl`         | Baseline `LinearRegression` model capturing the linear relationship between GDP per capita and life expectancy.   |
| `random_forest_optimized.pkl`   | `RandomForestRegressor` with tuned hyperparameters, adept at modeling non-linear trends and interactions.         |
| `gradient_boosting_optimized.pkl` | `GradientBoostingRegressor` with optimized parameters for sequential error correction and robust predictions.    |
| `svr_optimized.pkl`             | `SVR` (Support Vector Regression) with RBF kernel and tuned `C`/`gamma`, suited for complex, small-to-medium data. |
| `voting_regressor.pkl`          | `VotingRegressor` ensemble combining Linear, RF, GB, and SVR models for balanced performance.                      |

## Usage

Load the scaler and a model using `joblib`:

```python
import joblib
import pandas as pd
import numpy as np

# Load artifacts
dirs = 'models'
scaler = joblib.load(f'{dirs}/scaler.pkl')
rf = joblib.load(f'{dirs}/random_forest_optimized.pkl')

# Example new data
data = pd.DataFrame({
    'GDP_per_capita': [10000],
    'Year': [2020],
    'Country Code': ['USA']
})

# Preprocess: log-transform
data['GDP_per_capita_log'] = np.log1p(data['GDP_per_capita'])

# One-hot encode Country Code to match training features
# e.g., X_new = pd.get_dummies(data[['GDP_per_capita_log','Year','Country Code']], prefix='CC')

# Align columns, then scale
X_new = X_new.reindex(columns=feature_columns, fill_value=0)
X_scaled = scaler.transform(X_new)

# Predict
y_pred = rf.predict(X_scaled)
