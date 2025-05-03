# Models Directory

This folder contains serialized model artifacts and the scaler used in the Country Health and Economy Analysis project.

## File Descriptions

| Filename                          | Description                                                                                                                                       |
|-----------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------|
| `scaler.pkl`                      | A `StandardScaler` fitted on the training set features (`GDP_per_capita_log`, `Year`, and one-hot country codes). Use this to scale new inputs. |
| `linear_regression.pkl`           | Baseline `LinearRegression` model capturing the average linear relationship between GDP and life expectancy.                                        |
| `random_forest_optimized.pkl`     | `RandomForestRegressor` with tuned hyperparameters (`n_estimators`, `max_depth`, `max_features`), excellent at modeling non-linear effects.      |
| `gradient_boosting_optimized.pkl` | `GradientBoostingRegressor` with tuned hyperparameters (`n_estimators`, `learning_rate`, `max_depth`), sequentially corrects previous errors.     |
| `svr_optimized.pkl`               | `SVR` (Support Vector Regression) with RBF kernel and optimized `C` and `gamma`, robust to outliers and useful for small-to-medium datasets.     |
| `voting_regressor.pkl`            | `VotingRegressor` ensemble that averages predictions from Linear, Random Forest, Gradient Boosting, and SVR for balanced performance.             |

## Usage

Load the scaler and any model using `joblib`:

```python
import joblib

# Load scaler
dirs = 'models'
scaler = joblib.load(f'{dirs}/scaler.pkl')

# Load a model (e.g., Random Forest)
rf_model = joblib.load(f'{dirs}/random_forest_optimized.pkl')

# Prepare new data
# - Compute log: data['GDP_per_capita_log'] = np.log1p(data['GDP_per_capita'])
# - Include Year and one-hot encode Country Code
# - Create DataFrame X_new with same columns used in training

X_scaled = scaler.transform(X_new)
y_pred = rf_model.predict(X_scaled)

