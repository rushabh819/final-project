#!/usr/bin/env python3
"""
modeling.py

Loads the processed dataset, one-hot encodes countries,
splits into train/test, scales features,
runs streamlined hyperparameter tuning on multiple models
to predict life expectancy,
evaluates performance, and saves results.
"""
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib


def load_data(path="data/processed/country_health_econ.csv"):
    """Load processed CSV, log-transform GDP, one-hot encode Country Code"""
    df = pd.read_csv(path)
    df['GDP_per_capita_log'] = np.log1p(df['GDP_per_capita'])
    X_base = df[['GDP_per_capita_log', 'Year']]
    country_dummies = pd.get_dummies(df['Country Code'], prefix='CC', drop_first=True)
    X = pd.concat([X_base, country_dummies], axis=1)
    y = df['Life_Expectancy']
    return X, y


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    print(f"Model: {model.__class__.__name__}")
    print(f"  R2 Score:  {r2_score(y_test, preds):.4f}")
    print(f"  MAE:       {mean_absolute_error(y_test, preds):.4f}")
    print(f"  MSE:       {mean_squared_error(y_test, preds):.4f}\n")
    return preds


def main():
    os.makedirs("models", exist_ok=True)
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    joblib.dump(scaler, "models/scaler.pkl")

    models = {}
    
    # Linear Regression
    lin = LinearRegression()
    lin.fit(X_train, y_train)
    models['LinearRegression'] = lin

    # Random Forest (streamlined search)
    rf = RandomForestRegressor(random_state=42)
    rf_params = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10],
        'max_features': ['sqrt']
    }
    rf_search = RandomizedSearchCV(rf, rf_params, n_iter=4, cv=3, scoring='r2', random_state=42, n_jobs=-1)
    rf_search.fit(X_train, y_train)
    best_rf = rf_search.best_estimator_
    print(f"Best RF params: {rf_search.best_params_}\n")
    models['RandomForest'] = best_rf

    # Gradient Boosting (streamlined search)
    gb = GradientBoostingRegressor(random_state=42)
    gb_params = {
        'n_estimators': [100],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3]
    }
    gb_search = RandomizedSearchCV(gb, gb_params, n_iter=2, cv=3, scoring='r2', random_state=42, n_jobs=-1)
    gb_search.fit(X_train, y_train)
    best_gb = gb_search.best_estimator_
    print(f"Best GB params: {gb_search.best_params_}\n")
    models['GradientBoosting'] = best_gb

    # SVR (simplified grid, fewer folds)
    svr = SVR()
    svr_params = {'C': [10, 100], 'gamma': ['scale'], 'kernel': ['rbf']}
    svr_search = GridSearchCV(svr, svr_params, cv=3, scoring='r2', n_jobs=-1)
    svr_search.fit(X_train, y_train)
    best_svr = svr_search.best_estimator_
    print(f"Best SVR params: {svr_search.best_params_}\n")
    models['SVR'] = best_svr

    # Voting Ensemble
    voting = VotingRegressor(estimators=[(name, model) for name, model in models.items()])
    voting.fit(X_train, y_train)
    models['VotingRegressor'] = voting

    # Evaluate and save
    for name, model in models.items():
        evaluate_model(model, X_test, y_test)
        joblib.dump(model, f"models/{name}.pkl")

    print("All models evaluated and saved in models/.")

if __name__ == '__main__':
    main()
