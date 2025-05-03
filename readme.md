# Country Health and Economy Analysis

This repository contains a regression-based analysis of the relationship between economic prosperity (GDP per capita) and population health (life expectancy) across countries over time. The project is organized into: data wrangling, exploratory visualization, modeling, and evaluation.

A data-driven study that cleans and merges World Bank GDP per capita and life expectancy data, explores key trends through visualizations, and builds regression models (Linear, Random Forest, Gradient Boosting, SVR, and ensemble) to predict life expectancy from economic and temporal features.

---

## 📁 Repository Structure

```
├── data
│   ├── raw
│   │   ├── API_NY.GDP.PCAP.CD_...csv   # Raw GDP per capita data
│   │   └── API_SP.DYN.LE00.IN_...csv   # Raw life expectancy data
│   └── processed
│       └── country_health_econ.csv    # Cleaned, merged dataset
│
├── notebook
│   ├── plots.ipynb                    # Jupyter notebook with EDA visualizations
│   └── notebook.ipynb                 # Additional exploratory notebook
│
├── data_wrangling.py                  # Script to load, clean, and merge indicators
├── modeling.py                        # Script to train and tune regression models
│
├── models                             # Saved scaler and model artifacts
│   ├── scaler.pkl
│   ├── linear_regression.pkl
│   ├── random_forest.pkl
│   ├── gradient_boosting.pkl
│   ├── svr.pkl
│   └── voting_regressor.pkl
│
├── requirements.txt
|
└── README.md                          # Project overview and instructions
```

---

## 📊 Data Description

- **Source:** World Bank Open Data – World Development Indicators  
- **Indicators:**  
  - `GDP per capita (current US$)`  
  - `Life expectancy at birth (years)`

- **Time span:** 1960–2024 (varies by country)  
- **Observations:** One row per country-year with both indicators available

### Variables in `country_health_econ.csv`

| Column             | Description                                           |
|--------------------|-------------------------------------------------------|
| `Country Name`     | Official country name                                 |
| `Country Code`     | ISO 3166-1 alpha-3 country code                       |
| `Year`             | 4-digit calendar year                                 |
| `GDP_per_capita`   | GDP per person in current US dollars                  |
| `Life_Expectancy`  | Average lifespan in years                             |

After cleaning:  
- `GDP_per_capita_log` is computed as `log1p(GDP_per_capita)` to reduce skew.  
- One-hot encoding applied to `Country Code` for modeling.

---

## 🔍 Modeling Approach

1. **Preprocessing:**  
   - Log-transform `GDP_per_capita`  
   - One-hot encode `Country Code`  
   - Include `Year` as a feature  
   - Split data into train (80%) and test (20%) sets  
   - Standardize features

2. **Models & Tuning:**  
   - Linear Regression  
   - Random Forest (tuned hyperparameters)  
   - Gradient Boosting (tuned hyperparameters)  
   - Support Vector Regressor (tuned hyperparameters)  
   - Voting Ensemble of all models

3. **Evaluation Metrics:**  
   - **R² Score**  
   - **MAE** (Mean Absolute Error)  
   - **MSE** (Mean Squared Error)

---

## 🏆 Results

| Model               | R² Score | MAE   | MSE    |
|---------------------|----------|-------|--------|
| Linear Regression   | 0.6273   | 4.968 | 45.727 |
| Random Forest       | 0.6648   | 4.519 | 41.120 |
| Gradient Boosting   | 0.6672   | 4.516 | 40.830 |
| SVR                 | 0.6312   | 4.463 | 45.247 |
| Voting Ensemble     | 0.6800   | 4.504 | 40.729 |

*Insight:* The ensemble achieves the best performance (R² ≈ 0.68).

---

## 🚀 How to Run

1. **Data Wrangling:**

   ```bash
   # Always show details
   python data_wrangling.py
   ```

2. **Visualization:**

   ```bash
   # Always show details
   jupyter notebook notebook/plots.ipynb
   ```

3. **Modeling:**

   ```bash
   # Always show details
   python modeling.py
   ```
