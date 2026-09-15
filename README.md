# Empirical Modeling and Forecasting of Global Primary Energy Consumption via Gradient-Boosted Decision Trees

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://energy-con.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Abstract

Accurately modeling and forecasting macro-scale energy consumption is vital for national grid planning, energy security, and international climate decarbonization policy. This study presents a comprehensive machine learning framework evaluating **22,012 country-year observations across 129 economic and infrastructural variables** sourced from the **Our World in Data (OWID) Energy Dataset**. We formulate and benchmark five distinct predictive architectures: Ordinary Least Squares (OLS), L2-regularized Ridge Regression, L1-regularized Lasso Regression, Random Forest Ensembles, and **Gradient-Boosted Decision Trees (GBDT)**. Incorporating nonlinear feature transformations, logarithmic scaling, and macroeconomic interaction terms ($\text{GDP}_{\text{per capita}} \times \text{Renewable Share}$), the optimized Gradient Boosting model achieves a state-of-the-art coefficient of determination of **$R^2 = 0.9998$**. The deployed platform provides an interactive Security and Energy Operations dashboard featuring an empirical **Country Energy Transition Explorer** and a **Green Energy Policy "What-If" Simulator** capable of estimating multi-year energy trajectories and avoided greenhouse gas ($\text{CO}_2$) emissions.

**Keywords:** Energy Economics, Primary Energy Consumption, Gradient Boosting, Decarbonization Policy, Macroeconomic Modeling, Scikit-Learn, Climate Informatics.

---

## 🚀 Live Interactive Dashboard
The complete predictive analytics suite is accessible on Streamlit Cloud:  
👉 **[https://energy-con.streamlit.app/](https://energy-con.streamlit.app/)**

---

## Table of Contents
- [1. Introduction & Global Policy Context](#1-introduction--global-policy-context)
- [2. Dataset Architecture & Macro Indicators](#2-dataset-architecture--macro-indicators)
- [3. Mathematical Formulations & Feature Engineering](#3-mathematical-formulations--feature-engineering)
  - [3.1 Gradient Boosted Regression Trees](#31-gradient-boosted-regression-trees)
  - [3.2 Regularized Linear Baselines](#32-regularized-linear-baselines)
  - [3.3 Feature Engineering & Interaction Dynamics](#33-feature-engineering--interaction-dynamics)
- [4. Empirical Results & Model Comparison](#4-empirical-results--model-comparison)
- [5. Interactive Capabilities & Policy Simulator](#5-interactive-capabilities--policy-simulator)
- [6. Project Structure](#6-project-structure)
- [7. Installation & Local Execution](#7-installation--local-execution)
- [8. Academic References & Citations](#8-academic-references--citations)

---

## 1. Introduction & Global Policy Context

The dual imperatives of economic development and decarbonization under the Paris Agreement demand precise forecasting tools. Energy demand is driven by non-linear relationships between gross domestic product (GDP), demographic growth, industrialization index, and renewable infrastructure buildouts. 

This research develops an empirical data science architecture that:
1. Uncovers multi-decade shifts from fossil-dominant energy regimes to clean energy mixes across 200+ nations.
2. Formulates non-linear regression models that map macroeconomic inputs directly to primary energy consumption (TWh).
3. Provides actionable scenario-based simulation tools for energy economists and policymakers.

---

## 2. Dataset Architecture & Macro Indicators

The framework analyzes the longitudinal **World Energy Consumption Dataset** curated by *Our World in Data (OWID)*:

| Dimension | Specification | Description |
|:---|:---:|:---|
| **Total Observations** | **22,012** | Sequential country-year records spanning 1900 to present |
| **Total Features** | **129** | Production, trade, consumption, and economic indicators |
| **Target Variable** | $y$ | Primary Energy Consumption ($\text{TWh}$) |
| **Primary Predictors** | $\mathbf{x}$ | GDP, Population, Energy per Capita, Fossil Fuel Volume, Renewables % |

```
Key Feature Groups:
├── Demographics & Economy : [population, gdp, gdp_per_capita]
├── Conventional Energy   : [fossil_fuel_consumption, coal_consumption, oil_consumption, gas_consumption]
├── Clean Energy Mix      : [renewables_consumption, hydro_consumption, solar_consumption, wind_consumption, nuclear_consumption]
└── Structural Metrics    : [energy_per_capita, renewables_share_energy, energy_intensity]
```

---

## 3. Mathematical Formulations & Feature Engineering

### 3.1 Gradient Boosted Regression Trees
The primary predictor trains an additive model of $M$ decision trees:

$$F_M(x) = \sum_{m=1}^M \gamma_m h_m(x)$$

where each tree $h_m(x)$ fits the pseudo-residuals of the previous ensemble under the Mean Squared Error loss function $\mathcal{L}(y, F(x)) = \frac{1}{2}(y - F(x))^2$:

$$r_{im} = -\left[ \frac{\partial \mathcal{L}(y_i, F(x_i))}{\partial F(x_i)} \right]_{F(x) = F_{m-1}(x)} = y_i - F_{m-1}(x_i)$$

The optimal multiplier $\gamma_m$ for each terminal region is computed via line search:

$$\gamma_m = \arg\min_\gamma \sum_{i=1}^n \mathcal{L}(y_i, F_{m-1}(x_i) + \gamma h_m(x_i))$$

### 3.2 Regularized Linear Baselines
To evaluate parametric baselines, we train L1 (Lasso) and L2 (Ridge) regularized estimators:

$$\text{Ridge Objective:} \quad \min_{\mathbf{w}} \|\mathbf{y} - X\mathbf{w}\|_2^2 + \lambda_2 \|\mathbf{w}\|_2^2$$

$$\text{Lasso Objective:} \quad \min_{\mathbf{w}} \frac{1}{2n} \|\mathbf{y} - X\mathbf{w}\|_2^2 + \lambda_1 \|\mathbf{w}\|_1$$

### 3.3 Feature Engineering & Interaction Dynamics
1. **Variance-Stabilizing Logarithmic Transforms**: Highly skewed macroeconomic distributions are normalized via:
   $$x' = \ln(1 + x)$$
   applied to $\text{GDP}$, $\text{Population}$, and $\text{GDP}_{\text{per capita}}$.

2. **Energy Intensity Index**:
   $$\text{EI} = \frac{\text{Primary Energy Consumption}}{\text{GDP}}$$

3. **Macro-Renewable Interaction Term**:
   $$\text{Interaction} = \text{GDP}_{\text{per capita}} \times \text{Renewable Share}$$
   capturing the empirical principle that wealthier nations convert higher shares of GDP into renewable generation infrastructure.

---

## 4. Empirical Results & Model Comparison

All models were evaluated under identical 5-fold cross-validation protocol using the coefficient of determination ($R^2$):

| Model Architecture | Parameter Settings | $R^2$ Score | Convergence Speed |
|:---|:---|:---:|:---:|
| **Gradient Boosting Regressor** | $n=300$, max_depth$=5$, $\eta=0.05$ | **0.9998** | Moderate |
| **Random Forest Regressor** | $n=200$, max_depth$=12$ | **0.9992** | Moderate |
| **Ridge Regression** | $\alpha = 1.0$ (L2 norm) | $0.5046$ | Fast |
| **Lasso Regression** | $\alpha = 0.01$ (L1 sparsity) | $0.4933$ | Fast |
| **Linear Regression (OLS)** | Standard Least Squares | $0.4928$ | Instant |

```
Key Empirical Finding:
Ensemble tree architectures capture the extreme non-linear power laws 
governing industrial energy consumption that linear estimators systematically underfit.
```

---

## 5. Interactive Capabilities & Policy Simulator

The deployed web application provides five operational analytical modules:

1. **🌿 Green Energy Policy "What-If" Simulator**:
   - Allows users to select any nation (e.g. Bangladesh, United States, Germany, India, China) and dynamically test 5-year policy interventions.
   - Adjusts Renewable Energy Share targets ($+5\%$ to $+50\%$), Energy Intensity reduction, and GDP growth.
   - Computes forecasted energy consumption alongside annual **avoided $\text{CO}_2$ emissions** (abatement estimate based on $0.72\text{ Mt CO}_2/\text{TWh}$).
2. **🌐 Interactive Country Energy Transition Explorer**:
   - Queries historical records to generate dynamic Plotly multi-source energy stack charts (Fossil Fuels, Hydro, Nuclear, Solar/Wind).
3. **🔮 Custom Parameter Prediction Engine**:
   - Manual feature estimation with automatic population-density and interaction parameter synthesis.
4. **📈 Real-Time Model Comparison**:
   - Live visual evaluation of linear vs. tree ensemble benchmarks.

---

## 6. Project Structure

```
Energy_Consumption_Analysis/
├── app.py                                           # Streamlit interactive dashboard & policy lab
├── Energy_Consumption_Analysis.ipynb               # Exploratory data analysis & baseline modeling
├── Improved_Energy_Consumption_Prediction.ipynb     # Gradient Boosting & hyperparameter tuning
├── World Energy Consumption.csv                     # Our World in Data benchmark dataset (22,012 rows)
├── requirements.txt                                 # Optimized Python dependencies
├── README.md                                        # Academic research documentation
├── .gitignore                                       # Git exclusion rules
├── .github/
│   └── workflows/
│       └── keep_alive.yml                           # 24/7 Playwright keep-alive bot
├── models/                                          # Serialized model binaries
│   ├── best_model.pkl
│   ├── gradientboosting_model.pkl
│   ├── randomforest_model.pkl
│   └── model_performance_comparison.csv
└── images/                                          # Analytical and diagnostic figures
```

---

## 7. Installation & Local Execution

### Prerequisites
- Python 3.10+
- pip package manager

```bash
# Clone the repository
git clone https://github.com/bipulhstu/Energy_Consumption_Analysis.git
cd Energy_Consumption_Analysis

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit Dashboard
streamlit run app.py
```

---

## 8. Academic References & Citations

1. **Ritchie, H., Roser, M., & Rosado, P.** (2020). *Energy.* Published online at OurWorldInData.org. Retrieved from: [https://ourworldindata.org/energy](https://ourworldindata.org/energy).
2. **Friedman, J. H.** (2001). *Greedy function approximation: A gradient boosting machine.* The Annals of Statistics, 29(5), pp. 1189–1232. DOI: [10.1214/aos/1013203451](https://doi.org/10.1214/aos/1013203451).
3. **Breiman, L.** (2001). *Random Forests.* Machine Learning, 45(1), pp. 5–32. DOI: [10.1023/A:1010933404324](https://doi.org/10.1023/A:1010933404324).
4. **International Energy Agency (IEA)**. (2023). *World Energy Outlook 2023.* IEA Publications, Paris.
5. **Pedregosa, F. et al.** (2011). *Scikit-learn: Machine Learning in Python.* Journal of Machine Learning Research, 12, pp. 2825–2830.

---

**🌍 Dedicated to Open-Source Climate Data Science & Energy Transition Research**
