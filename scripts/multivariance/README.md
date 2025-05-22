# Multivariate Time Series Analysis for Romanian Inflation and Exchange Rates

This directory contains the implementation of multivariate time series analysis for Romanian inflation and exchange rate data, covering concepts like non-stationarity, cointegration, Granger causality, VAR/VECM models, and impulse response functions.

## Overview

The analysis follows these steps:

1. **Data Preparation**: Loading and inspecting Romanian inflation and exchange rate data.
2. **Visualization**: Plotting time series and examining relationships between variables.
3. **Non-stationary Analysis**: Checking for stationarity with Augmented Dickey-Fuller tests.
4. **Cointegration Analysis**: Testing for long-term equilibrium relationships between variables.
5. **Granger Causality Analysis**: Examining causal relationships between inflation and exchange rates.
6. **VAR/VECM Modeling**: Fitting either Vector Autoregression (VAR) or Vector Error Correction Models (VECM) depending on cointegration results.
7. **Impulse Response Functions**: Analyzing how variables respond to shocks in the system.
8. **Forecast Evaluation**: Assessing the predictive performance of the models.

## Dataset

The analysis uses two key datasets:

- Romanian inflation rate data: `datasets/romanian-part3/inflation_rate.csv`
- Romanian exchange rate data: `datasets/romanian-part3/exchange_rate.csv`

## Running the Analysis

To run the multivariate analysis, execute the following command from the project root directory:

```bash
python scripts/multivariance/mv_analysis.py
```

## Output

The script generates various plots and outputs saved to the `plots/multivariate/` directory:

1. `time_series_plots.png`: Plots of the original inflation and exchange rate time series
2. `scatter_relationship.png`: Scatter plot showing the relationship between variables
3. `inflation_diff.png`/`exchange_diff.png`: Plots of differenced series (if needed)
4. `var_model_summary.txt` or `vecm_model_summary.txt`: Summary of the fitted model
5. `var_forecast.png` or `vecm_forecast.png`: Visualization of forecasts vs actual values
6. `var_irf.png` or `vecm_irf.png`: Impulse response function plots

## Methodology

### 1. Non-stationary Analysis

We use the Augmented Dickey-Fuller test to check for stationarity in both time series. If the series are non-stationary, we apply differencing to achieve stationarity.

### 2. Cointegration Analysis

We test for cointegration between inflation and exchange rates using the Engle-Granger approach. If the series are cointegrated, it suggests a long-term equilibrium relationship.

### 3. Granger Causality

We test whether past values of one variable help predict future values of another variable, indicating potential causal relationships.

### 4. VAR/VECM Models

- If the series are not cointegrated, we fit a Vector Autoregression (VAR) model to the stationary (differenced) series.
- If the series are cointegrated, we fit a Vector Error Correction Model (VECM) to capture both long-term relationships and short-term dynamics.

### 5. Impulse Response Functions

We analyze how shocks to one variable affect both variables over time, providing insights into the dynamic relationships between inflation and exchange rates.

This script provides a comprehensive framework for analyzing the relationship between Romanian inflation and exchange rates using state-of-the-art time series methods.
