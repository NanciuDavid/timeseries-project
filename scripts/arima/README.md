# ARIMA Analysis for Gold Price Time Series

This directory contains the implementation of ARIMA (AutoRegressive Integrated Moving Average) modeling for gold price time series data using the Box-Jenkins methodology.

## Overview

The analysis follows these steps:

1. **Data Preparation**: Loading and inspecting the gold price dataset.
2. **Stationarity Testing**: Performing unit root tests (Augmented Dickey-Fuller) to check for stationarity.
3. **Transformation**: Applying differencing if needed to achieve stationarity.
4. **Model Identification**: Using ACF and PACF plots to identify potential ARIMA orders.
5. **Model Estimation**: Fitting multiple ARIMA models with different parameters.
6. **Model Selection**: Selecting the best model based on AIC and BIC criteria.
7. **Diagnostic Checking**: Analyzing residuals to validate the model.
8. **Forecasting**: Generating point forecasts and confidence intervals.
9. **Forecast Evaluation**: Assessing forecast accuracy using metrics like RMSE and MAPE.

## Dataset

The analysis uses the Gold Price dataset located at `datasets/gold-part1/Gold.csv`. The dataset contains daily gold prices with the following columns:
- DATE: The date of the observation
- VALUE: The gold price value

## Running the Analysis

To run the ARIMA analysis, execute the following command from the project root directory:

```bash
python scripts/arima/arima_analysis.py
```

## Output

The script generates various plots and outputs saved to the `plots/` directory:

1. `gold_time_series.png`: Plot of the original gold price time series
2. `gold_first_diff.png`: Plot of the first difference of the time series (if needed)
3. `gold_second_diff.png`: Plot of the second difference of the time series (if needed)
4. `gold_acf_pacf.png`: ACF and PACF plots of the stationary series
5. `arima_model_summary.txt`: Summary of the best ARIMA model
6. `residual_diagnostics.png`: Diagnostic plots for the model residuals
7. `residual_acf.png`: ACF plot of the residuals
8. `forecast_vs_actual.png`: Plot comparing forecasts with actual values
9. `forecast_original_scale.png`: Forecasts converted back to the original scale (if differencing was applied)

## Requirements

The following Python libraries are required:
- pandas
- numpy
- matplotlib
- seaborn
- statsmodels
- scikit-learn

## Box-Jenkins Methodology

The analysis follows the Box-Jenkins methodology for time series modeling:

1. **Model Identification**: Determine if the series is stationary and identify potential ARIMA orders using ACF and PACF plots.
2. **Parameter Estimation**: Fit the ARIMA model and estimate parameters.
3. **Diagnostic Checking**: Check residuals for white noise characteristics.
4. **Forecasting**: Use the validated model to generate forecasts.

## Interpretation

The script provides detailed output about:
- Stationarity of the time series
- The best ARIMA model parameters
- Residual analysis results
- Forecast accuracy metrics

The interpretation of these results should focus on:
- Whether the residuals exhibit white noise properties
- The confidence intervals of the forecasts
- The accuracy metrics of the forecasts (RMSE, MAPE) 