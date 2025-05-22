# ARIMA Analysis of US GDP Data

## Data Characteristics
- Time Series: Quarterly US GDP
- Time Period: 1947-01-01 to 2025-01-01
- Number of Observations: 313

## Stationarity Analysis
- Original Series: Non-stationary
- Differencing Required: d = 1

## Model Selection
- Complex model: ARIMA(3, 1, 3) with AIC = 2532.29
- Simple model: ARIMA(0, 1, 1) with BIC = 2557.23
- Model comparison: complex vs simple

## Forecast Accuracy
- ARIMA(3, 1, 3):
  - RMSE: 489.30
  - MAPE: 1.84%
  - MAE: 383.67
  - Theil's U: 1.1162

- ARIMA(0, 1, 1):
  - RMSE: 496.88
  - MAPE: 1.90%
  - MAE: 394.58
  - Theil's U: 1.1335

## Conclusion
The complex model performed better in terms of forecast accuracy.
A Theil's U value less than 1 indicates that the model performs better than a naive forecast.
