# Application 1 - ARIMA Analysis for Gold Price Time Series

## 1. Unit Root Tests and Transformations

The gold price time series from 1995 to 2016 (5526 observations) was analyzed to identify appropriate ARIMA models. As seen in Figure 1(a), the original gold price data shows a clear non-stationary trend.

Unlike the example with GNI data, where a log transformation was necessary due to exponential growth, our gold price data did not require a log transformation as the variance wasn't increasing proportionally with the level.

### Stationarity Analysis:

- **Original Series**: The Augmented Dickey-Fuller (ADF) test on the original series yielded a p-value of 0.908, confirming non-stationarity.
- **First Difference**: After applying first differencing, the ADF test gave a p-value of 1.32e-27, strongly rejecting the null hypothesis and confirming stationarity.

Therefore, we determined the order of integration to be d=1, meaning we proceed with ARIMA(p,1,q) models.

## 2. Box-Jenkins Methodology and Model Validity

### 2.1 Model Identification

The ACF and PACF plots of the stationary (differenced) series shown in Figure 2 indicate:

- Significant autocorrelation at multiple lags
- Sinusoidal decay pattern in both ACF and PACF, suggesting both AR and MA components

### 2.2 Model Selection

We fitted multiple ARIMA models with different orders and compared them using information criteria:

| Model        | AIC                | BIC                |
| ------------ | ------------------ | ------------------ |
| ARIMA(0,1,2) | 32542.43           | **32561.61** |
| ARIMA(1,1,1) | 32543.12           | 32562.30           |
| ARIMA(1,1,3) | 32532.64           | 32564.61           |
| ARIMA(2,1,2) | **32532.29** | 32564.26           |
| ARIMA(3,1,2) | 32534.23           | 32572.60           |

Based on AIC, ARIMA(2,1,2) is the best model, while ARIMA(0,1,2) is preferred according to BIC. We selected ARIMA(2,1,2) as our final model.

### 2.3 Model Parameters

From the estimation output, our ARIMA(2,1,2) model can be expressed as:

```
(1 + 0.9751B + 0.0825B²)(∇¹Gold_t) = (1 + 0.0869B + 0.9113B²)(ε_t)
```

Where:

- B is the backshift operator
- ∇¹ represents first differencing
- Gold_t is the gold price at time t
- ε_t is the error term at time t

All coefficients are statistically significant at the 1% level.

### 2.4 Model Diagnostics

The residual analysis shows:

- **Ljung-Box Test**: Significant autocorrelation remains in the residuals (p-value < 0.001), suggesting the model doesn't capture all patterns
- **Jarque-Bera Test**: The residuals are not normally distributed (high kurtosis of 28.22)
- **Heteroskedasticity**: The variance of residuals is not constant (p-value < 0.001)

Despite these issues, the model provides useful insights about the gold price dynamics.

## 3. Point Forecast and Confidence Intervals

The ARIMA(2,1,2) model was used to generate forecasts for the test set (20% of the data). The forecast accuracy metrics on the original scale are:

- **Mean Squared Error (MSE)**: 400,594.73
- **Root Mean Squared Error (RMSE)**: 632.93
- **Mean Absolute Percentage Error (MAPE)**: 45.74%

The high MAPE indicates substantial percentage errors in the forecasts, which is not surprising given the volatility of gold prices and the presence of multiple structural breaks in the time series.

## 4. Conclusions and Limitations

1. **Stationarity**: The gold price series requires first differencing to achieve stationarity.
2. **Model Selection**: ARIMA(2,1,2) provides the best fit according to AIC, suggesting that gold price movements are influenced by both autoregressive and moving average components.
3. **Forecast Performance**: The forecasting accuracy is moderate, with a high MAPE of 45.74%. This indicates that while the model captures some patterns in gold prices, it may not be sufficient for highly accurate predictions.
4. **Limitations**:

   - The residuals show signs of remaining autocorrelation and non-normality
   - The model assumes linearity after differencing
   - The model does not account for external factors affecting gold prices (e.g., economic indicators, market sentiment)
5. **Recommendations**:

   - Consider GARCH extensions to handle volatility clustering
   - Explore more complex non-linear models
   - Incorporate external variables that might impact gold prices
   - Consider regime-switching models to account for different market states
