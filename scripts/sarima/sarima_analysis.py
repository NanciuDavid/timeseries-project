import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# 1. Load and inspect the data
df = pd.read_csv("datasets/airtraffic-part2/air_traffic_merged.csv")
df['FLT_DATE'] = pd.to_datetime(df['FLT_DATE'], format='%d-%m-%y')
df = df.set_index('FLT_DATE')
time_series_data = df.groupby('FLT_DATE')['FLT_TOT_1'].sum().to_frame()

# 2. Stationarity tests and differencing
def adf_test(series, title=''):
    print(f"\nADF Test for {title}")
    result = adfuller(series.dropna())
    print(f"ADF Statistic: {result[0]}")
    print(f"p-value: {result[1]}")
    if result[1] <= 0.05:
        print("Stationary (reject H0)")
        return True
    else:
        print("Non-stationary (fail to reject H0)")
        return False

# Plot original series
plt.figure(figsize=(12, 6))
plt.plot(time_series_data)
plt.title('Total Daily Flights Over Time')
plt.xlabel('Date')
plt.ylabel('Total Flights')
plt.grid(True)
plt.tight_layout()
plt.savefig('plots/sarima_airtraffic_time_series.png')
plt.close()

# ADF test and differencing
y = time_series_data['FLT_TOT_1']
d = 0
D = 0
s = 7  # weekly seasonality

is_stationary = adf_test(y, 'Original Series')
if not is_stationary:
    y_diff = y.diff().dropna()
    plt.figure(figsize=(12, 6))
    plt.plot(y_diff)
    plt.title('First Difference')
    plt.savefig('plots/sarima_airtraffic_first_diff.png')
    plt.close()
    is_stationary = adf_test(y_diff, 'First Difference')
    if is_stationary:
        d = 1
        y_to_use = y_diff
    else:
        # Try seasonal differencing
        y_seasonal_diff = y.diff(s).dropna()
        plt.figure(figsize=(12, 6))
        plt.plot(y_seasonal_diff)
        plt.title('Seasonal Difference (lag 7)')
        plt.savefig('plots/sarima_airtraffic_seasonal_diff.png')
        plt.close()
        is_stationary = adf_test(y_seasonal_diff, 'Seasonal Difference')
        if is_stationary:
            D = 1
            y_to_use = y_seasonal_diff
        else:
            # Try both
            y_both_diff = y.diff().diff(s).dropna()
            plt.figure(figsize=(12, 6))
            plt.plot(y_both_diff)
            plt.title('First + Seasonal Difference')
            plt.savefig('plots/sarima_airtraffic_both_diff.png')
            plt.close()
            is_stationary = adf_test(y_both_diff, 'First + Seasonal Difference')
            if is_stationary:
                d = 1
                D = 1
                y_to_use = y_both_diff
            else:
                print("Warning: Series may still be non-stationary after differencing.")
                d = 1
                D = 1
                y_to_use = y_both_diff
else:
    y_to_use = y

print(f"\nUsing d={d}, D={D}, s={s} for SARIMA.")

# 3. ACF/PACF plots
plt.figure(figsize=(12, 10))
plt.subplot(211)
plot_acf(y_to_use, ax=plt.gca(), lags=40)
plt.title('ACF')
plt.subplot(212)
plot_pacf(y_to_use, ax=plt.gca(), lags=40)
plt.title('PACF')
plt.tight_layout()
plt.savefig('plots/sarima_airtraffic_acf_pacf.png')
plt.close()

# 4. Train/test split
train_size = int(len(y_to_use) * 0.8)
train, test = y_to_use[:train_size], y_to_use[train_size:]

# 5. SARIMA grid search (simple, for demo)
p = q = P = Q = range(0, 2)
best_aic = float('inf')
best_order = None
best_seasonal_order = None
best_model = None
print("\nGrid search for SARIMA parameters...")
for i in p:
    for k in q:
        for si in P:
            for sk in Q:
                try:
                    model = SARIMAX(train, order=(i, d, k), seasonal_order=(si, D, sk, s), enforce_stationarity=False, enforce_invertibility=False)
                    results = model.fit(disp=False)
                    if results.aic < best_aic:
                        best_aic = results.aic
                        best_order = (i, d, k)
                        best_seasonal_order = (si, D, sk, s)
                        best_model = results
                    print(f"SARIMA{(i, d, k)}x{(si, D, sk, s)} - AIC:{results.aic:.2f}")
                except Exception as e:
                    continue
print(f"\nBest SARIMA order: {best_order} Seasonal: {best_seasonal_order} AIC: {best_aic:.2f}")
with open('plots/sarima_airtraffic_model_summary.txt', 'w') as f:
    f.write(str(best_model.summary()))

# 6. Model diagnostics
best_model.plot_diagnostics(figsize=(12, 8))
plt.tight_layout()
plt.savefig('plots/sarima_airtraffic_diagnostics.png')
plt.close()

# 7. Forecasting
test_steps = len(test)
pred = best_model.get_forecast(steps=test_steps)
pred_mean = pred.predicted_mean
pred_ci = pred.conf_int()

plt.figure(figsize=(12, 6))
plt.plot(train.index, train, label='Train')
plt.plot(test.index, test, label='Test')
plt.plot(test.index, pred_mean, label='Forecast')
plt.fill_between(test.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='pink', alpha=0.3, label='95% CI')
plt.title('SARIMA Forecast vs Actual')
plt.xlabel('Date')
plt.ylabel('Total Flights')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('plots/sarima_airtraffic_forecast.png')
plt.close()

# 8. Forecast accuracy
mse = mean_squared_error(test, pred_mean)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((test.values - pred_mean.values) / test.values)) * 100
print(f"\nForecast Accuracy:\nMSE: {mse:.2f}\nRMSE: {rmse:.2f}\nMAPE: {mape:.2f}%")
with open('plots/sarima_airtraffic_forecast_accuracy.txt', 'w') as f:
    f.write(f"MSE: {mse:.2f}\nRMSE: {rmse:.2f}\nMAPE: {mape:.2f}%\n")

# 9. Residual analysis: Ljung-Box
test_lb = acorr_ljungbox(best_model.resid, lags=[10, 20, 30], return_df=True)
print("\nLjung-Box test for residuals:")
print(test_lb)
with open('plots/sarima_airtraffic_ljungbox.txt', 'w') as f:
    f.write(str(test_lb))

print("\nSARIMA analysis complete! All plots and results are saved in the 'plots/' folder.") 