import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

plt.style.use('ggplot')
sns.set_palette("deep")
plt.rcParams['figure.figsize'] = [12, 7]
plt.rcParams['figure.dpi'] = 100

plots_dir = "plots/sarima/"
os.makedirs(plots_dir, exist_ok=True)

def adf_test(series, title=''):
    print(f"Augmented Dickey-Fuller Test for {title}")
    result = adfuller(series.dropna())
    labels = ['ADF Test Statistic', 'p-value', '# Lags Used', '# Observations Used']
    for value, label in zip(result, labels):
        print(f'{label} : {value}')
    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis")
        print("Reject the null hypothesis")
        print("Data is stationary")
        return True
    else:
        print("Weak evidence against the null hypothesis")
        print("Fail to reject the null hypothesis")
        print("Data is non-stationary")
        return False

def sarima_analysis(y, label, plots_dir, s=7):
    print(f"\n--- SARIMA Analysis: {label} ---")
    d = 0
    D = 0
    is_stationary = adf_test(y, title=f'{label} Original Series')
    y_to_use = y.copy()
    if not is_stationary:
        y_diff = y.diff().dropna()
        plt.figure(figsize=(14, 7))
        plt.plot(y_diff.index, y_diff, color='blue')
        plt.title(f'{label} First Difference', fontsize=16)
        plt.savefig(f'{plots_dir}{label}_first_diff.png')
        plt.close()
        is_stationary_diff = adf_test(y_diff, title=f'{label} First Difference')
        if is_stationary_diff:
            d = 1
            y_to_use = y_diff
        else:
            y_seasonal_diff = y.diff(s).dropna()
            plt.figure(figsize=(14, 7))
            plt.plot(y_seasonal_diff.index, y_seasonal_diff, color='blue')
            plt.title(f'{label} Seasonal Difference (lag {s})', fontsize=16)
            plt.savefig(f'{plots_dir}{label}_seasonal_diff.png')
            plt.close()
            is_stationary_seasonal = adf_test(y_seasonal_diff, title=f'{label} Seasonal Difference')
            if is_stationary_seasonal:
                D = 1
                y_to_use = y_seasonal_diff
            else:
                y_both_diff = y.diff().diff(s).dropna()
                plt.figure(figsize=(14, 7))
                plt.plot(y_both_diff.index, y_both_diff, color='blue')
                plt.title(f'{label} First + Seasonal Difference', fontsize=16)
                plt.savefig(f'{plots_dir}{label}_both_diff.png')
                plt.close()
                is_stationary_both = adf_test(y_both_diff, title=f'{label} First + Seasonal Difference')
                if is_stationary_both:
                    d = 1
                    D = 1
                    y_to_use = y_both_diff
                else:
                    print(f"Warning: {label} series may still be non-stationary after differencing.")
                    d = 1
                    D = 1
                    y_to_use = y_both_diff
    # ACF/PACF
    plt.figure(figsize=(14, 10))
    plt.subplot(211)
    plot_acf(y_to_use, ax=plt.gca(), lags=40)
    plt.title(f'{label} ACF', fontsize=16)
    plt.subplot(212)
    plot_pacf(y_to_use, ax=plt.gca(), lags=40)
    plt.title(f'{label} PACF', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{plots_dir}{label}_acf_pacf.png')
    plt.close()
    # Train/test split
    train_size = int(len(y_to_use) * 0.8)
    train, test = y_to_use[:train_size], y_to_use[train_size:]
    # Grid search
    p_values = range(0, 3)
    q_values = range(0, 3)
    P_values = range(0, 2)
    Q_values = range(0, 2)
    best_aic = float("inf")
    best_model = None
    best_order = None
    best_seasonal_order = None
    print(f"\nFitting SARIMA models for {label}...")
    for p in p_values:
        for q in q_values:
            for P in P_values:
                for Q in Q_values:
                    try:
                        model = SARIMAX(train, order=(p, d, q), seasonal_order=(P, D, Q, s), enforce_stationarity=False, enforce_invertibility=False)
                        model_fit = model.fit(disp=False)
                        aic = model_fit.aic
                        if aic < best_aic:
                            best_aic = aic
                            best_model = model_fit
                            best_order = (p, d, q)
                            best_seasonal_order = (P, D, Q, s)
                        print(f"SARIMA({p},{d},{q})x({P},{D},{Q},{s}) - AIC: {aic:.2f}")
                    except Exception as e:
                        continue
    print(f"\nBest model for {label}: SARIMA{best_order}x{best_seasonal_order} with AIC={best_aic:.2f}")
    with open(f'{plots_dir}{label}_sarima_model_summary.txt', 'w') as f:
        f.write(str(best_model.summary()))
    # Diagnostics
    best_model.plot_diagnostics(figsize=(14, 10))
    plt.tight_layout()
    plt.savefig(f'{plots_dir}{label}_sarima_diagnostics.png')
    plt.close()
    # Forecast
    forecast_steps = len(test)
    forecast = best_model.get_forecast(steps=forecast_steps)
    forecast_mean = forecast.predicted_mean
    forecast_ci = forecast.conf_int()
    plt.figure(figsize=(14, 7))
    plt.plot(train.index, train, label='Train', color='blue')
    plt.plot(test.index, test, label='Test', color='green')
    plt.plot(test.index, forecast_mean, label='Forecast', color='red')
    plt.fill_between(test.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='pink', alpha=0.3, label='95% CI')
    plt.title(f'{label} SARIMA Forecast vs Actual', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Total Flights', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{plots_dir}{label}_sarima_forecast_vs_actual.png')
    plt.close()
    # Accuracy
    mse = mean_squared_error(test, forecast_mean)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((test.values - forecast_mean.values) / test.values)) * 100
    print(f"\nForecast Accuracy for {label}:")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")
    with open(f'{plots_dir}{label}_sarima_forecast_accuracy.txt', 'w') as f:
        f.write(f"MSE: {mse:.2f}\nRMSE: {rmse:.2f}\nMAPE: {mape:.2f}%\n")
    # Ljung-Box
    lb_test = acorr_ljungbox(best_model.resid, lags=[10, 20, 30], return_df=True)
    print(f"\nLjung-Box Test for {label} residuals:")
    print(lb_test)
    with open(f'{plots_dir}{label}_sarima_ljungbox.txt', 'w') as f:
        f.write(str(lb_test))
    print(f"\nSARIMA analysis for {label} complete!\n")

# Load and aggregate data
data_path = "datasets/airtraffic-part2/air_traffic_merged.csv"
df = pd.read_csv(data_path)
df['FLT_DATE'] = pd.to_datetime(df['FLT_DATE'], format='%d-%m-%y')
df.set_index('FLT_DATE', inplace=True)
time_series_data = df.groupby('FLT_DATE')['FLT_TOT_1'].sum().to_frame()

# Split periods
dates = time_series_data.index
pre_pandemic = time_series_data.loc[:'2020-02-29']['FLT_TOT_1']
pandemic = time_series_data.loc['2020-03-01':'2021-06-30']['FLT_TOT_1']
post_pandemic = time_series_data.loc['2021-07-01':]['FLT_TOT_1']

print(f"Pre-pandemic: {pre_pandemic.index.min()} to {pre_pandemic.index.max()} ({len(pre_pandemic)} days)")
print(f"Pandemic: {pandemic.index.min()} to {pandemic.index.max()} ({len(pandemic)} days)")
print(f"Post-pandemic: {post_pandemic.index.min()} to {post_pandemic.index.max()} ({len(post_pandemic)} days)")

# SARIMA analysis for pre-pandemic and post-pandemic
def run():
    sarima_analysis(pre_pandemic, 'pre_pandemic', plots_dir)
    sarima_analysis(post_pandemic, 'post_pandemic', plots_dir)

if __name__ == "__main__":
    run() 