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

plots_dir = "plots/sarima/personalConsumptionExpenditure/"
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

def sarima_analysis(y, label, plots_dir, s=12): # Changed seasonality to 12 for monthly data
    print(f"\n--- SARIMA Analysis: {label} ---")
    d = 0
    D = 0
    is_stationary = adf_test(y, title=f'{label} Original Series')
    y_to_use = y.copy()
    if not is_stationary:
        y_diff = y.diff().dropna()
        plt.figure(figsize=(14, 7))
        plt.plot(y_diff.index, y_diff, color='blue')
        plt.title(f'{label} First Difference - PCE', fontsize=16)
        plt.savefig(f'{plots_dir}{label}_pce_first_diff.png')
        plt.close()
        is_stationary_diff = adf_test(y_diff, title=f'{label} First Difference')
        if is_stationary_diff:
            d = 1
            y_to_use = y_diff
        else:
            y_seasonal_diff = y.diff(s).dropna()
            plt.figure(figsize=(14, 7))
            plt.plot(y_seasonal_diff.index, y_seasonal_diff, color='blue')
            plt.title(f'{label} Seasonal Difference (lag {s}) - PCE', fontsize=16)
            plt.savefig(f'{plots_dir}{label}_pce_seasonal_diff.png')
            plt.close()
            is_stationary_seasonal = adf_test(y_seasonal_diff, title=f'{label} Seasonal Difference')
            if is_stationary_seasonal:
                D = 1
                y_to_use = y_seasonal_diff
            else:
                y_both_diff = y.diff().diff(s).dropna()
                plt.figure(figsize=(14, 7))
                plt.plot(y_both_diff.index, y_both_diff, color='blue')
                plt.title(f'{label} First + Seasonal Difference - PCE', fontsize=16)
                plt.savefig(f'{plots_dir}{label}_pce_both_diff.png')
                plt.close()
                is_stationary_both = adf_test(y_both_diff, title=f'{label} First + Seasonal Difference')
                if is_stationary_both:
                    d = 1
                    D = 1
                    y_to_use = y_both_diff
                else:
                    print(f"Warning: {label} PCE series may still be non-stationary after differencing.")
                    d = 1 # Default to d=1, D=1 if still non-stationary
                    D = 1
                    y_to_use = y_both_diff
    # ACF/PACF
    plt.figure(figsize=(14, 10))
    plt.subplot(211)
    plot_acf(y_to_use, ax=plt.gca(), lags=40)
    plt.title(f'{label} ACF - PCE', fontsize=16)
    plt.subplot(212)
    plot_pacf(y_to_use, ax=plt.gca(), lags=40)
    plt.title(f'{label} PACF - PCE', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{plots_dir}{label}_pce_acf_pacf.png')
    plt.close()
    # Train/test split
    train_size = int(len(y_to_use) * 0.8)
    train, test = y_to_use[:train_size], y_to_use[train_size:]
    # Grid search
    p_values = range(0, 3) # Extended p,q search
    q_values = range(0, 3)
    P_values = range(0, 2)
    Q_values = range(0, 2)
    best_aic = float("inf")
    best_model = None
    best_order = None
    best_seasonal_order = None
    print(f"\nFitting SARIMA models for {label} PCE data...")
    for p_val in p_values: # Renamed loop variables to avoid conflict
        for q_val in q_values:
            for P_val in P_values:
                for Q_val in Q_values:
                    try:
                        model = SARIMAX(train, order=(p_val, d, q_val), seasonal_order=(P_val, D, Q_val, s), enforce_stationarity=False, enforce_invertibility=False)
                        model_fit = model.fit(disp=False)
                        aic = model_fit.aic
                        if aic < best_aic:
                            best_aic = aic
                            best_model = model_fit
                            best_order = (p_val, d, q_val)
                            best_seasonal_order = (P_val, D, Q_val, s)
                        print(f"SARIMA({p_val},{d},{q_val})x({P_val},{D},{Q_val},{s}) - AIC: {aic:.2f}")
                    except Exception as e:
                        # print(f"Error fitting SARIMA({p_val},{d},{q_val})x({P_val},{D},{Q_val},{s}): {e}")
                        continue
    if best_model is None:
        print(f"Could not find a suitable model for {label} PCE data.")
        return

    print(f"\nBest model for {label} PCE: SARIMA{best_order}x{best_seasonal_order} with AIC={best_aic:.2f}")
    with open(f'{plots_dir}{label}_pce_sarima_model_summary.txt', 'w') as f:
        f.write(str(best_model.summary()))
    # Diagnostics
    best_model.plot_diagnostics(figsize=(14, 10))
    plt.tight_layout()
    plt.savefig(f'{plots_dir}{label}_pce_sarima_diagnostics.png')
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
    plt.title(f'{label} SARIMA Forecast vs Actual - PCE', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Personal Consumption Expenditure (PCEC96)', fontsize=14) # Changed Y-axis label
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{plots_dir}{label}_pce_sarima_forecast_vs_actual.png')
    plt.close()
    # Accuracy
    mse = mean_squared_error(test, forecast_mean)
    rmse = np.sqrt(mse)
    # Check for zeros in test denominator for MAPE
    test_values_for_mape = test.values
    forecast_mean_values_for_mape = forecast_mean.values
    non_zero_mask = test_values_for_mape != 0
    if np.any(non_zero_mask): # Calculate MAPE only if there are non-zero values
        mape = np.mean(np.abs((test_values_for_mape[non_zero_mask] - forecast_mean_values_for_mape[non_zero_mask]) / test_values_for_mape[non_zero_mask])) * 100
    else:
        mape = float('nan') # Assign NaN if all test values are zero

    print(f"\nForecast Accuracy for {label} PCE:")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")
    with open(f'{plots_dir}{label}_pce_sarima_forecast_accuracy.txt', 'w') as f:
        f.write(f"MSE: {mse:.2f}\nRMSE: {rmse:.2f}\nMAPE: {mape:.2f}%\n")
    # Ljung-Box
    lb_test = acorr_ljungbox(best_model.resid, lags=[10, 20, 30], return_df=True)
    print(f"\nLjung-Box Test for {label} PCE residuals:")
    print(lb_test)
    with open(f'{plots_dir}{label}_pce_sarima_ljungbox.txt', 'w') as f:
        f.write(str(lb_test))
    print(f"\nSARIMA analysis for {label} PCE complete!\n")

# Load and aggregate data
data_path = "datasets/personalConsumptionExpenditure/RealPersonalConsumptionExpenditures.csv" # Changed data path
df = pd.read_csv(data_path)
df['observation_date'] = pd.to_datetime(df['observation_date'], format='%Y-%m-%d') # Changed date column and format
df.set_index('observation_date', inplace=True)
time_series_data = df['PCEC96'].to_frame() # Changed value column and removed groupby

# Split periods
pre_pandemic_end = '2020-02-29'
pandemic_start = '2020-03-01'
pandemic_end = '2021-06-30'
post_pandemic_start = '2021-07-01'

pre_pandemic = time_series_data.loc[:pre_pandemic_end]['PCEC96']
# pandemic_period = time_series_data.loc[pandemic_start:pandemic_end]['PCEC96'] # Not used in this script but defined for clarity
post_pandemic = time_series_data.loc[post_pandemic_start:]['PCEC96']


print(f"Pre-pandemic PCE: {pre_pandemic.index.min()} to {pre_pandemic.index.max()} ({len(pre_pandemic)} months)")
# print(f"Pandemic PCE: {pandemic_period.index.min()} to {pandemic_period.index.max()} ({len(pandemic_period)} months)")
print(f"Post-pandemic PCE: {post_pandemic.index.min()} to {post_pandemic.index.max()} ({len(post_pandemic)} months)")

# SARIMA analysis for pre-pandemic and post-pandemic PCE data
def run():
    if not pre_pandemic.empty:
        sarima_analysis(pre_pandemic, 'pre_pandemic_pce', plots_dir, s=12)
    else:
        print("Pre-pandemic PCE data is empty. Skipping analysis.")
    if not post_pandemic.empty:
        sarima_analysis(post_pandemic, 'post_pandemic_pce', plots_dir, s=12)
    else:
        print("Post-pandemic PCE data is empty. Skipping analysis.")


if __name__ == "__main__":
    run() 