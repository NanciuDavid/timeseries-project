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

# Set plotting style
plt.style.use('ggplot')
sns.set_palette("deep")
plt.rcParams['figure.figsize'] = [12, 7]
plt.rcParams['figure.dpi'] = 100

# Define plots directory
plots_dir = "plots/sarima/"
os.makedirs(plots_dir, exist_ok=True)

# Load the data
print("Loading and preparing the data...")
data_path = "datasets/airtraffic-part2/air_traffic_merged.csv"
df = pd.read_csv(data_path)
df['FLT_DATE'] = pd.to_datetime(df['FLT_DATE'], format='%d-%m-%y')
df.set_index('FLT_DATE', inplace=True)

time_series_data = df.groupby('FLT_DATE')['FLT_TOT_1'].sum().to_frame()

# Basic data inspection
print(f"Dataset shape: {time_series_data.shape}")
print("\nFirst few rows:")
print(time_series_data.head())
print("\nLast few rows:")
print(time_series_data.tail())

# Check for missing values
missing_values = time_series_data.isnull().sum()
print(f"\nMissing values: {missing_values}")

# Basic statistics
print("\nBasic statistics:")
print(time_series_data.describe())

# Save the plot of the time series
plt.figure(figsize=(14, 7))
plt.plot(time_series_data.index, time_series_data['FLT_TOT_1'], color='blue')
plt.title('Air Traffic Time Series', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Total Flights', fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig(plots_dir + 'airtraffic_time_series.png')
plt.close()

print(f"\nInitial time series plot saved to '{plots_dir}airtraffic_time_series.png'")
print("Data preparation complete.")

# Function to perform Augmented Dickey-Fuller test
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

# Check stationarity of the original series
print("\n--- Stationarity Test for Original Series ---")
y = time_series_data['FLT_TOT_1']
s = 7  # weekly seasonality
d = 0
D = 0
is_stationary = adf_test(y, title='Original Series')

y_to_use = y.copy()

# If non-stationary, apply differencing
if not is_stationary:
    print("\nApplying first differencing...")
    y_diff = y.diff().dropna()
    plt.figure(figsize=(14, 7))
    plt.plot(y_diff.index, y_diff, color='blue')
    plt.title('First Difference of Air Traffic', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Difference', fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plots_dir + 'airtraffic_first_diff.png')
    plt.close()
    print(f"\nFirst difference plot saved to '{plots_dir}airtraffic_first_diff.png'")
    is_stationary_diff = adf_test(y_diff, title='First Difference')
    if is_stationary_diff:
        d = 1
        y_to_use = y_diff
    else:
        print("\nApplying seasonal differencing...")
        y_seasonal_diff = y.diff(s).dropna()
        plt.figure(figsize=(14, 7))
        plt.plot(y_seasonal_diff.index, y_seasonal_diff, color='blue')
        plt.title('Seasonal Difference (lag 7)', fontsize=16)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Seasonal Difference', fontsize=14)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plots_dir + 'airtraffic_seasonal_diff.png')
        plt.close()
        print(f"\nSeasonal difference plot saved to '{plots_dir}airtraffic_seasonal_diff.png'")
        is_stationary_seasonal = adf_test(y_seasonal_diff, title='Seasonal Difference')
        if is_stationary_seasonal:
            D = 1
            y_to_use = y_seasonal_diff
        else:
            print("\nApplying both first and seasonal differencing...")
            y_both_diff = y.diff().diff(s).dropna()
            plt.figure(figsize=(14, 7))
            plt.plot(y_both_diff.index, y_both_diff, color='blue')
            plt.title('First + Seasonal Difference', fontsize=16)
            plt.xlabel('Date', fontsize=14)
            plt.ylabel('First + Seasonal Difference', fontsize=14)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(plots_dir + 'airtraffic_both_diff.png')
            plt.close()
            print(f"\nFirst + Seasonal difference plot saved to '{plots_dir}airtraffic_both_diff.png'")
            is_stationary_both = adf_test(y_both_diff, title='First + Seasonal Difference')
            if is_stationary_both:
                d = 1
                D = 1
                y_to_use = y_both_diff
            else:
                print("\nWarning: Series may still be non-stationary after differencing.")
                d = 1
                D = 1
                y_to_use = y_both_diff
else:
    print("\nThe original series is already stationary (d=0, D=0)")

# Plot ACF and PACF for the stationary series
plt.figure(figsize=(14, 10))
plt.subplot(211)
plot_acf(y_to_use, ax=plt.gca(), lags=40)
plt.title('Autocorrelation Function (ACF)', fontsize=16)
plt.subplot(212)
plot_pacf(y_to_use, ax=plt.gca(), lags=40)
plt.title('Partial Autocorrelation Function (PACF)', fontsize=16)
plt.tight_layout()
plt.savefig(plots_dir + 'airtraffic_acf_pacf.png')
plt.close()

print(f"\nACF and PACF plots saved to '{plots_dir}airtraffic_acf_pacf.png'")
print(f"Based on the stationarity tests, the differencing order (d) is {d}, seasonal order (D) is {D}")
print("Examine the ACF and PACF plots to determine the AR, MA, SAR, and SMA orders")

# Model Identification and Estimation
print("\n--- SARIMA Model Identification and Estimation ---")

# Split data into training and testing sets (80% training, 20% testing)
train_size = int(len(y_to_use) * 0.8)
train, test = y_to_use[:train_size], y_to_use[train_size:]

print(f"Training set size: {len(train)}")
print(f"Testing set size: {len(test)}")

# Define a range of p, q, P, Q values to try
p_values = range(0, 3)
q_values = range(0, 3)
P_values = range(0, 2)
Q_values = range(0, 2)

best_aic = float("inf")
best_model = None
best_order = None
best_seasonal_order = None

# Try different combinations of p, d, q, P, D, Q
print("\nFitting different SARIMA models...")
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
                    print(f"Error fitting SARIMA({p},{d},{q})x({P},{D},{Q},{s}): {e}")
                    continue

print(f"\nBest model by AIC: SARIMA{best_order}x{best_seasonal_order} with AIC={best_aic:.2f}")

print("\n--- Model Summary ---")
print(best_model.summary())

# Save model summary to file
with open(plots_dir + 'sarima_model_summary.txt', 'w') as f:
    f.write(str(best_model.summary()))
print(f"\nModel summary saved to '{plots_dir}sarima_model_summary.txt'")

# Model Diagnostic Checking
print("\n--- Model Diagnostic Checking ---")
residuals = best_model.resid

plt.figure(figsize=(14, 10))
plt.subplot(311)
plt.plot(residuals, color='blue')
plt.title('Residuals from SARIMA Model', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Residual Value', fontsize=14)
plt.grid(True)
plt.subplot(312)
plt.hist(residuals, bins=30, density=True, color='blue', alpha=0.7)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
plt.plot(x, np.exp(-(x**2)/2)/(np.sqrt(2*np.pi)), 'r', linewidth=2)
plt.title('Histogram of Residuals with Normal Curve', fontsize=16)
plt.xlabel('Residual Value', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.subplot(313)
from scipy import stats
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot of Residuals', fontsize=16)
plt.tight_layout()
plt.savefig(plots_dir + 'sarima_residual_diagnostics.png')
plt.close()

print(f"\nResidual diagnostics plots saved to '{plots_dir}sarima_residual_diagnostics.png'")

plt.figure(figsize=(14, 7))
plot_acf(residuals, lags=40)
plt.title('ACF of Residuals', fontsize=16)
plt.tight_layout()
plt.savefig(plots_dir + 'sarima_residual_acf.png')
plt.close()

print(f"\nResidual ACF plot saved to '{plots_dir}sarima_residual_acf.png'")

# Ljung-Box test for autocorrelation in residuals
lb_test = acorr_ljungbox(residuals, lags=[10, 20, 30], return_df=True)
print("\nLjung-Box Test for Autocorrelation in Residuals:")
print(lb_test)
with open(plots_dir + 'sarima_ljungbox.txt', 'w') as f:
    f.write(str(lb_test))

# Forecasting
print("\n--- Forecasting ---")
forecast_steps = len(test)
forecast = best_model.get_forecast(steps=forecast_steps)
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()

plt.figure(figsize=(14, 7))
plt.plot(train.index, train, label='Training Data', color='blue')
plt.plot(test.index, test, label='Test Data', color='green')
plt.plot(test.index, forecast_mean, label='Forecast', color='red')
plt.fill_between(test.index,
                 forecast_ci.iloc[:, 0],
                 forecast_ci.iloc[:, 1],
                 color='pink', alpha=0.3, label='95% Confidence Interval')
plt.title(f'SARIMA{best_order}x{best_seasonal_order} Forecast vs Actual', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Total Flights', fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(plots_dir + 'sarima_forecast_vs_actual.png')
plt.close()

print(f"\nForecast vs Actual plot saved to '{plots_dir}sarima_forecast_vs_actual.png'")

# Evaluate forecast accuracy
mse = mean_squared_error(test, forecast_mean)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((test.values - forecast_mean.values) / test.values)) * 100

print("\nForecast Accuracy Metrics:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

with open(plots_dir + 'sarima_forecast_accuracy.txt', 'w') as f:
    f.write(f"MSE: {mse:.4f}\nRMSE: {rmse:.4f}\nMAPE: {mape:.2f}%\n")

print("\nSARIMA analysis complete!") 