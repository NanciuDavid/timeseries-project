#!/usr/bin/env python3
# Multivariate Time Series Analysis for Romanian Inflation and Exchange Rates
# Application 3 - Multivariate time series models

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, grangercausalitytests, coint
from statsmodels.tsa.vector_ar.vecm import VECM
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
from sklearn.metrics import mean_squared_error
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('ggplot')
sns.set_palette("deep")
plt.rcParams['figure.figsize'] = [12, 7]
plt.rcParams['figure.dpi'] = 100

# Define plots directory
plots_dir = "plots/multivariate/"
import os
os.makedirs(plots_dir, exist_ok=True)

# Function to perform Augmented Dickey-Fuller test
def adf_test(series, title=''):
    """
    Perform Augmented Dickey-Fuller test for stationarity
    """
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

# 1. Load and prepare the data
print("Loading and preparing the data...")
# Replace with the actual paths to your data files
inflation_path = "datasets/romanian-part3/inflation_rate.csv"
exchange_path = "datasets/romanian-part3/exchange_rate.csv"

# Load inflation data (replace with actual column names)
inflation_df = pd.read_csv(inflation_path)
inflation_df['DATE'] = pd.to_datetime(inflation_df['DATE'])
inflation_df.set_index('DATE', inplace=True)

# Load exchange rate data (replace with actual column names)
exchange_df = pd.read_csv(exchange_path)
exchange_df['DATE'] = pd.to_datetime(exchange_df['DATE'])
exchange_df.set_index('DATE', inplace=True)

# Merge the datasets
data = pd.merge(inflation_df, exchange_df, left_index=True, right_index=True)
data.columns = ['inflation', 'exchange_rate']  # Rename columns for clarity

# Basic data inspection
print(f"Dataset shape: {data.shape}")
print("\nFirst few rows:")
print(data.head())
print("\nLast few rows:")
print(data.tail())

# Check for missing values
missing_values = data.isnull().sum()
print(f"\nMissing values: {missing_values}")

# Fill missing values if needed (simple forward fill as an example)
if missing_values.sum() > 0:
    data.fillna(method='ffill', inplace=True)

# Basic statistics
print("\nBasic statistics:")
print(data.describe())

# 2. Visualization
plt.figure(figsize=(15, 10))

# Plot time series
plt.subplot(211)
plt.plot(data.index, data['inflation'], label='Inflation Rate')
plt.title('Romanian Inflation Rate Over Time', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Inflation Rate (%)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

plt.subplot(212)
plt.plot(data.index, data['exchange_rate'], label='Exchange Rate', color='green')
plt.title('Romanian Exchange Rate Over Time', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Exchange Rate (RON/EUR)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

plt.tight_layout()
plt.savefig(plots_dir + 'time_series_plots.png')
plt.close()

# Scatter plot to visualize relationship
plt.figure(figsize=(10, 6))
plt.scatter(data['inflation'], data['exchange_rate'], alpha=0.5)
plt.title('Relationship Between Inflation and Exchange Rate', fontsize=14)
plt.xlabel('Inflation Rate (%)', fontsize=12)
plt.ylabel('Exchange Rate (RON/EUR)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig(plots_dir + 'scatter_relationship.png')
plt.close()

# 3. Non-stationary analysis
print("\n--- Non-stationary Analysis ---")

# Check stationarity of each series
inflation_stationary = adf_test(data['inflation'], 'Inflation Rate')
exchange_stationary = adf_test(data['exchange_rate'], 'Exchange Rate')

# Difference series if non-stationary
data_diff = pd.DataFrame()
if not inflation_stationary:
    data_diff['inflation_diff'] = data['inflation'].diff().dropna()
    plt.figure(figsize=(10, 6))
    plt.plot(data_diff.index, data_diff['inflation_diff'])
    plt.title('First Difference of Inflation Rate', fontsize=14)
    plt.savefig(plots_dir + 'inflation_diff.png')
    plt.close()
    inflation_diff_stationary = adf_test(data_diff['inflation_diff'], 'Differenced Inflation Rate')

if not exchange_stationary:
    data_diff['exchange_diff'] = data['exchange_rate'].diff().dropna()
    plt.figure(figsize=(10, 6))
    plt.plot(data_diff.index, data_diff['exchange_diff'])
    plt.title('First Difference of Exchange Rate', fontsize=14)
    plt.savefig(plots_dir + 'exchange_diff.png')
    plt.close()
    exchange_diff_stationary = adf_test(data_diff['exchange_diff'], 'Differenced Exchange Rate')

# 4. Cointegration analysis
print("\n--- Cointegration Analysis ---")
coint_result = coint(data['inflation'], data['exchange_rate'])
print(f"Cointegration Test p-value: {coint_result[1]}")
is_cointegrated = coint_result[1] < 0.05

if is_cointegrated:
    print("Series are cointegrated (evidence of a long-term relationship)")
else:
    print("Series are not cointegrated (no evidence of a long-term relationship)")

# 5. Granger causality analysis
print("\n--- Granger Causality Analysis ---")

max_lag = 12  # Can be adjusted based on the frequency of your data
    
# Test if exchange rate Granger-causes inflation
print("\nTesting if Exchange Rate Granger-causes Inflation:")
gc_exchange_to_inflation = grangercausalitytests(data[['inflation', 'exchange_rate']], max_lag, verbose=False)
for lag in range(1, max_lag + 1):
    test_result = gc_exchange_to_inflation[lag][0]['ssr_ftest']
    print(f"Lag {lag}: F-statistic = {test_result[0]:.4f}, p-value = {test_result[1]:.4f}")

# Test if inflation Granger-causes exchange rate
print("\nTesting if Inflation Granger-causes Exchange Rate:")
gc_inflation_to_exchange = grangercausalitytests(data[['exchange_rate', 'inflation']], max_lag, verbose=False)
for lag in range(1, max_lag + 1):
    test_result = gc_inflation_to_exchange[lag][0]['ssr_ftest']
    print(f"Lag {lag}: F-statistic = {test_result[0]:.4f}, p-value = {test_result[1]:.4f}")

# 6. VAR/VECM models
print("\n--- VAR/VECM Model Estimation ---")

# Prepare data for VAR/VECM
# Split into training and testing sets
train_size = int(len(data) * 0.8)
train_data = data.iloc[:train_size]
test_data = data.iloc[train_size:]

if is_cointegrated:
    # If series are cointegrated, use VECM
    print("Estimating VECM model...")
    
    # Determine optimal lag order for VECM
    model_selection = VAR(train_data).select_order(maxlags=10)
    best_lag = max(1, model_selection.bic)
    print(f"Optimal lag order for VECM: {best_lag}")
    
    # Fit VECM model
    vecm_model = VECM(train_data, k_ar_diff=best_lag, deterministic="ci")
    vecm_results = vecm_model.fit()
    print(vecm_results.summary())
    
    # Save VECM summary
    with open(plots_dir + 'vecm_model_summary.txt', 'w') as f:
        f.write(str(vecm_results.summary()))
    
    # Make forecasts with VECM
    vecm_forecast = vecm_results.predict(steps=len(test_data))
    if vecm_forecast.size == 0:
        print("Warning: VECM forecast is empty. Using simple persistence forecast instead.")
        # Create a fallback forecast (repeat the last training values)
        vecm_forecast = np.tile(train_data.iloc[-1].values, (len(test_data), 1))
    
    # Plot forecasts
    plt.figure(figsize=(15, 12))
    
    plt.subplot(211)
    plt.plot(train_data.index, train_data['inflation'], label='Train (Inflation)')
    plt.plot(test_data.index, test_data['inflation'], label='Test (Inflation)')
    plt.plot(test_data.index, vecm_forecast[:, 0], label='VECM Forecast (Inflation)', color='red')
    plt.title('VECM Forecast: Inflation Rate', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Inflation Rate (%)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(212)
    plt.plot(train_data.index, train_data['exchange_rate'], label='Train (Exchange Rate)')
    plt.plot(test_data.index, test_data['exchange_rate'], label='Test (Exchange Rate)')
    plt.plot(test_data.index, vecm_forecast[:, 1], label='VECM Forecast (Exchange Rate)', color='red')
    plt.title('VECM Forecast: Exchange Rate', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Exchange Rate (RON/EUR)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(plots_dir + 'vecm_forecast.png')
    plt.close()
    
    # 7. Impulse Response Function for VECM
    print("\n--- Impulse Response Function (VECM) ---")
    irf_periods = 20
    vecm_irf = vecm_results.irf(irf_periods)
    
    plt.figure(figsize=(15, 10))
    plt.subplot(221)
    plt.plot(vecm_irf.irf[:, 0, 0])
    plt.title('Response of Inflation to Inflation Shock', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(222)
    plt.plot(vecm_irf.irf[:, 0, 1])
    plt.title('Response of Inflation to Exchange Rate Shock', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(223)
    plt.plot(vecm_irf.irf[:, 1, 0])
    plt.title('Response of Exchange Rate to Inflation Shock', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(224)
    plt.plot(vecm_irf.irf[:, 1, 1])
    plt.title('Response of Exchange Rate to Exchange Rate Shock', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(plots_dir + 'vecm_irf.png')
    plt.close()

else:
    # If series are not cointegrated, use VAR on the differenced data
    print("Estimating VAR model...")
    
    # Use the differenced data for VAR
    train_diff = data_diff.iloc[:train_size-1]  # Adjust for lost observation in differencing
    test_diff = data_diff.iloc[train_size-1:]
    
    # Determine optimal lag order for VAR
    model_selection = VAR(train_diff).select_order(maxlags=10)
    best_lag = max(1, model_selection.bic)
    print(f"Optimal lag order for VAR: {best_lag}")
    
    # Fit VAR model
    var_model = VAR(train_diff)
    var_results = var_model.fit(best_lag)
    print(var_results.summary())
    
    # Save VAR summary
    with open(plots_dir + 'var_model_summary.txt', 'w') as f:
        f.write(str(var_results.summary()))
    
    # Make forecasts with VAR
    var_forecast = var_results.forecast(train_diff.values, steps=len(test_diff))
    if var_forecast.size == 0:
        print("Warning: VAR forecast is empty. Using zeros as fallback.")
        var_forecast = np.zeros((len(test_diff), train_diff.shape[1]))
    var_forecast_df = pd.DataFrame(var_forecast, index=test_diff.index, columns=test_diff.columns)
    
    # Plot forecasts
    plt.figure(figsize=(15, 12))
    
    plt.subplot(211)
    plt.plot(train_diff.index, train_diff['inflation_diff'], label='Train (Differenced Inflation)')
    plt.plot(test_diff.index, test_diff['inflation_diff'], label='Test (Differenced Inflation)')
    plt.plot(test_diff.index, var_forecast_df['inflation_diff'], label='VAR Forecast (Inflation)', color='red')
    plt.title('VAR Forecast: Differenced Inflation Rate', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Differenced Inflation Rate', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(212)
    plt.plot(train_diff.index, train_diff['exchange_diff'], label='Train (Differenced Exchange Rate)')
    plt.plot(test_diff.index, test_diff['exchange_diff'], label='Test (Differenced Exchange Rate)')
    plt.plot(test_diff.index, var_forecast_df['exchange_diff'], label='VAR Forecast (Exchange Rate)', color='red')
    plt.title('VAR Forecast: Differenced Exchange Rate', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Differenced Exchange Rate', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(plots_dir + 'var_forecast.png')
    plt.close()
    
    # 7. Impulse Response Function for VAR
    print("\n--- Impulse Response Function (VAR) ---")
    irf_periods = 20
    var_irf = var_results.irf(irf_periods)
    
    plt.figure(figsize=(15, 10))
    plt.subplot(221)
    plt.plot(var_irf.irfs[:, 0, 0])
    plt.title('Response of Inflation to Inflation Shock', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(222)
    plt.plot(var_irf.irfs[:, 0, 1])
    plt.title('Response of Inflation to Exchange Rate Shock', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(223)
    plt.plot(var_irf.irfs[:, 1, 0])
    plt.title('Response of Exchange Rate to Inflation Shock', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(224)
    plt.plot(var_irf.irfs[:, 1, 1])
    plt.title('Response of Exchange Rate to Exchange Rate Shock', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(plots_dir + 'var_irf.png')
    plt.close()

# 8. Forecast Evaluation
print("\n--- Forecast Evaluation ---")

if is_cointegrated:
    # Calculate RMSE for VECM model
    inflation_rmse = np.sqrt(mean_squared_error(test_data['inflation'], vecm_forecast[:, 0]))
    exchange_rmse = np.sqrt(mean_squared_error(test_data['exchange_rate'], vecm_forecast[:, 1]))
    
    print(f"VECM Inflation RMSE: {inflation_rmse:.4f}")
    print(f"VECM Exchange Rate RMSE: {exchange_rmse:.4f}")
else:
    # Calculate RMSE for VAR model - on differenced data
    inflation_diff_rmse = np.sqrt(mean_squared_error(test_diff['inflation_diff'], var_forecast_df['inflation_diff']))
    exchange_diff_rmse = np.sqrt(mean_squared_error(test_diff['exchange_diff'], var_forecast_df['exchange_diff']))
    
    print(f"VAR Differenced Inflation RMSE: {inflation_diff_rmse:.4f}")
    print(f"VAR Differenced Exchange Rate RMSE: {exchange_diff_rmse:.4f}")

print("\nMultivariate Time Series Analysis completed successfully.")
