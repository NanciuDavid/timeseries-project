#!/usr/bin/env python3
# ARIMA Analysis for US GDP Time Series
# Application 1 - ARMA/ARIMA models using Box-Jenkins methodology

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings
warnings.filterwarnings('ignore')
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.diagnostic import het_arch
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# Set plotting style
plt.style.use('ggplot')  # Using a built-in style instead of 'seaborn'
sns.set_palette("deep")
plt.rcParams['figure.figsize'] = [12, 7]
plt.rcParams['figure.dpi'] = 100

# Define plots directory
plots_dir = "plots/arima/gdp/"
import os
os.makedirs(plots_dir, exist_ok=True)

# Load the data
print("Loading and preparing the data...")
data_path = "datasets/USGDP/GDP.csv"  # Path relative to project root
df = pd.read_csv(data_path, delimiter=';', skiprows=1)

# Convert the observation_date column to datetime
df['observation_date'] = pd.to_datetime(df['observation_date'])

# Set observation_date as the index
df.set_index('observation_date', inplace=True)

# Basic data inspection
print(f"Dataset shape: {df.shape}")
print("\nFirst few rows:")
print(df.head())
print("\nLast few rows:")
print(df.tail())

# Check for missing values
missing_values = df.isnull().sum()
print(f"\nMissing values: {missing_values}")

# Basic statistics
print("\nBasic statistics:")
print(df.describe())

# Create a figure for multiple plots - similar to Figure 1 in the example PDF
plt.figure(figsize=(15, 12))

# Original time series
plt.subplot(311)
plt.plot(df.index, df['GDP'], color='blue', linewidth=1.5)
plt.title('(a) US GDP Time Series', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('GDP (Billions USD)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# Log transformation (for comparison, even if not necessary)
plt.subplot(312)
plt.plot(df.index, np.log(df['GDP']), color='green', linewidth=1.5)
plt.title('(b) Log Transformation of US GDP', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Log(GDP)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# First difference
df['first_diff'] = df['GDP'].diff()
plt.subplot(313)
plt.plot(df.index[1:], df['first_diff'][1:], color='red', linewidth=1)
plt.title('(c) First Difference of US GDP', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('GDP Difference', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(plots_dir + 'figure1_transformations.png')
plt.close()

print(f"\nTransformations plot saved to '{plots_dir}figure1_transformations.png'")

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

# Check stationarity of the original series
print("\n--- Stationarity Test for Original Series ---")
is_stationary = adf_test(df['GDP'], title='Original Series')

# Create a table with stationarity test results - similar to example PDF
test_results = pd.DataFrame({
    'Test': ['ADF Test', 'Phillips-Perron', 'KPSS Test'],
    'p-value': [f"{adfuller(df['GDP'].dropna())[1]:.4f}", 'N/A', 'N/A'],
    'Conclusion': ['Non-stationary' if not is_stationary else 'Stationary', 'N/A', 'N/A']
})

# If non-stationary, apply differencing
if not is_stationary:
    print("\nApplying differencing to achieve stationarity...")

    # First difference was already calculated above
    df_diff = df['first_diff'].dropna()

    # Check stationarity of the first difference
    print("\n--- Stationarity Test for First Difference ---")
    is_stationary_diff = adf_test(df_diff, title='First Difference')

    # Update test results table for differenced series
    test_results_diff = pd.DataFrame({
        'Test': ['ADF Test', 'Phillips-Perron', 'KPSS Test'],
        'p-value': [f"{adfuller(df_diff.dropna())[1]:.4f}", 'N/A', 'N/A'],
        'Conclusion': ['Stationary' if is_stationary_diff else 'Non-stationary', 'N/A', 'N/A']
    })

    # If still non-stationary, apply second differencing
    if not is_stationary_diff:
        print("\nApplying second differencing...")

        # Second difference
        df['second_diff'] = df['first_diff'].diff()

        # Remove NaN values
        df_diff2 = df['second_diff'].dropna()

        # Check stationarity of the second difference
        print("\n--- Stationarity Test for Second Difference ---")
        is_stationary_diff2 = adf_test(df_diff2, title='Second Difference')

        if is_stationary_diff2:
            stationary_series = df_diff2
            d_order = 2
            print("\nThe series is stationary after second differencing (d=2)")
        else:
            print("\nThe series remains non-stationary after second differencing.")
            print("Consider other transformations like log transformation.")
            stationary_series = df_diff2
            d_order = 2
    else:
        stationary_series = df_diff
        d_order = 1
        print("\nThe series is stationary after first differencing (d=1)")
else:
    stationary_series = df['GDP']
    d_order = 0
    print("\nThe original series is already stationary (d=0)")

# Create a visual table for unit root test results - similar to the example PDF
fig, ax = plt.figure(figsize=(9, 4)), plt.axes(frameon=False)
ax.set_title('Unit Root Test Results', fontsize=14, pad=20)
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)

# Create the table with test results
if d_order == 1:
    table_data = [
        ['Test', 'Original Series', 'First Difference'],
        ['ADF p-value', f"{adfuller(df['GDP'].dropna())[1]:.4f}", f"{adfuller(df_diff.dropna())[1]:.8f}"],
        ['Conclusion', 'Non-stationary', 'Stationary']
    ]
elif d_order == 2:
    table_data = [
        ['Test', 'Original Series', 'First Difference', 'Second Difference'],
        ['ADF p-value', f"{adfuller(df['GDP'].dropna())[1]:.4f}", f"{adfuller(df_diff.dropna())[1]:.4f}", f"{adfuller(df_diff2.dropna())[1]:.8f}"],
        ['Conclusion', 'Non-stationary', 'Non-stationary', 'Stationary']
    ]
else:
    table_data = [
        ['Test', 'Original Series'],
        ['ADF p-value', f"{adfuller(df['GDP'].dropna())[1]:.4f}"],
        ['Conclusion', 'Stationary']
    ]

# Create the table
table = ax.table(cellText=table_data, loc='center', cellLoc='center',
                 colColours=['#f2f2f2']*len(table_data[0]))
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.5)

plt.savefig(plots_dir + 'unit_root_test_table.png', bbox_inches='tight')
plt.close()

print(f"\nUnit root test table saved to '{plots_dir}unit_root_test_table.png'")

# Plot ACF and PACF for the stationary series - Similar to Figure 2 in example PDF
plt.figure(figsize=(15, 10))

plt.subplot(211)
plot_acf(stationary_series, ax=plt.gca(), lags=40, alpha=0.05)
plt.title('(a) Autocorrelation Function (ACF)', fontsize=14)
plt.xlabel('Lag', fontsize=12)
plt.ylabel('Correlation', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

plt.subplot(212)
plot_pacf(stationary_series, ax=plt.gca(), lags=40, alpha=0.05)
plt.title('(b) Partial Autocorrelation Function (PACF)', fontsize=14)
plt.xlabel('Lag', fontsize=12)
plt.ylabel('Partial Correlation', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(plots_dir + 'figure2_acf_pacf.png')
plt.close()

print(f"\nACF and PACF plots saved to '{plots_dir}figure2_acf_pacf.png'")
print(f"Based on the stationarity tests, the differencing order (d) is {d_order}")
print("Examine the ACF and PACF plots to determine the AR order (p) and MA order (q)")

# Model Identification and Estimation
print("\n--- ARIMA Model Identification and Estimation ---")

# Split data into training and testing sets (80% training, 20% testing)
train_size = int(len(stationary_series) * 0.8)
train, test = stationary_series[:train_size], stationary_series[train_size:]

print(f"Training set size: {len(train)}")
print(f"Testing set size: {len(test)}")

# Define a range of p and q values to try
p_values = range(0, 4)
q_values = range(0, 4)

# Store results
best_aic = float("inf")
best_bic = float("inf")
best_model_aic = None
best_model_bic = None
best_order_aic = None
best_order_bic = None

# Create a DataFrame to store model comparison results
model_results = pd.DataFrame(columns=['p', 'd', 'q', 'AIC', 'BIC'])

# Try different combinations of p and q
print("\nFitting different ARIMA models...")
for p in p_values:
    for q in q_values:
        # Skip if both p and q are 0
        if p == 0 and q == 0:
            continue

        try:
            # Fit ARIMA model
            model = ARIMA(train, order=(p, d_order, q))
            model_fit = model.fit()

            # Get AIC and BIC
            aic = model_fit.aic
            bic = model_fit.bic

            # Add to results DataFrame
            model_results = pd.concat([model_results, pd.DataFrame({
                'p': [p], 'd': [d_order], 'q': [q], 'AIC': [aic], 'BIC': [bic]
            })], ignore_index=True)

            # Check if this model is better than previous ones
            if aic < best_aic:
                best_aic = aic
                best_model_aic = model_fit
                best_order_aic = (p, d_order, q)

            if bic < best_bic:
                best_bic = bic
                best_model_bic = model_fit
                best_order_bic = (p, d_order, q)

            print(f"ARIMA({p},{d_order},{q}) - AIC: {aic:.2f}, BIC: {bic:.2f}")

        except Exception as e:
            print(f"Error fitting ARIMA({p},{d_order},{q}): {e}")

# Sort models by AIC
model_results = model_results.sort_values('AIC')
print("\nModel comparison by AIC:")
print(model_results.head(10))

# Print best model information
print(f"\nBest model by AIC: ARIMA{best_order_aic} with AIC={best_aic:.2f}")
print(f"Best model by BIC: ARIMA{best_order_bic} with BIC={best_bic:.2f}")

# We'll fit both best models based on AIC and BIC for comparison
aic_model = ARIMA(train, order=best_order_aic).fit()

# Also fit a simpler ARIMA(0,1,1) model for comparison
simple_order = (0, 1, 1)
simple_model = ARIMA(train, order=simple_order).fit()
print(f"\nSimple exponential smoothing equivalent - ARIMA{simple_order}: AIC={simple_model.aic:.2f}, BIC={simple_model.bic:.2f}")

# If best_order_aic and best_order_bic are different, use both. Otherwise, compare with the simple model
if best_order_aic == best_order_bic:
    # Set BIC model to simple model for comparison
    bic_model = simple_model
    best_order_bic = simple_order
    print(f"Since AIC and BIC selected the same model, we'll compare with ARIMA{simple_order}")
else:
    bic_model = ARIMA(train, order=best_order_bic).fit()

# ------------ Creating comparison plots for the two models --------------

# Function to inverse difference the predictions
def inverse_difference(history, yhat, intervals=1):
    """Inverse difference a forecasted value"""
    return yhat + history[-intervals]

# 1. Residual Analysis - Figure 3 - Residual comparison
plt.figure(figsize=(15, 10))

# Residuals from AIC model
aic_residuals = aic_model.resid
plt.subplot(211)
plt.plot(aic_residuals, color='blue')
plt.title(f'(a) Residuals Time Plot - ARIMA{best_order_aic} (Complex)')
plt.xlabel('Observation')
plt.ylabel('Residual')
plt.grid(True, linestyle='--', alpha=0.7)

# Residuals from BIC model
bic_residuals = bic_model.resid
plt.subplot(212)
plt.plot(bic_residuals, color='green')
plt.title(f'(b) Residuals Time Plot - ARIMA{best_order_bic} (Simple)')
plt.xlabel('Observation')
plt.ylabel('Residual')
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(plots_dir + 'figure3_residual_comparison.png')
plt.close()

print(f"\nResidual time plots saved to '{plots_dir}figure3_residual_comparison.png'")

# 2. ACF of Residuals - Figure 4
plt.figure(figsize=(15, 10))

# ACF of residuals for AIC model
plt.subplot(211)
plot_acf(aic_residuals, lags=40, alpha=0.05, ax=plt.gca())
plt.title(f'(a) ACF of Residuals - ARIMA{best_order_aic} (Complex)')
plt.grid(True, linestyle='--', alpha=0.7)

# ACF of residuals for BIC model
plt.subplot(212)
plot_acf(bic_residuals, lags=40, alpha=0.05, ax=plt.gca())
plt.title(f'(b) ACF of Residuals - ARIMA{best_order_bic} (Simple)')
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(plots_dir + 'figure4_acf_residuals_comparison.png')
plt.close()

print(f"\nACF of residuals plots saved to '{plots_dir}figure4_acf_residuals_comparison.png'")

# 3. Histogram of Residuals - Figure 5
plt.figure(figsize=(15, 10))

# Residuals Histogram for AIC model
plt.subplot(211)
plt.hist(aic_residuals, bins=30, color='blue', alpha=0.7)
plt.title(f'(a) Residuals Histogram - ARIMA{best_order_aic} (Complex)')
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.grid(True, linestyle='--', alpha=0.7)

# Residuals Histogram for BIC model
plt.subplot(212)
plt.hist(bic_residuals, bins=30, color='green', alpha=0.7)
plt.title(f'(b) Residuals Histogram - ARIMA{best_order_bic} (Simple)')
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(plots_dir + 'figure5_histogram_residuals_comparison.png')
plt.close()

print(f"\nResidual histograms saved to '{plots_dir}figure5_histogram_residuals_comparison.png'")

# 4. QQ Plot of Residuals - Figure 6
plt.figure(figsize=(15, 10))

# QQ Plot for AIC model
plt.subplot(211)
stats.probplot(aic_residuals, plot=plt)
plt.title(f'(a) Q-Q Plot - ARIMA{best_order_aic} (Complex)')
plt.grid(True, linestyle='--', alpha=0.7)

# QQ Plot for BIC model
plt.subplot(212)
stats.probplot(bic_residuals, plot=plt)
plt.title(f'(b) Q-Q Plot - ARIMA{best_order_bic} (Simple)')
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(plots_dir + 'figure6_qq_residuals_comparison.png')
plt.close()

print(f"\nQ-Q plots saved to '{plots_dir}figure6_qq_residuals_comparison.png'")

# 5. Model Fit Comparison (rename from figure5 to figure7 to match gold dataset)
plt.figure(figsize=(15, 10))

# Get the history data (training period) in original scale
history = df['GDP'][:train_size+d_order].values

# Predict for the training period using AIC model
aic_train_predictions = []
for i in range(len(train)):
    # Get prediction for AIC model
    yhat = aic_model.predict(start=i, end=i)[0]

    # If we've differenced, we need to invert the transformation
    if d_order == 1:
        yhat = inverse_difference(history[:train_size+1], yhat)
    elif d_order == 2:
        yhat = inverse_difference(history[:train_size+1], yhat)
        yhat = inverse_difference(history[:train_size], yhat)

    aic_train_predictions.append(yhat)

# Predict for the training period using BIC model
bic_train_predictions = []
for i in range(len(train)):
    # Get prediction for BIC model
    yhat = bic_model.predict(start=i, end=i)[0]

    # If we've differenced, we need to invert the transformation
    if d_order == 1:
        yhat = inverse_difference(history[:train_size+1], yhat)
    elif d_order == 2:
        yhat = inverse_difference(history[:train_size+1], yhat)
        yhat = inverse_difference(history[:train_size], yhat)

    bic_train_predictions.append(yhat)

# Plot original vs. predicted
plt.subplot(211)
plt.plot(df.index[:train_size], df['GDP'][:train_size], label='Actual GDP', color='blue')
plt.plot(df.index[:train_size], aic_train_predictions, label=f'ARIMA{best_order_aic} (Complex)', color='red', alpha=0.7)
plt.plot(df.index[:train_size], bic_train_predictions, label=f'ARIMA{best_order_bic} (Simple)', color='green', alpha=0.7)
plt.title('(a) Model Fit Comparison - Training Period', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('GDP', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Zoom in to a section for better visibility
mid_point = len(train) // 2
window_size = len(train) // 4
plt.subplot(212)
plt.plot(df.index[mid_point:mid_point+window_size],
         df['GDP'][mid_point:mid_point+window_size],
         label='Actual GDP', color='blue')
plt.plot(df.index[mid_point:mid_point+window_size],
         aic_train_predictions[mid_point:mid_point+window_size],
         label=f'ARIMA{best_order_aic} (Complex)', color='red', alpha=0.7)
plt.plot(df.index[mid_point:mid_point+window_size],
         bic_train_predictions[mid_point:mid_point+window_size],
         label=f'ARIMA{best_order_bic} (Simple)', color='green', alpha=0.7)
plt.title('(b) Model Fit Comparison - Zoomed In', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('GDP', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(plots_dir + 'figure7_fitted_actual_comparison.png')
plt.close()

print(f"\nModel fit comparison plot saved to '{plots_dir}figure7_fitted_actual_comparison.png'")

# 6. Out-of-sample Forecasting Comparison - Split into figure8 and figure9
plt.figure(figsize=(15, 10))

# Get the history and test data
history = df['GDP'][:train_size+d_order].values
actual_test = df['GDP'][train_size:].values

# Forecast using AIC model
aic_predictions = []
bic_predictions = []

for i in range(len(test)):
    # Forecast using AIC model
    aic_yhat = aic_model.predict(start=len(train)+i, end=len(train)+i)[0]

    # Forecast using BIC model
    bic_yhat = bic_model.predict(start=len(train)+i, end=len(train)+i)[0]

    # If we've differenced, we need to invert the transformation
    if d_order == 1:
        # AIC model
        aic_yhat = inverse_difference(df['GDP'][train_size+i-1:train_size+i].values, aic_yhat)
        # BIC model
        bic_yhat = inverse_difference(df['GDP'][train_size+i-1:train_size+i].values, bic_yhat)
    elif d_order == 2:
        # Need to handle double differencing
        # This is simplified and may need refinement
        aic_yhat = aic_yhat + 2*df['GDP'][train_size+i-1] - df['GDP'][train_size+i-2]
        bic_yhat = bic_yhat + 2*df['GDP'][train_size+i-1] - df['GDP'][train_size+i-2]

    aic_predictions.append(aic_yhat)
    bic_predictions.append(bic_yhat)

# Plot the forecasts - ensure arrays have the same dimensions
plt.plot(df.index[train_size:train_size+len(aic_predictions)], actual_test[:len(aic_predictions)], label='Actual GDP', color='blue')
plt.plot(df.index[train_size:train_size+len(aic_predictions)], aic_predictions, label=f'ARIMA{best_order_aic} (Complex)', color='red', linestyle='--', marker='o')
plt.plot(df.index[train_size:train_size+len(aic_predictions)], bic_predictions, label=f'ARIMA{best_order_bic} (Simple)', color='green', linestyle='--', marker='s')
plt.title('Out-of-Sample Forecast Comparison', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('GDP', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(plots_dir + 'figure8_forecast_comparison.png')
plt.close()

print(f"\nOut-of-sample forecast comparison plot saved to '{plots_dir}figure8_forecast_comparison.png'")

# 7. Forecast Errors Comparison
plt.figure(figsize=(15, 10))

# Calculate forecast errors - ensure consistent lengths
aic_errors = actual_test[:len(aic_predictions)] - aic_predictions
bic_errors = actual_test[:len(bic_predictions)] - bic_predictions

# Plot the forecast errors
plt.plot(df.index[train_size:train_size+len(aic_predictions)], aic_errors, label=f'ARIMA{best_order_aic} (Complex) Errors', color='red', marker='o')
plt.plot(df.index[train_size:train_size+len(aic_predictions)], bic_errors, label=f'ARIMA{best_order_bic} (Simple) Errors', color='green', marker='s')
plt.axhline(y=0, color='blue', linestyle='-')
plt.title('Forecast Error Comparison', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Error', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(plots_dir + 'figure9_forecast_error_comparison.png')
plt.close()

print(f"\nForecast error comparison plot saved to '{plots_dir}figure9_forecast_error_comparison.png'")

# 8. Forecast Accuracy Metrics - Figure 10
# Calculate metrics
aic_rmse = np.sqrt(mean_squared_error(actual_test[:len(aic_predictions)], aic_predictions))
bic_rmse = np.sqrt(mean_squared_error(actual_test[:len(bic_predictions)], bic_predictions))

aic_mape = np.mean(np.abs((actual_test[:len(aic_predictions)] - aic_predictions) / actual_test[:len(aic_predictions)])) * 100
bic_mape = np.mean(np.abs((actual_test[:len(bic_predictions)] - bic_predictions) / actual_test[:len(bic_predictions)])) * 100

aic_mae = np.mean(np.abs(actual_test[:len(aic_predictions)] - aic_predictions))
bic_mae = np.mean(np.abs(actual_test[:len(bic_predictions)] - bic_predictions))

# Calculate Theil's U statistic (forecast accuracy relative to naive forecast)
# For naive forecast, use previous values (no-change forecast)
naive_forecast = df['GDP'][train_size-1:train_size+len(aic_predictions)-1].values
aic_theil_u = np.sqrt(np.sum((aic_predictions - actual_test[:len(aic_predictions)])**2)) / np.sqrt(np.sum((naive_forecast - actual_test[:len(aic_predictions)])**2))
bic_theil_u = np.sqrt(np.sum((bic_predictions - actual_test[:len(bic_predictions)])**2)) / np.sqrt(np.sum((naive_forecast - actual_test[:len(bic_predictions)])**2))

# Create a comparison table
plt.figure(figsize=(10, 6))
plt.text(0.5, 0.9, "Forecast Accuracy Comparison", fontsize=16, ha='center', transform=plt.gca().transAxes)

metrics_data = [
    ['Metric', f'ARIMA{best_order_aic} (Complex)', f'ARIMA{best_order_bic} (Simple)'],
    ['RMSE', f"{aic_rmse:.2f}", f"{bic_rmse:.2f}"],
    ['MAPE (%)', f"{aic_mape:.2f}", f"{bic_mape:.2f}"],
    ['MAE', f"{aic_mae:.2f}", f"{bic_mae:.2f}"],
    ["Theil's U", f"{aic_theil_u:.4f}", f"{bic_theil_u:.4f}"]
]

metrics_table = plt.table(cellText=metrics_data[1:], colLabels=metrics_data[0],
                          loc='center', cellLoc='center')
metrics_table.auto_set_font_size(False)
metrics_table.set_fontsize(12)
metrics_table.scale(1.2, 1.5)

# Highlight the best model for each metric
for i in range(4):  # 4 metrics now including Theil's U
    if i == 3:  # For Theil's U, lower is better
        if float(metrics_data[i+1][1].split()[0]) < float(metrics_data[i+1][2].split()[0]):
            metrics_table[(i, 1)].set_facecolor('#CCFFCC')  # Light green for Complex
        else:
            metrics_table[(i, 2)].set_facecolor('#CCFFCC')  # Light green for Simple
    else:  # For other metrics, lower is also better
        if float(metrics_data[i+1][1].split()[0]) < float(metrics_data[i+1][2].split()[0]):
            metrics_table[(i, 1)].set_facecolor('#CCFFCC')  # Light green for Complex
        else:
            metrics_table[(i, 2)].set_facecolor('#CCFFCC')  # Light green for Simple

plt.axis('off')
plt.tight_layout()
plt.savefig(plots_dir + 'figure10_forecast_accuracy.png')
plt.close()

print(f"\nForecast accuracy metrics saved to '{plots_dir}figure10_forecast_accuracy.png'")

# Summary of findings
print("\n--- ARIMA Analysis Summary ---")
print(f"Complex model: ARIMA{best_order_aic}")
print(f"Simple model: ARIMA{best_order_bic}")
print("\nForecast Accuracy:")
print(f"ARIMA{best_order_aic} - RMSE: {aic_rmse:.2f}, MAPE: {aic_mape:.2f}%, Theil's U: {aic_theil_u:.4f}")
print(f"ARIMA{best_order_bic} - RMSE: {bic_rmse:.2f}, MAPE: {bic_mape:.2f}%, Theil's U: {bic_theil_u:.4f}")

# Create a summary markdown file
model_comparison = "complex vs simple" if best_order_aic != best_order_bic else "best AIC/BIC vs simple exponential smoothing"

summary_content = f"""# ARIMA Analysis of US GDP Data

## Data Characteristics
- Time Series: Quarterly US GDP
- Time Period: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}
- Number of Observations: {len(df)}

## Stationarity Analysis
- Original Series: {'Stationary' if is_stationary else 'Non-stationary'}
- Differencing Required: d = {d_order}

## Model Selection
- Complex model: ARIMA{best_order_aic} with AIC = {best_aic:.2f}
- Simple model: ARIMA{best_order_bic} with BIC = {simple_model.bic if best_order_bic == simple_order else best_bic:.2f}
- Model comparison: {model_comparison}

## Forecast Accuracy
- ARIMA{best_order_aic}:
  - RMSE: {aic_rmse:.2f}
  - MAPE: {aic_mape:.2f}%
  - MAE: {aic_mae:.2f}
  - Theil's U: {aic_theil_u:.4f}

- ARIMA{best_order_bic}:
  - RMSE: {bic_rmse:.2f}
  - MAPE: {bic_mape:.2f}%
  - MAE: {bic_mae:.2f}
  - Theil's U: {bic_theil_u:.4f}

## Conclusion
The {'complex' if aic_rmse < bic_rmse else 'simple'} model performed better in terms of forecast accuracy.
A Theil's U value less than 1 indicates that the model performs better than a naive forecast.
"""

with open(plots_dir + 'USGDP_ARIMA_SUMMARY.md', 'w') as f:
    f.write(summary_content)

print(f"\nAnalysis summary saved to '{plots_dir}USGDP_ARIMA_SUMMARY.md'")
print("\nARIMA analysis of US GDP data completed successfully!")

# After finding the best models - add model visualization plots

# Create a visual representation of the model equation - similar to example PDF
plt.figure(figsize=(10, 6))
plt.text(0.5, 0.9, r"ARIMA Model Equation Visualization", fontsize=16, ha='center')

# Complex model
plt.text(0.5, 0.75, 
         r"$\mathbf{ARIMA(%d,%d,%d)}$ model (Complex)" % (best_order_aic[0], best_order_aic[1], best_order_aic[2]),
         fontsize=14, ha='center')

# Extract coefficients for complex model
ar_params = aic_model.arparams if hasattr(aic_model, 'arparams') else aic_model.params[1:best_order_aic[0]+1]
ma_params = aic_model.maparams if hasattr(aic_model, 'maparams') else aic_model.params[-(best_order_aic[2]):]

# Create the model equation text for complex model using proper LaTeX
ar_terms = []
for i, param in enumerate(ar_params, 1):
    sign = "-" if param > 0 else "+"  # Note: In ARIMA notation, positive coefficient means negative in equation
    ar_terms.append(f"{sign} {abs(param):.4f} B^{{{i}}}")

ma_terms = []
for i, param in enumerate(ma_params, 1):
    sign = "+" if param > 0 else "-"  # For MA terms, positive coefficient means positive in equation
    ma_terms.append(f"{sign} {abs(param):.4f} B^{{{i}}}")

# Full equation for complex model in LaTeX
ar_eq = r"$\left(1 " + " ".join(ar_terms) + r"\right)$"
ma_eq = r"$\left(1 " + " ".join(ma_terms) + r"\right)$"
full_eq = r"$" + ar_eq[1:-1] + r" \nabla^{" + str(best_order_aic[1]) + r"} GDP_t = " + ma_eq[1:-1] + r" \varepsilon_t$"

plt.text(0.5, 0.6, full_eq, fontsize=13, ha='center')

# Simple model
plt.text(0.5, 0.4, 
         r"$\mathbf{ARIMA(%d,%d,%d)}$ model (Simple)" % (best_order_bic[0], best_order_bic[1], best_order_bic[2]),
         fontsize=14, ha='center')

# Extract coefficients for simple model
ar_params_simple = bic_model.arparams if hasattr(bic_model, 'arparams') else []
ma_params_simple = bic_model.maparams if hasattr(bic_model, 'maparams') else bic_model.params[-1:]

# Create the model equation text for simple model using proper LaTeX
ar_terms_simple = []
for i, param in enumerate(ar_params_simple, 1):
    sign = "-" if param > 0 else "+"
    ar_terms_simple.append(f"{sign} {abs(param):.4f} B^{{{i}}}")

ma_terms_simple = []
for i, param in enumerate(ma_params_simple, 1):
    sign = "+" if param > 0 else "-"
    ma_terms_simple.append(f"{sign} {abs(param):.4f} B^{{{i}}}")

# Full equation for simple model in LaTeX
ar_eq_simple = r"$\left(1 " + " ".join(ar_terms_simple) + r"\right)$" if ar_terms_simple else r"$\left(1\right)$"
ma_eq_simple = r"$\left(1 " + " ".join(ma_terms_simple) + r"\right)$" if ma_terms_simple else r"$\left(1\right)$"
full_eq_simple = r"$" + ar_eq_simple[1:-1] + r" \nabla^{" + str(best_order_bic[1]) + r"} GDP_t = " + ma_eq_simple[1:-1] + r" \varepsilon_t$"

plt.text(0.5, 0.25, full_eq_simple, fontsize=13, ha='center')

# Legend with proper LaTeX
plt.text(0.5, 0.1, r"Where:", fontsize=12, ha='center')
plt.text(0.5, 0.05, r"$B$ is the backshift operator, $\nabla$ is the differencing operator, and $\varepsilon_t$ is white noise", fontsize=10, ha='center')

# Remove axes
plt.axis('off')
plt.tight_layout()
plt.savefig(plots_dir + 'model_equation.png', dpi=150)
plt.close()

print(f"\nModel equation visualization saved to '{plots_dir}model_equation.png'")

# Create a visual table of model comparisons - similar to example PDF
plt.figure(figsize=(10, 6))
plt.text(0.5, 0.95, "ARIMA Model Comparison", fontsize=16, ha='center')

# Get top 5 models by AIC
top_models = model_results.sort_values('AIC').head(5)

# Format comparison data as a styled table - similar to example PDF
table_data = []
headers = ['Model', 'AIC', 'BIC', 'Log-Likelihood']
for _, row in top_models.iterrows():
    p, d, q = int(row['p']), int(row['d']), int(row['q'])
    try:
        model = ARIMA(train, order=(p, d, q))
        model_fit = model.fit()
        ll = model_fit.llf

        table_data.append([f"ARIMA({p},{d},{q})", f"{row['AIC']:.2f}", f"{row['BIC']:.2f}", f"{ll:.2f}"])
    except:
        table_data.append([f"ARIMA({p},{d},{q})", f"{row['AIC']:.2f}", f"{row['BIC']:.2f}", "Error"])

# Create the table
table = plt.table(cellText=table_data,
                  colLabels=headers,
                  loc='center',
                  cellLoc='center',
                  bbox=[0.1, 0.1, 0.8, 0.7])  # [left, bottom, width, height]

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.5)

# Highlight the best model by AIC and BIC
for i, model_info in enumerate(table_data):
    if f"ARIMA({best_order_aic[0]},{best_order_aic[1]},{best_order_aic[2]})" in model_info[0]:
        # Highlight AIC column
        table[(i+1, 1)].set_facecolor('#CCFFCC')  # Light green

    if f"ARIMA({best_order_bic[0]},{best_order_bic[1]},{best_order_bic[2]})" in model_info[0]:
        # Highlight BIC column
        table[(i+1, 2)].set_facecolor('#CCFFCC')  # Light green

# Remove axes
plt.axis('off')
plt.tight_layout()
plt.savefig(plots_dir + 'model_comparison_table.png')
plt.close()

print(f"\nModel comparison table saved to '{plots_dir}model_comparison_table.png'")

# Create model coefficient tests table 
plt.figure(figsize=(14, 8))

# Get coefficient values and significance tests for complex model
try:
    complex_model_summary = aic_model.summary()
    complex_model_table = pd.read_html(complex_model_summary.tables[1].as_html(), header=0)[0]
    # Keep only the coefficient rows (remove constants)
    complex_model_table = complex_model_table[~complex_model_table[''].str.contains('const', na=False)]
    
    # Create a coefficient test table
    plt.subplot(211)
    plt.text(0.5, 0.9, f"(a) Coefficients Test - ARIMA{best_order_aic} (Complex)", fontsize=14, ha='center', transform=plt.gca().transAxes)
    
    complex_cell_text = []
    for _, row in complex_model_table.iterrows():
        param_name = row['']
        coef = row['coef']
        std_err = row['std err']
        t_value = row['z'] if 'z' in complex_model_table.columns else row['t']
        p_value = row['P>|z|'] if 'P>|z|' in complex_model_table.columns else row['P>|t|']
        complex_cell_text.append([f"{param_name}", f"{coef:.4f}", f"{std_err:.4f}", f"{t_value:.4f}", f"{p_value:.4f}"])
    
    complex_column_labels = ['Parameter', 'Coefficient', 'Std Error', 'z-value', 'p-value']
    complex_table = plt.table(cellText=complex_cell_text, colLabels=complex_column_labels, 
                       loc='center', cellLoc='center', colColours=['#f2f2f2']*5)
    complex_table.auto_set_font_size(False)
    complex_table.set_fontsize(10)
    complex_table.scale(1.2, 1.5)
    plt.axis('off')
except:
    plt.text(0.5, 0.5, "Could not extract coefficient data for complex model", fontsize=12, ha='center')

# Get coefficient values and significance tests for simple model
try:
    simple_model_summary = bic_model.summary()
    simple_model_table = pd.read_html(simple_model_summary.tables[1].as_html(), header=0)[0]
    # Keep only the coefficient rows (remove constants)
    simple_model_table = simple_model_table[~simple_model_table[''].str.contains('const', na=False)]
    
    # Create a coefficient test table
    plt.subplot(212)
    plt.text(0.5, 0.9, f"(b) Coefficients Test - ARIMA{best_order_bic} (Simple)", fontsize=14, ha='center', transform=plt.gca().transAxes)
    
    simple_cell_text = []
    for _, row in simple_model_table.iterrows():
        param_name = row['']
        coef = row['coef']
        std_err = row['std err']
        t_value = row['z'] if 'z' in simple_model_table.columns else row['t']
        p_value = row['P>|z|'] if 'P>|z|' in simple_model_table.columns else row['P>|t|']
        simple_cell_text.append([f"{param_name}", f"{coef:.4f}", f"{std_err:.4f}", f"{t_value:.4f}", f"{p_value:.4f}"])
    
    simple_column_labels = ['Parameter', 'Coefficient', 'Std Error', 'z-value', 'p-value']
    simple_table = plt.table(cellText=simple_cell_text, colLabels=simple_column_labels, 
                     loc='center', cellLoc='center', colColours=['#f2f2f2']*5)
    simple_table.auto_set_font_size(False)
    simple_table.set_fontsize(10)
    simple_table.scale(1.2, 1.5)
    plt.axis('off')
except:
    plt.text(0.5, 0.5, "Could not extract coefficient data for simple model", fontsize=12, ha='center')

plt.tight_layout()
plt.savefig(plots_dir + 'coefficient_tests.png')
plt.close()

print(f"\nCoefficient tests saved to '{plots_dir}coefficient_tests.png'")

# Create a specialized Theil's U comparison plot
plt.figure(figsize=(10, 6))
plt.text(0.5, 0.9, "Theil's U Statistic Comparison", fontsize=16, ha='center', transform=plt.gca().transAxes)
plt.text(0.5, 0.8, "Measures forecast accuracy relative to naive forecast (U < 1 is better than naive)", 
         fontsize=12, ha='center', transform=plt.gca().transAxes)

# Create bar chart
models = [f'ARIMA{best_order_aic}\n(Complex)', f'ARIMA{best_order_bic}\n(Simple)', 'Naive Forecast']
u_values = [aic_theil_u, bic_theil_u, 1.0]
colors = ['royalblue', 'forestgreen', 'firebrick']

plt.bar(models, u_values, color=colors, alpha=0.7)
plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label="Naive Forecast Threshold")

# Add values on top of bars
for i, v in enumerate(u_values):
    plt.text(i, v + 0.05, f"{v:.4f}", ha='center', fontsize=10)

plt.ylabel("Theil's U Value")
plt.title("Theil's U Comparison (lower is better)")
plt.tight_layout()
plt.savefig(plots_dir + 'figure10_theil_u.png')
plt.close()

print(f"\nTheil's U comparison plot saved to '{plots_dir}figure10_theil_u.png'")

# After the existing plots, add the following additional plots:

# 1. Seasonal Decomposition plot
plt.figure(figsize=(14, 12))
plt.suptitle('Seasonal Decomposition of GDP Series', fontsize=16)

# Use seasonal_decompose with period=4 for quarterly data
decomposition = seasonal_decompose(df['GDP'], model='additive', period=4)

# Plot the original, trend, seasonal, and residual components
plt.subplot(411)
plt.plot(decomposition.observed)
plt.title('Original Series', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)

plt.subplot(412)
plt.plot(decomposition.trend)
plt.title('Trend Component', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)

plt.subplot(413)
plt.plot(decomposition.seasonal)
plt.title('Seasonal Component', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)

plt.subplot(414)
plt.plot(decomposition.resid)
plt.title('Residual Component', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.savefig(plots_dir + 'figure11_seasonal_decomposition.png')
plt.close()

print(f"\nSeasonal decomposition plot saved to '{plots_dir}figure11_seasonal_decomposition.png'")

# 2. Rolling Statistics Plot
plt.figure(figsize=(14, 8))
plt.suptitle('Rolling Statistics of GDP Series', fontsize=16)

# Calculate the rolling mean and standard deviation
rolling_mean = df['GDP'].rolling(window=12).mean()  # 3-year window (12 quarters)
rolling_std = df['GDP'].rolling(window=12).std()

# Plot the original data, rolling mean, and rolling std
plt.subplot(211)
plt.plot(df.index, df['GDP'], label='Original GDP', color='blue')
plt.plot(df.index, rolling_mean, label='Rolling Mean (12Q)', color='red')
plt.title('GDP with Rolling Mean', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('GDP', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.subplot(212)
plt.plot(df.index, rolling_std, label='Rolling Std Dev (12Q)', color='green')
plt.title('Rolling Standard Deviation', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Standard Deviation', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.savefig(plots_dir + 'figure12_rolling_statistics.png')
plt.close()

print(f"\nRolling statistics plot saved to '{plots_dir}figure12_rolling_statistics.png'")

# 3. Residual Squares Plot (to check for heteroskedasticity)
plt.figure(figsize=(14, 6))

# Squared residuals for both models
aic_residuals_squared = aic_residuals**2
bic_residuals_squared = bic_residuals**2

plt.subplot(211)
plt.plot(aic_residuals_squared, color='blue')
plt.title(f'(a) Squared Residuals - ARIMA{best_order_aic} (Complex)', fontsize=14)
plt.xlabel('Observation', fontsize=12)
plt.ylabel('Squared Residual', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

plt.subplot(212)
plt.plot(bic_residuals_squared, color='green')
plt.title(f'(b) Squared Residuals - ARIMA{best_order_bic} (Simple)', fontsize=14)
plt.xlabel('Observation', fontsize=12)
plt.ylabel('Squared Residual', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(plots_dir + 'figure13_residual_squares.png')
plt.close()

print(f"\nResidual squares plot saved to '{plots_dir}figure13_residual_squares.png'")

# 4. ARCH LM Test for Heteroskedasticity
plt.figure(figsize=(10, 6))
plt.text(0.5, 0.9, "ARCH LM Test for Heteroskedasticity", fontsize=16, ha='center', transform=plt.gca().transAxes)

# Perform ARCH LM test
complex_arch_test = het_arch(aic_residuals, nlags=12)
simple_arch_test = het_arch(bic_residuals, nlags=12)

# Create table data
test_data = [
    ['Model', 'LM Statistic', 'p-value', 'F Statistic', 'p-value (F)', 'Conclusion'],
    [f'ARIMA{best_order_aic} (Complex)', 
     f"{complex_arch_test[0]:.4f}", 
     f"{complex_arch_test[1]:.4f}",
     f"{complex_arch_test[2]:.4f}", 
     f"{complex_arch_test[3]:.4f}",
     "Homoskedastic" if complex_arch_test[1] > 0.05 else "Heteroskedastic"],
    [f'ARIMA{best_order_bic} (Simple)', 
     f"{simple_arch_test[0]:.4f}", 
     f"{simple_arch_test[1]:.4f}",
     f"{simple_arch_test[2]:.4f}", 
     f"{simple_arch_test[3]:.4f}",
     "Homoskedastic" if simple_arch_test[1] > 0.05 else "Heteroskedastic"]
]

# Create the table
arch_table = plt.table(cellText=test_data[1:],
                      colLabels=test_data[0],
                      loc='center',
                      cellLoc='center')

# Style the table
arch_table.auto_set_font_size(False)
arch_table.set_fontsize(10)
arch_table.scale(1.2, 1.5)

# Hide axes
plt.axis('off')
plt.tight_layout()
plt.savefig(plots_dir + 'figure14_arch_test.png')
plt.close()

print(f"\nARCH test plot saved to '{plots_dir}figure14_arch_test.png'")

# 5. Long-term Forecast with Confidence Intervals
# We'll use the better model (complex) for this
forecast_steps = min(20, len(test))  # 20 steps or the length of test data, whichever is smaller

# Get forecast with confidence intervals
forecast_results = aic_model.get_forecast(steps=forecast_steps)
forecast_mean = forecast_results.predicted_mean
forecast_conf_int = forecast_results.conf_int(alpha=0.05)  # 95% confidence interval

# Convert the differenced forecast back to original scale
forecast_original = []
last_value = df['GDP'][train_size-1]  # Last observed training value

for f in forecast_mean:
    if d_order == 1:
        next_value = last_value + f
    elif d_order == 2:
        # This is simplified
        next_value = 2*last_value - df['GDP'][train_size-2] + f
    else:
        next_value = f
    
    forecast_original.append(next_value)
    last_value = next_value

# Also convert confidence intervals
lower_ci_original = []
upper_ci_original = []
last_value = df['GDP'][train_size-1]
last_lower = last_value
last_upper = last_value

for i in range(len(forecast_mean)):
    f_lower = forecast_conf_int.iloc[i, 0]
    f_upper = forecast_conf_int.iloc[i, 1]
    
    if d_order == 1:
        next_lower = last_value + f_lower
        next_upper = last_value + f_upper
    elif d_order == 2:
        next_lower = 2*last_value - df['GDP'][train_size-2] + f_lower
        next_upper = 2*last_value - df['GDP'][train_size-2] + f_upper
    else:
        next_lower = f_lower
        next_upper = f_upper
    
    lower_ci_original.append(next_lower)
    upper_ci_original.append(next_upper)
    
    # Update the last value for the next iteration
    last_value = forecast_original[i]

# Plot the long-term forecast with confidence intervals
plt.figure(figsize=(14, 8))

# Plot the historical data
historical_end = train_size + forecast_steps
plt.plot(df.index[:train_size], df['GDP'][:train_size], label='Historical Data (Training)', color='blue')
plt.plot(df.index[train_size:historical_end], df['GDP'][train_size:historical_end], label='Actual Values (Test)', color='green')

# Plot the forecast and confidence intervals
forecast_index = df.index[train_size:train_size+forecast_steps]
plt.plot(forecast_index, forecast_original, label=f'ARIMA{best_order_aic} Forecast', color='red', linestyle='--')
plt.fill_between(forecast_index, lower_ci_original, upper_ci_original, color='red', alpha=0.2, label='95% Confidence Interval')

plt.title('Long-term GDP Forecast with 95% Confidence Intervals', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('GDP', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Format x-axis to show fewer dates
locator = mdates.YearLocator(5)  # Show every 5 years
plt.gca().xaxis.set_major_locator(locator)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout()
plt.savefig(plots_dir + 'figure15_long_term_forecast.png')
plt.close()

print(f"\nLong-term forecast plot saved to '{plots_dir}figure15_long_term_forecast.png'")