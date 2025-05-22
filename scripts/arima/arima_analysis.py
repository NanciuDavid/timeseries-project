#!/usr/bin/env python3
# ARIMA Analysis for Gold Price Time Series
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

# Set plotting style
plt.style.use('ggplot')  # Using a built-in style instead of 'seaborn'
sns.set_palette("deep")
plt.rcParams['figure.figsize'] = [12, 7]
plt.rcParams['figure.dpi'] = 100

# Define plots directory
plots_dir = "plots/arima/"

# Load the data
print("Loading and preparing the data...")
data_path = "datasets/gold-part1/Gold.csv"  # Path relative to project root
df = pd.read_csv(data_path)

# Convert the DATE column to datetime
df['DATE'] = pd.to_datetime(df['DATE'], format='%m/%d/%Y')

# Set DATE as the index
df.set_index('DATE', inplace=True)

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
plt.plot(df.index, df['VALUE'], color='blue', linewidth=1.5)
plt.title('(a) Gold Price Time Series', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price (USD)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# Log transformation (for comparison, even if not necessary)
plt.subplot(312)
plt.plot(df.index, np.log(df['VALUE']), color='green', linewidth=1.5)
plt.title('(b) Log Transformation of Gold Price', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Log(Price)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# First difference
df['first_diff'] = df['VALUE'].diff()
plt.subplot(313)
plt.plot(df.index[1:], df['first_diff'][1:], color='red', linewidth=1)
plt.title('(c) First Difference of Gold Price', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price Difference', fontsize=12)
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
is_stationary = adf_test(df['VALUE'], title='Original Series')

# Create a table with stationarity test results - similar to example PDF
test_results = pd.DataFrame({
    'Test': ['ADF Test', 'Phillips-Perron', 'KPSS Test'],
    'p-value': [f"{adfuller(df['VALUE'].dropna())[1]:.4f}", 'N/A', 'N/A'],
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
    stationary_series = df['VALUE']
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
        ['ADF p-value', f"{adfuller(df['VALUE'].dropna())[1]:.4f}", f"{adfuller(df_diff.dropna())[1]:.8f}"],
        ['Conclusion', 'Non-stationary', 'Stationary']
    ]
elif d_order == 2:
    table_data = [
        ['Test', 'Original Series', 'First Difference', 'Second Difference'],
        ['ADF p-value', f"{adfuller(df['VALUE'].dropna())[1]:.4f}", f"{adfuller(df_diff.dropna())[1]:.4f}", f"{adfuller(df_diff2.dropna())[1]:.8f}"],
        ['Conclusion', 'Non-stationary', 'Non-stationary', 'Stationary']
    ]
else:
    table_data = [
        ['Test', 'Original Series'],
        ['ADF p-value', f"{adfuller(df['VALUE'].dropna())[1]:.4f}"],
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

# We'll proceed with the best model by AIC
best_model = best_model_aic
best_order = best_order_aic

# Also fit the best model by BIC for comparison (similar to example PDF)
bic_model = ARIMA(train, order=best_order_bic).fit()

print("\n--- Model Summary ---")
print(best_model.summary())

# Save model summary to file
with open(plots_dir + 'arima_model_summary.txt', 'w') as f:
    f.write(str(best_model.summary()))
print(f"\nModel summary saved to '{plots_dir}arima_model_summary.txt'")

# Create a visual representation of the model equation - similar to example PDF
plt.figure(figsize=(10, 6))
plt.text(0.5, 0.7, 
         r"$ARIMA({0},{1},{2})$ model:".format(best_order[0], best_order[1], best_order[2]), 
         fontsize=16, ha='center')

# Extract coefficients
ar_params = best_model.arparams
ma_params = best_model.maparams

# Create the model equation text
ar_eq = "(1"
for i, param in enumerate(ar_params, 1):
    ar_eq += f" + {param:.4f}B^{i}"
ar_eq += ")"

ma_eq = "(1"
for i, param in enumerate(ma_params, 1):
    ma_eq += f" + {param:.4f}B^{i}"
ma_eq += ")"

# Full equation
full_eq = f"{ar_eq}(∇^{best_order[1]}Gold_t) = {ma_eq}(ε_t)"

plt.text(0.5, 0.5, full_eq, fontsize=14, ha='center')
plt.text(0.5, 0.3, "Where:", fontsize=12, ha='center')
plt.text(0.5, 0.2, "B is the backshift operator", fontsize=12, ha='center')
plt.text(0.5, 0.1, "∇ represents differencing", fontsize=12, ha='center')

# Remove axes
plt.axis('off')
plt.tight_layout()
plt.savefig(plots_dir + 'model_equation.png')
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

# Model Diagnostic Checking - Similar to Figure 3 in example PDF
print("\n--- Model Diagnostic Checking ---")

# Get residuals
residuals = best_model.resid

# Create model coefficient tests table - similar to Fig 4 in example PDF
plt.figure(figsize=(14, 8))

# Get coefficient values and significance tests
aic_model_coefs = pd.DataFrame({
    'Parameter': ['AR(1)', 'AR(2)', 'MA(1)', 'MA(2)'],
    'Coefficient': best_model.arparams.tolist() + best_model.maparams.tolist(),
    'Std Error': [best_model.bse[i] for i in range(len(best_model.arparams) + len(best_model.maparams))],
    'z-value': [best_model.tvalues[i] for i in range(len(best_model.arparams) + len(best_model.maparams))],
    'p-value': [best_model.pvalues[i] for i in range(len(best_model.arparams) + len(best_model.maparams))]
})

# Plot coefficients test for AIC model
plt.subplot(211)
plt.text(0.5, 0.9, f"(a) Coefficients test for ARIMA{best_order_aic}", fontsize=14, ha='center', transform=plt.gca().transAxes)

# Create the table for AIC model
cell_text = [[f"{row['Parameter']}", f"{row['Coefficient']:.4f}", f"{row['Std Error']:.4f}", 
              f"{row['z-value']:.4f}", f"{row['p-value']:.4f}"] for _, row in aic_model_coefs.iterrows()]
column_labels = ['Parameter', 'Coefficient', 'Std Error', 'z-value', 'p-value']

aic_table = plt.table(cellText=cell_text, colLabels=column_labels, loc='center', 
                      cellLoc='center', colColours=['#f2f2f2']*5)
aic_table.auto_set_font_size(False)
aic_table.set_fontsize(10)
aic_table.scale(1.2, 1.5)
plt.axis('off')

# Plot Ljung-Box and Jarque-Bera test results
plt.subplot(212)
plt.text(0.5, 0.9, "(b) Diagnostic Tests", fontsize=14, ha='center', transform=plt.gca().transAxes)

# Extract test statistics
lb_test = acorr_ljungbox(residuals, lags=10)
jb_test = stats.jarque_bera(residuals)

diag_table_data = [
    ['Test', 'Statistic', 'p-value', 'Conclusion'],
    ['Ljung-Box (lag 10)', f"{lb_test.iloc[0, 0]:.4f}", f"{lb_test.iloc[0, 1]:.8f}", 'Reject H0' if lb_test.iloc[0, 1] < 0.05 else 'Fail to reject H0'],
    ['Jarque-Bera', f"{jb_test[0]:.4f}", f"{jb_test[1]:.8f}", 'Reject H0' if jb_test[1] < 0.05 else 'Fail to reject H0']
]

diag_table = plt.table(cellText=diag_table_data[1:], colLabels=diag_table_data[0], 
                       loc='center', cellLoc='center', colColours=['#f2f2f2']*4)
diag_table.auto_set_font_size(False)
diag_table.set_fontsize(10)
diag_table.scale(1.2, 1.5)
plt.axis('off')

plt.tight_layout()
plt.savefig(plots_dir + 'coefficient_tests.png')
plt.close()

print(f"\nCoefficient tests saved to '{plots_dir}coefficient_tests.png'")

# Plot residuals (similar to Fig 3)
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Time plot of residuals
axes[0, 0].plot(residuals.index, residuals, color='blue')
axes[0, 0].set_title('(a) Time Plot of Residuals', fontsize=14)
axes[0, 0].set_xlabel('Date', fontsize=12)
axes[0, 0].set_ylabel('Residual Value', fontsize=12)
axes[0, 0].grid(True, linestyle='--', alpha=0.7)

# Histogram of residuals
axes[0, 1].hist(residuals, bins=30, density=True, color='blue', alpha=0.7)
xmin, xmax = axes[0, 1].get_xlim()
x = np.linspace(xmin, xmax, 100)
axes[0, 1].plot(x, np.exp(-(x**2)/2)/(np.sqrt(2*np.pi)), 'r', linewidth=2)
axes[0, 1].set_title('(b) Histogram of Residuals with Normal Curve', fontsize=14)
axes[0, 1].set_xlabel('Residual Value', fontsize=12)
axes[0, 1].set_ylabel('Density', fontsize=12)
axes[0, 1].grid(True, linestyle='--', alpha=0.7)

# ACF of residuals
plot_acf(residuals, ax=axes[1, 0], lags=40, alpha=0.05)
axes[1, 0].set_title('(c) ACF of Residuals', fontsize=14)
axes[1, 0].grid(True, linestyle='--', alpha=0.7)

# QQ plot of residuals
stats.probplot(residuals, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('(d) Q-Q Plot of Residuals', fontsize=14)
axes[1, 1].grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(plots_dir + 'figure3_residual_diagnostics.png')
plt.close()

print(f"\nResidual diagnostics plots saved to '{plots_dir}figure3_residual_diagnostics.png'")

# Forecasting - Similar to Figures 8-9 in example PDF
print("\n--- Forecasting ---")

# Forecast on test set
forecast_steps = len(test)
forecast = best_model.get_forecast(steps=forecast_steps)
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()

# Create a plot similar to Figure 8 in example PDF
plt.figure(figsize=(15, 7))
plt.plot(forecast_mean.index, forecast_mean, 'r-', label='Point Forecast')
plt.fill_between(forecast_mean.index, 
                 forecast_ci.iloc[:, 0], 
                 forecast_ci.iloc[:, 1], 
                 color='pink', alpha=0.3)
plt.plot(test.index, test, 'b-', label='Actual Values')

plt.title('Gold Price Forecast (Differenced Series)', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Price Difference', fontsize=14)
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.7)

# Add error metrics as a table in the plot
mse = mean_squared_error(test, forecast_mean)
rmse = np.sqrt(mse)
try:
    mape = np.mean(np.abs((test.values - forecast_mean.values) / test.values)) * 100
except:
    mape = float('nan')

error_text = f"MSE: {mse:.2f}\nRMSE: {rmse:.2f}\nMAPE: {mape:.2f}%"
plt.text(0.02, 0.96, "Error Metrics:", transform=plt.gca().transAxes, fontsize=12, 
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
plt.text(0.02, 0.90, error_text, transform=plt.gca().transAxes, fontsize=10, 
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(plots_dir + 'figure8_point_forecast.png')
plt.close()

print(f"\nForecast plot saved to '{plots_dir}figure8_point_forecast.png'")

# Evaluate forecast accuracy
print("\nForecast Accuracy Metrics:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

# If the series was differenced, we need to convert the forecasts back to the original scale
if d_order > 0:
    print("\n--- Converting Forecasts to Original Scale ---")

    # Create Figure 9 in example PDF - the original scale forecast vs actual
    plt.figure(figsize=(15, 10))

    # Get the last values from the original series to use for undifferencing
    last_values = df['VALUE'].iloc[train_size-d_order:train_size].values

    # Function to undo differencing
    def inverse_difference(history, yhat, intervals=1):
        return yhat + history[-intervals]

    # Convert forecast to original scale
    original_scale_forecast = []
    original_scale_ci_lower = []
    original_scale_ci_upper = []

    if d_order == 1:
        # Undo first differencing
        last_value = df['VALUE'].iloc[train_size-1]

        # Initialize with last known value
        history = [last_value]

        # Undo differencing for each forecast step
        for i in range(len(forecast_mean)):
            # Undo differencing for mean forecast
            inverted = inverse_difference(history, forecast_mean.iloc[i])
            original_scale_forecast.append(inverted)
            history.append(inverted)

            # Undo differencing for confidence intervals
            original_scale_ci_lower.append(inverse_difference([history[0]], forecast_ci.iloc[i, 0]))
            original_scale_ci_upper.append(inverse_difference([history[0]], forecast_ci.iloc[i, 1]))

    elif d_order == 2:
        # For second order differencing, we need the last two values
        last_values = df['VALUE'].iloc[train_size-2:train_size].values
        last_diff = df['first_diff'].iloc[train_size-1]

        # Initialize history with last known values
        history_values = list(last_values)
        history_diff = [last_diff]

        # Undo differencing for each forecast step
        for i in range(len(forecast_mean)):
            # First undo second differencing to get first difference
            inverted_diff = inverse_difference(history_diff, forecast_mean.iloc[i])
            history_diff.append(inverted_diff)

            # Then undo first differencing to get original scale
            inverted = inverse_difference(history_values, inverted_diff)
            original_scale_forecast.append(inverted)
            history_values.append(inverted)

            # Similar process for confidence intervals
            lower_diff = inverse_difference(history_diff[:1], forecast_ci.iloc[i, 0])
            upper_diff = inverse_difference(history_diff[:1], forecast_ci.iloc[i, 1])

            original_scale_ci_lower.append(inverse_difference(history_values[:1], lower_diff))
            original_scale_ci_upper.append(inverse_difference(history_values[:1], upper_diff))

    # Create a DataFrame with the original scale forecasts
    original_forecast_df = pd.DataFrame({
        'Forecast': original_scale_forecast,
        'Lower CI': original_scale_ci_lower,
        'Upper CI': original_scale_ci_upper
    }, index=test.index)

    # Create subplots for a layout similar to Figure 9 in the example PDF
    plt.subplot(211)
    plt.plot(test.index, df['VALUE'].loc[test.index], label='Actual', color='blue')
    plt.plot(original_forecast_df.index, original_forecast_df['Forecast'], label='Forecast', color='red')
    plt.fill_between(original_forecast_df.index, 
                    original_forecast_df['Lower CI'], 
                    original_forecast_df['Upper CI'], 
                    color='pink', alpha=0.3, label='95% CI')
    plt.title('(a) Gold Price Forecast vs Actual (Original Scale)', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Gold Price', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Plot the fitted values vs actual for training data
    plt.subplot(212)
    fitted_values = best_model.fittedvalues
    plt.plot(train.index, train, label='Actual', color='blue')
    plt.plot(fitted_values.index, fitted_values, label='Fitted', color='green')
    plt.title('(b) Fitted Values vs Actual (Training Data)', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price Difference', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(plots_dir + 'figure9_forecast_comparison.png')
    plt.close()

    print(f"\nOriginal scale forecast comparison saved to '{plots_dir}figure9_forecast_comparison.png'")

    # Evaluate forecast accuracy on the original scale - make sure the lengths match
    actual_values = df['VALUE'].iloc[train_size:].values[:len(original_forecast_df)]
    predicted_values = original_forecast_df['Forecast'].values

    original_mse = mean_squared_error(actual_values, predicted_values)
    original_rmse = np.sqrt(original_mse)
    original_mape = np.mean(np.abs((actual_values - predicted_values) / actual_values)) * 100

    print("\nForecast Accuracy Metrics (Original Scale):")
    print(f"Mean Squared Error (MSE): {original_mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {original_rmse:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {original_mape:.2f}%")

    # Calculate Theil's U statistic (as shown in Figure 10 of example PDF)
    from sklearn.metrics import mean_squared_error
    
    # Calculate naive forecast (using previous value as forecast)
    naive_forecast = df['VALUE'].iloc[train_size-1:-1].values
    
    # Calculate MSE for naive forecast
    naive_mse = mean_squared_error(actual_values, naive_forecast[:len(actual_values)])
    
    # Theil's U statistic
    theil_u = np.sqrt(original_mse) / np.sqrt(naive_mse)
    
    # Create Figure 10 - Theil's U visualization
    plt.figure(figsize=(8, 6))
    plt.text(0.5, 0.8, "Theil's U Statistic", fontsize=16, ha='center')
    plt.text(0.5, 0.6, f"U = {theil_u:.4f}", fontsize=14, ha='center')
    
    # Add interpretation
    if theil_u < 1:
        interpretation = f"U < 1: The forecast is better than a naive guess\n({theil_u:.2f} < 1)"
    elif theil_u == 1:
        interpretation = f"U = 1: The forecast is equal to a naive guess\n({theil_u:.2f} = 1)"
    else:
        interpretation = f"U > 1: The forecast is worse than a naive guess\n({theil_u:.2f} > 1)"
    
    plt.text(0.5, 0.4, interpretation, fontsize=12, ha='center')
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(plots_dir + 'figure10_theil_u.png')
    plt.close()
    
    print(f"\nTheil's U statistic plot saved to '{plots_dir}figure10_theil_u.png'")
    print(f"Theil's U value: {theil_u:.4f}")
    
print("\nARIMA analysis complete!")