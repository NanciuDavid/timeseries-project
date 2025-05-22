#!/usr/bin/env python3
# ARIMA Intervention Analysis for US GDP Time Series
# Extension: Adding COVID-19 intervention dummy variables

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# Set plotting style
plt.style.use('ggplot')
sns.set_palette("deep")
plt.rcParams['figure.figsize'] = [12, 7]
plt.rcParams['figure.dpi'] = 100

# Define plots directory
plots_dir = "plots/arima/gdp/intervention/"
import os
os.makedirs(plots_dir, exist_ok=True)

# Load the data
print("Loading and preparing the data...")
data_path = "datasets/USGDP/GDP.csv"
df = pd.read_csv(data_path, delimiter=';', skiprows=1)

# Convert the observation_date column to datetime
df['observation_date'] = pd.to_datetime(df['observation_date'])

# Set observation_date as the index
df.set_index('observation_date', inplace=True)

# Print dataset info
print(f"Dataset shape: {df.shape}")
print(f"Date range: {df.index.min()} to {df.index.max()}")

# Create COVID-19 intervention dummy variables
# 1. Initial shock dummy (Q1-Q2 2020 - sharp decline)
df['covid_shock'] = 0
covid_shock_period = (df.index >= '2020-01-01') & (df.index <= '2020-06-30')
df.loc[covid_shock_period, 'covid_shock'] = 1

# 2. Recovery dummy (Q3 2020 - Q4 2020 - rapid rebound)
df['covid_recovery'] = 0
covid_recovery_period = (df.index >= '2020-07-01') & (df.index <= '2020-12-31')
df.loc[covid_recovery_period, 'covid_recovery'] = 1

# 3. Post-covid growth dummy (2021 onwards - different growth trajectory)
df['post_covid'] = 0
post_covid_period = df.index >= '2021-01-01'
df.loc[post_covid_period, 'post_covid'] = 1

# Plot the GDP series with intervention periods highlighted
plt.figure(figsize=(14, 8))
plt.plot(df.index, df['GDP'], color='blue', linewidth=2, label='US GDP')

# Add vertical spans for intervention periods
plt.axvspan(pd.Timestamp('2020-01-01'), pd.Timestamp('2020-06-30'),
            color='red', alpha=0.2, label='COVID Shock')
plt.axvspan(pd.Timestamp('2020-07-01'), pd.Timestamp('2020-12-31'),
            color='green', alpha=0.2, label='COVID Recovery')
plt.axvspan(pd.Timestamp('2021-01-01'), df.index.max(),
            color='purple', alpha=0.2, label='Post-COVID')

plt.title('US GDP with COVID-19 Intervention Periods', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('GDP (Billions USD)', fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(plots_dir + 'gdp_with_interventions.png')
plt.close()

print(f"GDP with intervention periods plot saved to '{plots_dir}gdp_with_interventions.png'")

# Check the dummies were created properly
print("\nDummy Variables Summary:")
print(f"COVID Shock periods (2020 Q1-Q2): {df['covid_shock'].sum()} quarters")
print(f"COVID Recovery periods (2020 Q3-Q4): {df['covid_recovery'].sum()} quarters")
print(f"Post-COVID periods (2021 onwards): {df['post_covid'].sum()} quarters")

# Split data into training and testing sets (80% training, 20% testing)
train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]

print(f"\nTraining set: {train.index.min()} to {train.index.max()}")
print(f"Testing set: {test.index.min()} to {test.index.max()}")

# Determine differencing order based on stationarity test
def adf_test(series, title=''):
    result = adfuller(series.dropna())
    print(f"Augmented Dickey-Fuller Test for {title}")
    print(f"ADF Test Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")
    print(f"Critical Values:")
    for key, value in result[4].items():
        print(f"    {key}: {value:.4f}")
    if result[1] <= 0.05:
        print("Series is stationary.")
        return True
    else:
        print("Series is non-stationary.")
        return False

# Test stationarity of original series
print("\n--- Stationarity Test ---")
is_stationary = adf_test(df['GDP'], title='Original GDP Series')

# Apply differencing if needed
d_order = 0
if not is_stationary:
    print("\nApplying first differencing...")
    df['diff_GDP'] = df['GDP'].diff()
    is_stationary_diff = adf_test(df['diff_GDP'].dropna(), title='First Difference')
    if is_stationary_diff:
        d_order = 1
        print("\nSeries is stationary after first differencing (d=1)")
    else:
        df['diff2_GDP'] = df['diff_GDP'].diff()
        is_stationary_diff2 = adf_test(df['diff2_GDP'].dropna(), title='Second Difference')
        if is_stationary_diff2:
            d_order = 2
            print("\nSeries is stationary after second differencing (d=2)")

print(f"\nDifferencing order: d = {d_order}")

# Fit models and compare
print("\n--- Model Fitting ---")

# 1. Original ARIMA model without intervention (based on previous analysis)
original_order = (3, 1, 3)  # From previous analysis
print(f"\nFitting original ARIMA{original_order} model (no intervention)...")
arima_model = ARIMA(train['GDP'], order=original_order)
arima_results = arima_model.fit()
print(f"AIC: {arima_results.aic:.2f}, BIC: {arima_results.bic:.2f}")

# 2. ARIMAX model with intervention dummies
print(f"\nFitting ARIMAX{original_order} model with intervention dummies...")
exog_vars = train[['covid_shock', 'covid_recovery', 'post_covid']]
arimax_model = ARIMA(train['GDP'],
                    exog=exog_vars,
                    order=original_order)
arimax_results = arimax_model.fit()
print(f"AIC: {arimax_results.aic:.2f}, BIC: {arimax_results.bic:.2f}")

# 3. Simple ARIMA model for comparison
simple_order = (0, 1, 1)
print(f"\nFitting simple ARIMA{simple_order} model (no intervention)...")
simple_model = ARIMA(train['GDP'], order=simple_order)
simple_results = simple_model.fit()
print(f"AIC: {simple_results.aic:.2f}, BIC: {simple_results.bic:.2f}")

# 4. Simple ARIMAX with intervention
print(f"\nFitting simple ARIMAX{simple_order} model with intervention dummies...")
simple_arimax_model = ARIMA(train['GDP'],
                           exog=exog_vars,
                           order=simple_order)
simple_arimax_results = simple_arimax_model.fit()
print(f"AIC: {simple_arimax_results.aic:.2f}, BIC: {simple_arimax_results.bic:.2f}")

# Model comparison table
models = [
    ('ARIMA(3,1,3)', arima_results.aic, arima_results.bic),
    ('ARIMAX(3,1,3) with intervention', arimax_results.aic, arimax_results.bic),
    ('ARIMA(0,1,1)', simple_results.aic, simple_results.bic),
    ('ARIMAX(0,1,1) with intervention', simple_arimax_results.aic, simple_arimax_results.bic)
]

print("\n--- Model Comparison ---")
print(f"{'Model':<30} {'AIC':<12} {'BIC':<12}")
print("-" * 54)
for model, aic, bic in models:
    print(f"{model:<30} {aic:<12.2f} {bic:<12.2f}")

# Create a visual comparison table
plt.figure(figsize=(10, 6))
plt.text(0.5, 0.95, "Model Comparison with Intervention Analysis",
         ha='center', va='center', fontsize=16, fontweight='bold')

table_data = [[model, f"{aic:.2f}", f"{bic:.2f}"] for model, aic, bic in models]
headers = ["Model", "AIC", "BIC"]

table = plt.table(cellText=table_data, colLabels=headers,
                 loc='center', cellLoc='center', colColours=['#f2f2f2']*3)
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.5)

# Highlight the best model (lowest AIC)
best_model_idx = min(range(len(models)), key=lambda i: models[i][1])
for j in range(3):  # for each column
    table[(best_model_idx+1, j)].set_facecolor('#CCFFCC')

plt.axis('off')
plt.tight_layout()
plt.savefig(plots_dir + 'model_comparison_table.png')
plt.close()

print(f"\nModel comparison table saved to '{plots_dir}model_comparison_table.png'")

# Examine the intervention coefficients
print("\n--- Intervention Coefficients ---")
print(arimax_results.summary().tables[1])

# Create a coefficients table plot
plt.figure(figsize=(12, 6))
plt.text(0.5, 0.95, "ARIMAX Model Coefficients with COVID-19 Intervention",
         ha='center', va='center', fontsize=16, fontweight='bold')

# Extract coefficient information
coef_summary = arimax_results.summary().tables[1].data
coef_data = []
for i in range(1, len(coef_summary)):
    coef_data.append([str(coef_summary[i][0]),
                     str(coef_summary[i][1]),
                     str(coef_summary[i][2]),
                     str(coef_summary[i][3]),
                     str(coef_summary[i][4])])

coef_table = plt.table(cellText=coef_data, colLabels=coef_summary[0],
                      loc='center', cellLoc='center', colColours=['#f2f2f2']*5)
coef_table.auto_set_font_size(False)
coef_table.set_fontsize(12)
coef_table.scale(1.2, 1.5)

# Highlight significant intervention coefficients (p<0.05)
for i, row in enumerate(coef_data):
    var_name = row[0]
    p_value = float(row[4]) if row[4] != 'nan' else 1.0
    if ('covid' in var_name or 'post_covid' in var_name) and p_value < 0.05:
        for j in range(5):
            coef_table[(i+1, j)].set_facecolor('#CCFFCC')

plt.axis('off')
plt.tight_layout()
plt.savefig(plots_dir + 'intervention_coefficients.png')
plt.close()

print(f"\nIntervention coefficients table saved to '{plots_dir}intervention_coefficients.png'")

# Out-of-sample forecasting with both models
print("\n--- Forecast Comparison ---")

# Function to inverse difference the predictions (if needed)
def inverse_difference(history, yhat, intervals=1):
    return yhat + history[-intervals]

# Prepare test data exogenous variables
test_exog = test[['covid_shock', 'covid_recovery', 'post_covid']]

# Generate forecasts
# 1. ARIMA forecast
arima_forecast = arima_results.get_forecast(steps=len(test))
arima_predictions = arima_forecast.predicted_mean

# 2. ARIMAX forecast with intervention dummies
arimax_forecast = arimax_results.get_forecast(steps=len(test), exog=test_exog)
arimax_predictions = arimax_forecast.predicted_mean

# 3. Simple ARIMA forecast
simple_forecast = simple_results.get_forecast(steps=len(test))
simple_predictions = simple_forecast.predicted_mean

# 4. Simple ARIMAX forecast with intervention
simple_arimax_forecast = simple_arimax_results.get_forecast(steps=len(test), exog=test_exog)
simple_arimax_predictions = simple_arimax_forecast.predicted_mean

# Plot forecasts
plt.figure(figsize=(14, 10))

# Plot actual values
plt.plot(test.index, test['GDP'], color='blue', label='Actual GDP', linewidth=2)

# Plot forecasts
plt.plot(test.index, arima_predictions, color='red',
         label='ARIMA(3,1,3)', linestyle='--', linewidth=1.5)
plt.plot(test.index, arimax_predictions, color='green',
         label='ARIMAX(3,1,3) with intervention', linestyle='-', linewidth=1.5)
plt.plot(test.index, simple_predictions, color='purple',
         label='ARIMA(0,1,1)', linestyle='--', linewidth=1.5)
plt.plot(test.index, simple_arimax_predictions, color='orange',
         label='ARIMAX(0,1,1) with intervention', linestyle='-', linewidth=1.5)

# Add COVID period shading
if any(covid_shock_period & (df.index >= test.index.min()) & (df.index <= test.index.max())):
    plt.axvspan(max(pd.Timestamp('2020-01-01'), test.index.min()),
                pd.Timestamp('2020-06-30'), color='red', alpha=0.1)

if any(covid_recovery_period & (df.index >= test.index.min()) & (df.index <= test.index.max())):
    plt.axvspan(pd.Timestamp('2020-07-01'),
                pd.Timestamp('2020-12-31'), color='green', alpha=0.1)

if any(post_covid_period & (df.index >= test.index.min()) & (df.index <= test.index.max())):
    plt.axvspan(pd.Timestamp('2021-01-01'),
                test.index.max(), color='purple', alpha=0.1)

plt.title('GDP Forecast Comparison with and without Intervention', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('GDP (Billions USD)', fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(plots_dir + 'forecast_comparison.png')
plt.close()

print(f"\nForecast comparison plot saved to '{plots_dir}forecast_comparison.png'")

# Calculate forecast accuracy metrics
# 1. RMSE
arima_rmse = np.sqrt(mean_squared_error(test['GDP'], arima_predictions))
arimax_rmse = np.sqrt(mean_squared_error(test['GDP'], arimax_predictions))
simple_rmse = np.sqrt(mean_squared_error(test['GDP'], simple_predictions))
simple_arimax_rmse = np.sqrt(mean_squared_error(test['GDP'], simple_arimax_predictions))

# 2. MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

arima_mape = mean_absolute_percentage_error(test['GDP'], arima_predictions)
arimax_mape = mean_absolute_percentage_error(test['GDP'], arimax_predictions)
simple_mape = mean_absolute_percentage_error(test['GDP'], simple_predictions)
simple_arimax_mape = mean_absolute_percentage_error(test['GDP'], simple_arimax_predictions)

# 3. MAE
def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

arima_mae = mean_absolute_error(test['GDP'], arima_predictions)
arimax_mae = mean_absolute_error(test['GDP'], arimax_predictions)
simple_mae = mean_absolute_error(test['GDP'], simple_predictions)
simple_arimax_mae = mean_absolute_error(test['GDP'], simple_arimax_predictions)

# 4. Theil's U
def theil_u(y_true, y_pred):
    # For naive forecast, use previous values (no change forecast)
    naive_forecast = y_true[:-1]
    actual = y_true[1:]
    # Calculate Theil's U
    mse_model = np.mean((actual - y_pred[:-1])**2)
    mse_naive = np.mean((actual - naive_forecast)**2)
    return np.sqrt(mse_model / mse_naive)

# Ensure lengths match for Theil's U calculation
if len(test) > 1:  # Need at least 2 observations for Theil's U
    arima_theil = theil_u(test['GDP'].values, arima_predictions.values)
    arimax_theil = theil_u(test['GDP'].values, arimax_predictions.values)
    simple_theil = theil_u(test['GDP'].values, simple_predictions.values)
    simple_arimax_theil = theil_u(test['GDP'].values, simple_arimax_predictions.values)
else:
    arima_theil = arimax_theil = simple_theil = simple_arimax_theil = np.nan

# Print accuracy metrics
print("\n--- Forecast Accuracy Metrics ---")
print(f"{'Model':<30} {'RMSE':<12} {'MAPE (%)':<12} {'MAE':<12} {'Theil\'s U':<12}")
print("-" * 78)
print(f"{'ARIMA(3,1,3)':<30} {arima_rmse:<12.2f} {arima_mape:<12.2f} {arima_mae:<12.2f} {arima_theil:<12.4f}")
print(f"{'ARIMAX(3,1,3) with intervention':<30} {arimax_rmse:<12.2f} {arimax_mape:<12.2f} {arimax_mae:<12.2f} {arimax_theil:<12.4f}")
print(f"{'ARIMA(0,1,1)':<30} {simple_rmse:<12.2f} {simple_mape:<12.2f} {simple_mae:<12.2f} {simple_theil:<12.4f}")
print(f"{'ARIMAX(0,1,1) with intervention':<30} {simple_arimax_rmse:<12.2f} {simple_arimax_mape:<12.2f} {simple_arimax_mae:<12.2f} {simple_arimax_theil:<12.4f}")

# Identify best model
models_accuracy = [
    ('ARIMA(3,1,3)', arima_rmse, arima_mape, arima_mae, arima_theil),
    ('ARIMAX(3,1,3) with intervention', arimax_rmse, arimax_mape, arimax_mae, arimax_theil),
    ('ARIMA(0,1,1)', simple_rmse, simple_mape, simple_mae, simple_theil),
    ('ARIMAX(0,1,1) with intervention', simple_arimax_rmse, simple_arimax_mape, simple_arimax_mae, simple_arimax_theil)
]

# Find best model by RMSE
best_rmse_model = min(models_accuracy, key=lambda x: x[1])
print(f"\nBest model by RMSE: {best_rmse_model[0]} with RMSE = {best_rmse_model[1]:.2f}")

# Find best model by MAPE
best_mape_model = min(models_accuracy, key=lambda x: x[2])
print(f"Best model by MAPE: {best_mape_model[0]} with MAPE = {best_mape_model[2]:.2f}%")

# Create a visual accuracy comparison table
plt.figure(figsize=(12, 6))
plt.text(0.5, 0.95, "Forecast Accuracy Comparison",
         ha='center', va='center', fontsize=16, fontweight='bold')

metrics_data = []
for model, rmse, mape, mae, theil in models_accuracy:
    metrics_data.append([model, f"{rmse:.2f}", f"{mape:.2f}%", f"{mae:.2f}", f"{theil:.4f}"])

metrics_headers = ["Model", "RMSE", "MAPE", "MAE", "Theil's U"]
metrics_table = plt.table(cellText=metrics_data, colLabels=metrics_headers,
                         loc='center', cellLoc='center', colColours=['#f2f2f2']*5)
metrics_table.auto_set_font_size(False)
metrics_table.set_fontsize(12)
metrics_table.scale(1.2, 1.5)

# Highlight best values
best_rmse_idx = min(range(len(models_accuracy)), key=lambda i: models_accuracy[i][1])
best_mape_idx = min(range(len(models_accuracy)), key=lambda i: models_accuracy[i][2])
best_mae_idx = min(range(len(models_accuracy)), key=lambda i: models_accuracy[i][3])
best_theil_idx = min(range(len(models_accuracy)), key=lambda i: models_accuracy[i][4])

metrics_table[(best_rmse_idx+1, 1)].set_facecolor('#CCFFCC')
metrics_table[(best_mape_idx+1, 2)].set_facecolor('#CCFFCC')
metrics_table[(best_mae_idx+1, 3)].set_facecolor('#CCFFCC')
metrics_table[(best_theil_idx+1, 4)].set_facecolor('#CCFFCC')

plt.axis('off')
plt.tight_layout()
plt.savefig(plots_dir + 'accuracy_metrics_comparison.png')
plt.close()

print(f"\nAccuracy metrics comparison table saved to '{plots_dir}accuracy_metrics_comparison.png'")

# Focus on COVID period specifically for error analysis
if any(test.index >= '2020-01-01'):
    print("\n--- COVID Period Specific Analysis ---")
    covid_test_period = test.index >= '2020-01-01'
    if any(covid_test_period):
        covid_test = test[covid_test_period]

        # Filter predictions for COVID period
        covid_arima_pred = arima_predictions[covid_test_period]
        covid_arimax_pred = arimax_predictions[covid_test_period]
        covid_simple_pred = simple_predictions[covid_test_period]
        covid_simple_arimax_pred = simple_arimax_predictions[covid_test_period]

        # Calculate COVID-specific MAPE
        covid_arima_mape = mean_absolute_percentage_error(covid_test['GDP'], covid_arima_pred)
        covid_arimax_mape = mean_absolute_percentage_error(covid_test['GDP'], covid_arimax_pred)
        covid_simple_mape = mean_absolute_percentage_error(covid_test['GDP'], covid_simple_pred)
        covid_simple_arimax_mape = mean_absolute_percentage_error(covid_test['GDP'], covid_simple_arimax_pred)

        print(f"\nCOVID Period (2020 onwards) MAPE:")
        print(f"{'Model':<30} {'MAPE (%)':<12}")
        print("-" * 42)
        print(f"{'ARIMA(3,1,3)':<30} {covid_arima_mape:<12.2f}")
        print(f"{'ARIMAX(3,1,3) with intervention':<30} {covid_arimax_mape:<12.2f}")
        print(f"{'ARIMA(0,1,1)':<30} {covid_simple_mape:<12.2f}")
        print(f"{'ARIMAX(0,1,1) with intervention':<30} {covid_simple_arimax_mape:<12.2f}")

        # Plot COVID period specific forecasts
        plt.figure(figsize=(14, 8))

        # Plot actual values
        plt.plot(covid_test.index, covid_test['GDP'], color='blue', label='Actual GDP', linewidth=2)

        # Plot forecasts
        plt.plot(covid_test.index, covid_arima_pred, color='red',
                 label='ARIMA(3,1,3)', linestyle='--', linewidth=1.5)
        plt.plot(covid_test.index, covid_arimax_pred, color='green',
                 label='ARIMAX(3,1,3) with intervention', linestyle='-', linewidth=1.5)
        plt.plot(covid_test.index, covid_simple_pred, color='purple',
                 label='ARIMA(0,1,1)', linestyle='--', linewidth=1.5)
        plt.plot(covid_test.index, covid_simple_arimax_pred, color='orange',
                 label='ARIMAX(0,1,1) with intervention', linestyle='-', linewidth=1.5)

        # Add COVID period shading
        plt.axvspan(pd.Timestamp('2020-01-01'), pd.Timestamp('2020-06-30'),
                    color='red', alpha=0.1, label='COVID Shock')
        plt.axvspan(pd.Timestamp('2020-07-01'), pd.Timestamp('2020-12-31'),
                    color='green', alpha=0.1, label='COVID Recovery')
        plt.axvspan(pd.Timestamp('2021-01-01'), covid_test.index.max(),
                    color='purple', alpha=0.1, label='Post-COVID')

        plt.title('COVID Period GDP Forecast Comparison', fontsize=16)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('GDP (Billions USD)', fontsize=14)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(plots_dir + 'covid_period_forecast.png')
        plt.close()

        print(f"\nCOVID period forecast comparison plot saved to '{plots_dir}covid_period_forecast.png'")

# Create summary
summary = f"""# ARIMA Intervention Analysis of US GDP Data

## Intervention Analysis Overview
In this extension of the ARIMA analysis, we added COVID-19 intervention dummy variables to explicitly model the structural break:

- **COVID Shock**: Q1-Q2 2020 (sharp decline)
- **COVID Recovery**: Q3-Q4 2020 (rapid rebound)
- **Post-COVID**: 2021 onwards (different growth trajectory)

## Model Comparison
We compared the following models:
- ARIMA(3,1,3) - original complex model without intervention
- ARIMAX(3,1,3) - complex model with COVID intervention
- ARIMA(0,1,1) - original simple model without intervention
- ARIMAX(0,1,1) - simple model with COVID intervention

## Key Findings
1. Best model: {best_mape_model[0]} with MAPE = {best_mape_model[2]:.2f}%
2. The intervention coefficients were {'statistically significant' if any(p_value < 0.05 for p_value in [float(row[4]) if row[4] != 'nan' else 1.0 for row in coef_data if ('covid' in row[0] or 'post_covid' in row[0])]) else 'not statistically significant'}
3. {'The intervention models improved forecast accuracy during the COVID period.' if (covid_arimax_mape < covid_arima_mape or covid_simple_arimax_mape < covid_simple_mape) else 'The intervention did not significantly improve forecast accuracy.'}

## Improvement from Original ARIMA
- Original ARIMA(3,1,3): MAPE = {arima_mape:.2f}%
- Best model: MAPE = {best_mape_model[2]:.2f}%
- Improvement: {abs(arima_mape - best_mape_model[2]):.2f}%

## Conclusion
{'The intervention analysis significantly improved the model by explicitly accounting for the COVID-19 structural break.' if (best_mape_model[0].find('intervention') > 0) else 'The intervention analysis did not significantly improve upon the original model.'}
"""

with open(plots_dir + "intervention_analysis_summary.md", "w") as f:
    f.write(summary)

print(f"\nIntervention analysis summary saved to '{plots_dir}intervention_analysis_summary.md'")
print("\nIntervention analysis completed successfully!")