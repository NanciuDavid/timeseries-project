#!/usr/bin/env python3
# SARIMA Analysis for Real Personal Consumption Expenditures (PCE) Time Series

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('ggplot')
sns.set_palette("deep")
plt.rcParams['figure.figsize'] = [14, 8] # Adjusted for potentially more subplots
plt.rcParams['figure.dpi'] = 100

# Define plots directory
plots_dir = "plots/sarima/personalConsumptionExpenditure_detailed/"
os.makedirs(plots_dir, exist_ok=True)

# Load the data
print("Loading and preparing the PCE data...")
data_path = "datasets/personalConsumptionExpenditure/RealPersonalConsumptionExpenditures.csv"
df = pd.read_csv(data_path)

# Convert the observation_date column to datetime
df['observation_date'] = pd.to_datetime(df['observation_date'], format='%Y-%m-%d')

# Set observation_date as the index
df.set_index('observation_date', inplace=True)

# Rename VALUE column to 'PCEC96' for clarity if it's not already
if 'VALUE' in df.columns and 'PCEC96' not in df.columns:
    df.rename(columns={'VALUE': 'PCEC96'}, inplace=True)
elif 'PCEC96' not in df.columns and len(df.columns) == 1:
    df.rename(columns={df.columns[0]: 'PCEC96'}, inplace=True)


# Basic data inspection
print(f"Dataset shape: {df.shape}")
print("\nFirst few rows:")
print(df.head())
print("\nLast few rows:")
print(df.tail())

# Check for missing values
missing_values = df.isnull().sum()
print(f"\nMissing values:\n{missing_values}")

# Basic statistics
print("\nBasic statistics:")
print(df.describe())

# Create a figure for multiple plots - similar to Figure 1 in arima_analysis.py
plt.figure(figsize=(15, 18)) # Adjusted height for more plots

# Original time series
plt.subplot(411) # Changed to 4 rows
plt.plot(df.index, df['PCEC96'], color='blue', linewidth=1.5)
plt.title('(a) Personal Consumption Expenditures (PCEC96) Time Series', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('PCE Value', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# Log transformation
plt.subplot(412) # Changed to 4 rows
plt.plot(df.index, np.log(df['PCEC96']), color='green', linewidth=1.5)
plt.title('(b) Log Transformation of PCE', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Log(PCE)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# First difference of original series
df['first_diff'] = df['PCEC96'].diff()
plt.subplot(413) # Changed to 4 rows
plt.plot(df.index[1:], df['first_diff'][1:], color='red', linewidth=1)
plt.title('(c) First Difference of PCE', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('PCE Difference', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# Seasonal difference (s=12 for monthly data) of original series
df['seasonal_diff_12'] = df['PCEC96'].diff(12)
plt.subplot(414) # Changed to 4 rows
plt.plot(df.index[12:], df['seasonal_diff_12'][12:], color='purple', linewidth=1)
plt.title('(d) Seasonal Difference (s=12) of PCE', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('PCE Seasonal Difference', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout(pad=2.0)
plt.savefig(plots_dir + 'figure1_pce_transformations.png')
plt.close()

print(f"\nTransformations plot saved to '{plots_dir}figure1_pce_transformations.png'")

# Function to perform Augmented Dickey-Fuller test
def adf_test(series, title=''):
    """
    Perform Augmented Dickey-Fuller test for stationarity
    """
    print(f"\nAugmented Dickey-Fuller Test for {title}")
    result = adfuller(series.dropna()) # Ensure no NaNs are passed

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

# Stationarity Tests
print("\n--- Stationarity Tests for PCE Data ---")

# Test original series
print("\nTesting original PCE series:")
is_stationary_original = adf_test(df['PCEC96'], title='Original PCE Series')
adf_original_pvalue = adfuller(df['PCEC96'].dropna())[1]

# Test first differenced series
df_first_diff = df['PCEC96'].diff().dropna()
print("\nTesting first differenced PCE series:")
is_stationary_first_diff = adf_test(df_first_diff, title='First Differenced PCE')
adf_first_diff_pvalue = adfuller(df_first_diff.dropna())[1]

# Test seasonally differenced series (s=12)
df_seasonal_diff = df['PCEC96'].diff(12).dropna()
print("\nTesting seasonally differenced (s=12) PCE series:")
is_stationary_seasonal_diff = adf_test(df_seasonal_diff, title='Seasonally Differenced (s=12) PCE')
adf_seasonal_diff_pvalue = adfuller(df_seasonal_diff.dropna())[1]

# Test first then seasonally differenced series
df_combined_diff = df['PCEC96'].diff().diff(12).dropna()
print("\nTesting first then seasonally differenced (d=1, D=1, s=12) PCE series:")
is_stationary_combined_diff = adf_test(df_combined_diff, title='First + Seasonally Differenced PCE')
adf_combined_diff_pvalue = adfuller(df_combined_diff.dropna())[1]

# Determine d and D based on tests
d_order = 0
D_order = 0
stationary_series_for_acf_pacf = df['PCEC96'] # Default to original

if not is_stationary_original:
    if is_stationary_first_diff:
        print("\nSeries is stationary after 1st non-seasonal differencing (d=1).")
        d_order = 1
        stationary_series_for_acf_pacf = df_first_diff
        # Check if seasonal differencing is still needed on this d=1 series
        temp_seasonal_on_first_diff = df_first_diff.diff(12).dropna()
        if adf_test(temp_seasonal_on_first_diff, title='Seasonally diff (D=1) on First Diff (d=1) PCE'):
            print("First differenced series becomes stationary with additional seasonal differencing (D=1).")
            D_order = 1
            stationary_series_for_acf_pacf = temp_seasonal_on_first_diff
    elif is_stationary_seasonal_diff:
        print("\nSeries is stationary after 1st seasonal differencing (D=1, s=12).")
        D_order = 1
        stationary_series_for_acf_pacf = df_seasonal_diff
        # Check if non-seasonal differencing is still needed on this D=1 series
        temp_first_on_seasonal_diff = df_seasonal_diff.diff().dropna()
        if adf_test(temp_first_on_seasonal_diff, title='First diff (d=1) on Seasonally Diff (D=1) PCE'):
            print("Seasonally differenced series becomes stationary with additional first differencing (d=1).")
            d_order = 1
            stationary_series_for_acf_pacf = temp_first_on_seasonal_diff
    elif is_stationary_combined_diff:
        print("\nSeries is stationary after 1st non-seasonal and 1st seasonal differencing (d=1, D=1, s=12).")
        d_order = 1
        D_order = 1
        stationary_series_for_acf_pacf = df_combined_diff
    else:
        print("\nWarning: Series may still be non-stationary. Defaulting to d=1, D=1 for SARIMA.")
        d_order = 1
        D_order = 1
        stationary_series_for_acf_pacf = df_combined_diff # Use this for ACF/PACF
elif not is_stationary_seasonal_diff: # Original is stationary, but check if seasonal component exists
    print("\nOriginal series is stationary, but checking for lingering seasonality for D.")
    # Even if original is stationary, seasonal differencing might be needed for SARIMA
    # Check if seasonally differencing the original makes it "more" stationary or reveals seasonal patterns
    # This is a heuristic for SARIMA's D. If seasonal_diff is stationary, D=1 might be good.
    if adf_test(df['PCEC96'].diff(12).dropna(), title='Seasonally Differenced Original (for D decision)'):
         D_order = 1 # Suggests a seasonal component that differencing handles
         print("Setting D=1 as seasonal differencing on original stationary series is also stationary.")
         # For ACF/PACF, if original is stationary, use that or seasonally differenced original
         # If d=0, D=1, then use seasonally differenced original data for ACF/PACF
         if d_order == 0 and D_order == 1:
             stationary_series_for_acf_pacf = df['PCEC96'].diff(12).dropna()

print(f"\nSelected differencing orders for SARIMA: d={d_order}, D={D_order}, s=12")

# Create a visual table for unit root test results
fig, ax = plt.figure(figsize=(12, 5)), plt.axes(frameon=False)
ax.set_title('Unit Root Test Results for PCE Data', fontsize=14, pad=20)
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)

table_data = [
    ['Series', 'ADF p-value', 'Conclusion'],
    ['Original PCE', f"{adf_original_pvalue:.4f}", 'Stationary' if is_stationary_original else 'Non-stationary'],
    ['First Difference', f"{adf_first_diff_pvalue:.4f}", 'Stationary' if is_stationary_first_diff else 'Non-stationary'],
    ['Seasonal Difference (s=12)', f"{adf_seasonal_diff_pvalue:.4f}", 'Stationary' if is_stationary_seasonal_diff else 'Non-stationary'],
    ['First + Seasonal Diff (d=1,D=1,s=12)', f"{adf_combined_diff_pvalue:.8f}", 'Stationary' if is_stationary_combined_diff else 'Non-stationary']
]

# Create the table
table = ax.table(cellText=table_data, loc='center', cellLoc='center',
                 colLabels=['Series Tested', 'ADF Test p-value', 'Stationarity Conclusion'],
                 colColours=['#f2f2f2']*3)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.1, 1.3)

plt.savefig(plots_dir + 'unit_root_test_pce_table.png', bbox_inches='tight')
plt.close()

print(f"\nUnit root test table saved to '{plots_dir}unit_root_test_pce_table.png'")


# Plot ACF and PACF for the appropriately differenced series
plt.figure(figsize=(15, 10))

plt.subplot(211)
plot_acf(stationary_series_for_acf_pacf, ax=plt.gca(), lags=48, alpha=0.05) # Increased lags for monthly data
plt.title(f'(a) Autocorrelation Function (ACF) for PCE (d={d_order}, D={D_order}, s=12)', fontsize=14)
plt.xlabel('Lag', fontsize=12)
plt.ylabel('Correlation', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

plt.subplot(212)
plot_pacf(stationary_series_for_acf_pacf, ax=plt.gca(), lags=48, alpha=0.05) # Increased lags for monthly data
plt.title(f'(b) Partial Autocorrelation Function (PACF) for PCE (d={d_order}, D={D_order}, s=12)', fontsize=14)
plt.xlabel('Lag', fontsize=12)
plt.ylabel('Partial Correlation', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout(pad=2.0)
plt.savefig(plots_dir + 'figure2_pce_acf_pacf.png')
plt.close()

print(f"\nACF and PACF plots saved to '{plots_dir}figure2_pce_acf_pacf.png'")
print(f"Based on the stationarity tests, the differencing orders are: d={d_order}, D={D_order}, s=12")
print("Examine the ACF and PACF plots to determine candidate AR (p), MA (q), SAR (P), SMA (Q) orders.")

# --- Model Identification and Estimation ---
print("\n--- SARIMA Model Identification and Estimation for PCE ---")

# Split data into training and testing sets (80% training, 20% testing)
# We use the original series for training, SARIMAX handles differencing via d and D orders
original_series = df['PCEC96'].dropna() # Ensure no NaNs in the original series used for modeling
train_size = int(len(original_series) * 0.8)
train, test = original_series[:train_size], original_series[train_size:]

print(f"Training set size: {len(train)}")
print(f"Testing set size: {len(test)}")

# Define a range of p, q, P, Q values to try
# Based on typical ACF/PACF patterns, keeping these small initially
p_values = range(0, 3) # Non-seasonal AR order
d_val = d_order          # Determined non-seasonal differencing
q_values = range(0, 3) # Non-seasonal MA order

P_values = range(0, 2) # Seasonal AR order
D_val = D_order          # Determined seasonal differencing
Q_values = range(0, 2) # Seasonal MA order
s_val = 12               # Seasonality period

# Store results
best_aic = float("inf")
best_bic = float("inf")
best_model_aic = None
best_model_bic = None
best_order_aic = None
best_order_bic = None

# Create a DataFrame to store model comparison results
model_results_list = []

print("\nFitting different SARIMA models...")
for p in p_values:
    for q in q_values:
        for P_s in P_values:
            for Q_s in Q_values:
                # Skip if all non-seasonal and seasonal orders are zero (unless d or D is non-zero)
                if p == 0 and q == 0 and P_s == 0 and Q_s == 0 and d_val == 0 and D_val == 0:
                    continue
                try:
                    current_order = (p, d_val, q)
                    current_seasonal_order = (P_s, D_val, Q_s, s_val)

                    # Fit SARIMA model on the training data
                    # SARIMAX handles differencing internally using d and D
                    model = SARIMAX(train, 
                                    order=current_order, 
                                    seasonal_order=current_seasonal_order,
                                    enforce_stationarity=False, # Let the model try to find parameters
                                    enforce_invertibility=False,
                                    initialization='approximate_diffuse') # Helps with some datasets
                    model_fit = model.fit(disp=False) # disp=False to suppress convergence messages

                    aic = model_fit.aic
                    bic = model_fit.bic
                    log_likelihood = model_fit.llf

                    model_results_list.append({
                        'p': p, 'd': d_val, 'q': q, 
                        'P': P_s, 'D': D_val, 'Q': Q_s, 's': s_val,
                        'AIC': aic, 'BIC': bic, 'LogLikelihood': log_likelihood
                    })

                    if aic < best_aic:
                        best_aic = aic
                        best_model_aic = model_fit
                        best_order_aic = current_order
                        best_seasonal_order_aic = current_seasonal_order

                    if bic < best_bic:
                        best_bic = bic
                        best_model_bic = model_fit # Storing the model object itself
                        best_order_bic = current_order
                        best_seasonal_order_bic = current_seasonal_order

                    print(f"SARIMA({p},{d_val},{q})x({P_s},{D_val},{Q_s},{s_val}) - AIC: {aic:.2f}, BIC: {bic:.2f}")

                except Exception as e:
                    print(f"Error fitting SARIMA({p},{d_val},{q})x({P_s},{D_val},{Q_s},{s_val}): {e}")
                    continue

model_results = pd.DataFrame(model_results_list)

if model_results.empty:
    print("\nNo SARIMA models were successfully fitted. Exiting.")
    # Potentially add exit() or raise error if this is critical
else:
    # Sort models by AIC
    model_results = model_results.sort_values('AIC')
    print("\nTop 10 Model comparison by AIC:")
    print(model_results.head(10).to_string())

    # Print best model information
    if best_model_aic:
        print(f"\nBest model by AIC: SARIMA{best_order_aic}x{best_seasonal_order_aic} with AIC={best_aic:.2f}")
        print("\n--- AIC Best Model Summary ---")
        print(best_model_aic.summary())
        with open(plots_dir + 'sarima_pce_model_summary_aic.txt', 'w') as f:
            f.write(str(best_model_aic.summary()))
        print(f"\nAIC Best Model summary saved to '{plots_dir}sarima_pce_model_summary_aic.txt'")
    else:
        print("\nNo best model found based on AIC.")

    if best_model_bic:
        print(f"\nBest model by BIC: SARIMA{best_order_bic}x{best_seasonal_order_bic} with BIC={best_bic:.2f}")
        # Fit the best BIC model again if it was overwritten by a later AIC model, or just use its summary
        # For simplicity, we are using the stored best_model_bic object
        print("\n--- BIC Best Model Summary ---")
        print(best_model_bic.summary())
        with open(plots_dir + 'sarima_pce_model_summary_bic.txt', 'w') as f:
            f.write(str(best_model_bic.summary()))
        print(f"\nBIC Best Model summary saved to '{plots_dir}sarima_pce_model_summary_bic.txt'")
    else:
        print("\nNo best model found based on BIC.")

    # Proceed with the best model by AIC for further diagnostics and forecasting
    final_best_model = best_model_aic
    final_best_order = best_order_aic
    final_best_seasonal_order = best_seasonal_order_aic

    if not final_best_model:
        print("\nNo model selected as best. Cannot proceed with diagnostics or forecast.")
    else:
        # Create a visual representation of the model equation - similar to arima_analysis.py
        plt.figure(figsize=(12, 7))
        plt.text(0.5, 0.8, 
                 f"Best SARIMA Model (AIC): SARIMA({final_best_order[0]},{final_best_order[1]},{final_best_order[2]})"
                 f"x({final_best_seasonal_order[0]},{final_best_seasonal_order[1]},{final_best_seasonal_order[2]},{final_best_seasonal_order[3]}) for PCE",
                 fontsize=14, ha='center')

        # Extract coefficients (handling potential absence of AR/MA/SAR/SMA parts)
        ar_params = final_best_model.arparams
        ma_params = final_best_model.maparams
        sar_params = final_best_model.seasonalarparams
        sma_params = final_best_model.seasonalmaparams

        # Build equation string components
        ar_eq_phi = "(1"
        for i, param in enumerate(ar_params, 1):
            ar_eq_phi += f" {'-' if param < 0 else '+'} {abs(param):.3f}B^{i}"
        ar_eq_phi += ")"
        if not ar_params.any(): ar_eq_phi = ""

        sar_eq_Phi = "(1"
        for i, param in enumerate(sar_params, 1):
            sar_eq_Phi += f" {'-' if param < 0 else '+'} {abs(param):.3f}B^{{{i*s_val}}}"
        sar_eq_Phi += ")"
        if not sar_params.any(): sar_eq_Phi = ""

        ma_eq_theta = "(1"
        for i, param in enumerate(ma_params, 1):
            ma_eq_theta += f" {'+' if param > 0 else '-'} {abs(param):.3f}B^{i}"
        ma_eq_theta += ")"
        if not ma_params.any(): ma_eq_theta = ""
        
        sma_eq_Theta = "(1"
        for i, param in enumerate(sma_params, 1):
            sma_eq_Theta += f" {'+' if param > 0 else '-'} {abs(param):.3f}B^{{{i*s_val}}}"
        sma_eq_Theta += ")"
        if not sma_params.any(): sma_eq_Theta = ""
        
        # Differencing part
        diff_part = ""
        if final_best_order[1] > 0:
            diff_part += f"(1-B)^{{{final_best_order[1]}}}"
        if final_best_seasonal_order[1] > 0:
            diff_part += f"(1-B^{{{s_val}}})^{{{final_best_seasonal_order[1]}}}"
        
        lhs = f"{ar_eq_phi} {sar_eq_Phi} {diff_part} PCE_t"
        rhs = f"{ma_eq_theta} {sma_eq_Theta} ε_t"
        full_eq = f"{lhs} = {rhs}"

        plt.text(0.5, 0.55, full_eq, fontsize=11, ha='center', family='monospace')
        plt.text(0.5, 0.3, "Where:", fontsize=12, ha='center')
        plt.text(0.5, 0.2, "B is the backshift operator", fontsize=12, ha='center')
        plt.text(0.5, 0.1, "ε_t is white noise", fontsize=12, ha='center')

        plt.axis('off')
        plt.tight_layout()
        plt.savefig(plots_dir + 'sarima_pce_model_equation.png')
        plt.close()
        print(f"\nModel equation visualization saved to '{plots_dir}sarima_pce_model_equation.png'")

        # Create a visual table of model comparisons
        plt.figure(figsize=(12, 7))
        plt.text(0.5, 0.95, "Top SARIMA Models for PCE (by AIC)", fontsize=16, ha='center')

        top_models_table = model_results.sort_values('AIC').head(5)
        table_data_comp = []
        headers_comp = ['Model', 'AIC', 'BIC', 'Log-Likelihood']
        
        for _, row in top_models_table.iterrows():
            model_str = f"SARIMA({int(row['p'])},{int(row['d'])},{int(row['q'])})x({int(row['P'])},{int(row['D'])},{int(row['Q'])},{int(row['s'])})"
            table_data_comp.append([model_str, f"{row['AIC']:.2f}", f"{row['BIC']:.2f}", f"{row['LogLikelihood']:.2f}"])

        table_comp = plt.table(cellText=table_data_comp, colLabels=headers_comp, 
                               loc='center', cellLoc='center', bbox=[0.05, 0.1, 0.9, 0.7])
        table_comp.auto_set_font_size(False)
        table_comp.set_fontsize(10)
        table_comp.scale(1.1, 1.3)
        
        # Highlight best AIC and BIC models in the table if they are in top 5
        best_aic_str = f"SARIMA({best_order_aic[0]},{best_order_aic[1]},{best_order_aic[2]})x({best_seasonal_order_aic[0]},{best_seasonal_order_aic[1]},{best_seasonal_order_aic[2]},{best_seasonal_order_aic[3]})"
        best_bic_str = f"SARIMA({best_order_bic[0]},{best_order_bic[1]},{best_order_bic[2]})x({best_seasonal_order_bic[0]},{best_seasonal_order_bic[1]},{best_seasonal_order_bic[2]},{best_seasonal_order_bic[3]})"

        for i_row, model_info_row in enumerate(table_data_comp):
            if model_info_row[0] == best_aic_str:
                table_comp[(i_row + 1, 1)].set_facecolor('#CCFFCC') # Highlight AIC column
            if model_info_row[0] == best_bic_str:
                 table_comp[(i_row + 1, 2)].set_facecolor('#CCFFCC') # Highlight BIC column

        plt.axis('off')
        plt.tight_layout()
        plt.savefig(plots_dir + 'sarima_pce_model_comparison_table.png')
        plt.close()
        print(f"\nModel comparison table saved to '{plots_dir}sarima_pce_model_comparison_table.png'")

# --- Model Diagnostic Checking ---
if not final_best_model:
    print("\nSkipping diagnostic checking as no best model was selected.")
else:
    print("\n--- Model Diagnostic Checking for Best AIC SARIMA Model (PCE) ---")
    residuals = final_best_model.resid

    # Plot built-in diagnostics from statsmodels
    final_best_model.plot_diagnostics(figsize=(15, 12))
    plt.suptitle(f'Diagnostic Plots for Best AIC SARIMA Model - PCE\nSARIMA{final_best_order}x{final_best_seasonal_order}', fontsize=16, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout to make space for suptitle
    plt.savefig(plots_dir + 'figure3_sarima_pce_diagnostics_auto.png')
    plt.close()
    print(f"\nStatsmodels diagnostic plots saved to '{plots_dir}figure3_sarima_pce_diagnostics_auto.png'")

    # Coefficient significance tests table (similar to arima_analysis.py Fig 4a)
    plt.figure(figsize=(14, 8))
    plt.subplot(211) # Using subplot to combine with Ljung-Box/Jarque-Bera table
    plt.text(0.5, 0.95, f"(a) Coefficient Significance for SARIMA{final_best_order}x{final_best_seasonal_order}", fontsize=14, ha='center', transform=plt.gca().transAxes)
    
    # Prepare data for coefficient table
    coef_df = pd.DataFrame({
        'coefficient': final_best_model.params,
        'std err': final_best_model.bse,
        'z': final_best_model.tvalues,
        'P>|z|': final_best_model.pvalues
    })
    # Exclude variance (sigma2) if present, as it's not a typical coefficient
    if 'sigma2' in coef_df.index:
        coef_df = coef_df.drop('sigma2')

    cell_text_coef = [[idx] + [f"{val:.4f}" for val in row] for idx, row in coef_df.iterrows()]
    column_labels_coef = ['Parameter'] + list(coef_df.columns)

    coef_table = plt.table(cellText=cell_text_coef, colLabels=column_labels_coef, 
                           loc='center', cellLoc='center', colColours=['#f2f2f2']*len(column_labels_coef))
    coef_table.auto_set_font_size(False)
    coef_table.set_fontsize(9)
    coef_table.scale(1.1, 1.3)
    plt.axis('off')

    # Ljung-Box and Jarque-Bera test results (similar to arima_analysis.py Fig 4b)
    plt.subplot(212)
    plt.text(0.5, 0.9, "(b) Residual Diagnostic Tests", fontsize=14, ha='center', transform=plt.gca().transAxes)
    lb_test_lags = [12, 24, 36] # Lags appropriate for monthly data (1, 2, 3 years)
    lb_results = acorr_ljungbox(residuals, lags=lb_test_lags, model_df=len(final_best_model.params) - int('sigma2' in final_best_model.params), return_df=True)
    jb_test = stats.jarque_bera(residuals)

    diag_table_data = [
        ['Test', 'Statistic', 'p-value', 'Conclusion']
    ]
    for lag in lb_test_lags:
        lb_stat = lb_results.loc[lag, 'lb_stat']
        lb_pvalue = lb_results.loc[lag, 'lb_pvalue']
        diag_table_data.append([
            f'Ljung-Box (lag {lag})', 
            f"{lb_stat:.4f}", 
            f"{lb_pvalue:.4f}", 
            'Residuals are independent' if lb_pvalue > 0.05 else 'Residuals show dependence'
        ])
    diag_table_data.append([
        'Jarque-Bera', 
        f"{jb_test[0]:.4f}", 
        f"{jb_test[1]:.4f}", 
        'Residuals are normally distributed' if jb_test[1] > 0.05 else 'Residuals not normal'
    ])

    diag_table = plt.table(cellText=diag_table_data[1:], colLabels=diag_table_data[0],
                           loc='center', cellLoc='center', colColours=['#f2f2f2']*4)
    diag_table.auto_set_font_size(False)
    diag_table.set_fontsize(9)
    diag_table.scale(1.1, 1.3)
    plt.axis('off')

    plt.tight_layout(pad=3.0)
    plt.savefig(plots_dir + 'figure4_sarima_pce_coefficient_diag_tests.png')
    plt.close()
    print(f"\nCoefficient and diagnostic tests table saved to '{plots_dir}figure4_sarima_pce_coefficient_diag_tests.png'")

    # Individual Residual Plots (Time series, ACF, Histogram, Q-Q)
    # Similar to arima_analysis.py Figures 3, 4, 5, 6 combined for the best model

    plt.figure(figsize=(15, 12))
    plt.suptitle(f'Residual Analysis for Best AIC SARIMA Model - PCE\nSARIMA{final_best_order}x{final_best_seasonal_order}', fontsize=16, y=1.02)

    # Residuals over time
    plt.subplot(221)
    plt.plot(residuals.index, residuals, color='blue')
    plt.title('(a) Residuals Over Time', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Residual Value', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # ACF of Residuals
    plt.subplot(222)
    plot_acf(residuals, ax=plt.gca(), lags=48, alpha=0.05)
    plt.title('(b) ACF of Residuals', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Histogram of Residuals
    plt.subplot(223)
    sns.histplot(residuals, kde=True, ax=plt.gca(), bins=30, color='blue')
    plt.title('(c) Histogram of Residuals with KDE', fontsize=14)
    plt.xlabel('Residual Value', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Q-Q Plot of Residuals
    plt.subplot(224)
    stats.probplot(residuals, dist="norm", plot=plt.gca())
    plt.title('(d) Q-Q Plot of Residuals', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(plots_dir + 'figure5_sarima_pce_residual_analysis.png')
    plt.close()
    print(f"\nCombined residual analysis plot saved to '{plots_dir}figure5_sarima_pce_residual_analysis.png'")

# --- Forecasting ---
if not final_best_model:
    print("\nSkipping forecasting as no best model was selected.")
else:
    print("\n--- Forecasting with Best AIC SARIMA Model (PCE) ---")

    # Figure: Fitted vs Actual Values Comparison (Training Data)
    plt.figure(figsize=(15, 7))
    fitted_values = final_best_model.fittedvalues # These are on the original scale of the training data
    plt.plot(train.index, train, label='Actual Training Data', color='blue', alpha=0.7)
    plt.plot(fitted_values.index, fitted_values, label=f'Fitted SARIMA{final_best_order}x{final_best_seasonal_order}', color='green')
    plt.title(f'(a) Fitted vs Actual Values (Training Data) - PCE', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('PCE Value', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(plots_dir + 'figure6_sarima_pce_fitted_vs_actual.png')
    plt.close()
    print(f"\nFitted vs actual values plot saved to '{plots_dir}figure6_sarima_pce_fitted_vs_actual.png'")

    # Figure: Point Forecast Comparison (Test Data)
    plt.figure(figsize=(15, 7))
    forecast_steps = len(test)
    forecast_object = final_best_model.get_forecast(steps=forecast_steps)
    forecast_mean = forecast_object.predicted_mean # Forecasts are on the original scale
    forecast_ci = forecast_object.conf_int() # Confidence intervals are also on the original scale

    plt.plot(train.index, train, label='Training Data', color='blue', alpha=0.3) # Optional: show training data
    plt.plot(test.index, test, label='Actual Test Data', color='blue')
    plt.plot(forecast_mean.index, forecast_mean, label='SARIMA Forecast', color='red')
    plt.fill_between(forecast_ci.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], 
                     color='pink', alpha=0.4, label='95% Confidence Interval')
    plt.title(f'(b) PCE Forecast vs Actual (Test Data) - SARIMA{final_best_order}x{final_best_seasonal_order}', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('PCE Value', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Calculate and display metrics on the plot
    mse = mean_squared_error(test, forecast_mean)
    rmse = np.sqrt(mse)
    # Ensure test values are not zero for MAPE calculation
    test_values_for_mape = test.values
    forecast_mean_values_for_mape = forecast_mean.values
    non_zero_mask_mape = test_values_for_mape != 0
    mape = np.nan # Default to NaN
    if np.any(non_zero_mask_mape):
        mape = np.mean(np.abs((test_values_for_mape[non_zero_mask_mape] - forecast_mean_values_for_mape[non_zero_mask_mape]) / test_values_for_mape[non_zero_mask_mape])) * 100
    
    error_text = f"MSE: {mse:.2f}\nRMSE: {rmse:.2f}\nMAPE: {mape:.2f}%"
    plt.text(0.02, 0.95, "Forecast Accuracy:", transform=plt.gca().transAxes, fontsize=12, 
             verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.7))
    plt.text(0.02, 0.85, error_text, transform=plt.gca().transAxes, fontsize=10, 
             verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.7))

    plt.tight_layout()
    plt.savefig(plots_dir + 'figure7_sarima_pce_forecast_vs_actual.png')
    plt.close()
    print(f"\nForecast comparison plot saved to '{plots_dir}figure7_sarima_pce_forecast_vs_actual.png'")

    print("\nForecast Accuracy Metrics (on original scale):")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

    # Calculate Theil's U statistic
    # Naive forecast: last value of training set, or a simple seasonal naive model for seasonal data
    # For simplicity with SARIMA, using a non-seasonal naive forecast (last value of train for all test steps)
    naive_forecast_values = np.full(len(test), train.iloc[-1])
    mse_naive = mean_squared_error(test, naive_forecast_values)
    
    theil_u1 = np.nan
    if mse_naive > 0: # Avoid division by zero if naive forecast is perfect (unlikely)
        theil_u1 = rmse / (np.sqrt(mse_naive) + rmse) # Theil's U1 variant (ranges 0 to 1)
        # Alternative Theil's U (U2): RMSE(model) / RMSE(naive)
        # theil_u2 = rmse / np.sqrt(mse_naive)

    # Create Figure for Theil's U visualization
    plt.figure(figsize=(8, 6))
    plt.text(0.5, 0.8, "Theil's U Statistic (U1 variant)", fontsize=16, ha='center')
    plt.text(0.5, 0.6, f"U1 = {theil_u1:.4f}" if not np.isnan(theil_u1) else "U1 = NaN (Naive MSE likely zero or non-positive)", fontsize=14, ha='center')

    interpretation = "Interpretation (U1 variant):\nCloser to 0 is better. 0.5 indicates naive forecast accuracy."
    if not np.isnan(theil_u1):
        if theil_u1 < 0.5:
            interpretation += f"\n({theil_u1:.2f} < 0.5: SARIMA is better than naive guess)"
        elif theil_u1 == 0.5:
            interpretation += f"\n({theil_u1:.2f} = 0.5: SARIMA is similar to naive guess)"
        else: # theil_u1 > 0.5
            interpretation += f"\n({theil_u1:.2f} > 0.5: SARIMA may be worse than naive guess)"
    
    plt.text(0.5, 0.35, interpretation, fontsize=12, ha='center', va='center')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(plots_dir + 'figure8_sarima_pce_theil_u.png')
    plt.close()
    print(f"\nTheil's U statistic plot saved to '{plots_dir}figure8_sarima_pce_theil_u.png'")
    if not np.isnan(theil_u1):
        print(f"Theil's U1 value: {theil_u1:.4f}")
    else:
        print("Theil's U1 value: NaN")

print("\nSARIMA analysis for PCE complete!")


# --- End of script --- 