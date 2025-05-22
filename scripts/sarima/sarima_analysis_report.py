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
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.style.use('ggplot')
sns.set_palette("deep")
plt.rcParams['figure.figsize'] = [12, 7]
plt.rcParams['figure.dpi'] = 100

plots_dir = "plots/sarima/"
os.makedirs(plots_dir, exist_ok=True)

# Load the data
data_path = "datasets/airtraffic-part2/air_traffic_merged.csv"
df = pd.read_csv(data_path)
df['FLT_DATE'] = pd.to_datetime(df['FLT_DATE'], format='%d-%m-%y')
df.set_index('FLT_DATE', inplace=True)
time_series_data = df.groupby('FLT_DATE')['FLT_TOT_1'].sum().to_frame()

y = time_series_data['FLT_TOT_1']

# 1. Data inspection and transformations
plt.figure(figsize=(15, 12))
plt.subplot(311)
plt.plot(y.index, y, color='blue', linewidth=1.5)
plt.title('(a) Air Traffic Time Series', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Total Flights', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

plt.subplot(312)
plt.plot(y.index, np.log(y+1), color='green', linewidth=1.5)
plt.title('(b) Log Transformation of Air Traffic', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Log(Total Flights)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

y_diff = y.diff()
plt.subplot(313)
plt.plot(y.index[1:], y_diff[1:], color='red', linewidth=1)
plt.title('(c) First Difference of Air Traffic', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Difference', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(plots_dir + 'figure1_transformations.png')
plt.close()

# 2. ADF tests and differencing
adf_results = []
def adf_test(series, label):
    result = adfuller(series.dropna())
    adf_results.append([label, result[1], 'Stationary' if result[1] <= 0.05 else 'Non-stationary'])
    return result[1] <= 0.05

is_stationary = adf_test(y, 'Original Series')
d_order = 0
D_order = 0
s = 7
series_to_use = y.copy()

if not is_stationary:
    y_first_diff = y.diff().dropna()
    is_stationary_diff = adf_test(y_first_diff, 'First Difference')
    if is_stationary_diff:
        d_order = 1
        series_to_use = y_first_diff
    else:
        y_seasonal_diff = y.diff(s).dropna()
        is_stationary_seasonal = adf_test(y_seasonal_diff, 'Seasonal Difference')
        if is_stationary_seasonal:
            D_order = 1
            series_to_use = y_seasonal_diff
        else:
            y_both_diff = y.diff().diff(s).dropna()
            is_stationary_both = adf_test(y_both_diff, 'First + Seasonal Difference')
            if is_stationary_both:
                d_order = 1
                D_order = 1
                series_to_use = y_both_diff
            else:
                d_order = 1
                D_order = 1
                series_to_use = y_both_diff

# 3. Unit root test table
adf_table = [['Test', 'p-value', 'Conclusion']] + [[row[0], f"{row[1]:.8f}", row[2]] for row in adf_results]
fig, ax = plt.figure(figsize=(8, 2)), plt.axes(frameon=False)
ax.set_title('Unit Root Test Results', fontsize=14, pad=20)
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
table = ax.table(cellText=adf_table, loc='center', cellLoc='center', colColours=['#f2f2f2']*3)
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.5)
plt.savefig(plots_dir + 'unit_root_test_table.png', bbox_inches='tight')
plt.close()

# 4. ACF and PACF plots
plt.figure(figsize=(15, 10))
plt.subplot(211)
plot_acf(series_to_use, ax=plt.gca(), lags=40, alpha=0.05)
plt.title('(a) Autocorrelation Function (ACF)', fontsize=14)
plt.xlabel('Lag', fontsize=12)
plt.ylabel('Correlation', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.subplot(212)
plot_pacf(series_to_use, ax=plt.gca(), lags=40, alpha=0.05)
plt.title('(b) Partial Autocorrelation Function (PACF)', fontsize=14)
plt.xlabel('Lag', fontsize=12)
plt.ylabel('Partial Correlation', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(plots_dir + 'figure2_acf_pacf.png')
plt.close()

# 5. Model Identification and Estimation
train_size = int(len(series_to_use) * 0.8)
train, test = series_to_use[:train_size], series_to_use[train_size:]
p_values = range(0, 3)
q_values = range(0, 3)
P_values = range(0, 2)
Q_values = range(0, 2)
best_aic = float('inf')
best_model = None
best_order = None
best_seasonal_order = None
model_results = []
print("\nFitting SARIMA models...")
for p in p_values:
    for q in q_values:
        for P in P_values:
            for Q in Q_values:
                try:
                    model = SARIMAX(train, order=(p, d_order, q), seasonal_order=(P, D_order, Q, s), enforce_stationarity=False, enforce_invertibility=False)
                    model_fit = model.fit(disp=False)
                    aic = model_fit.aic
                    bic = model_fit.bic
                    model_results.append([f"SARIMA({p},{d_order},{q})x({P},{D_order},{Q},{s})", aic, bic, model_fit.llf])
                    if aic < best_aic:
                        best_aic = aic
                        best_model = model_fit
                        best_order = (p, d_order, q)
                        best_seasonal_order = (P, D_order, Q, s)
                    print(f"SARIMA({p},{d_order},{q})x({P},{D_order},{Q},{s}) - AIC: {aic:.2f}, BIC: {bic:.2f}")
                except Exception as e:
                    continue
model_results_df = pd.DataFrame(model_results, columns=['Model', 'AIC', 'BIC', 'Log-Likelihood'])
model_results_df = model_results_df.sort_values('AIC').reset_index(drop=True)

# Save top 5 models table
top_models = model_results_df.head(5)
fig, ax = plt.figure(figsize=(10, 3)), plt.axes(frameon=False)
ax.set_title('Top 5 SARIMA Models by AIC', fontsize=14, pad=20)
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
table = ax.table(cellText=top_models.values, colLabels=top_models.columns, loc='center', cellLoc='center', colColours=['#f2f2f2']*4)
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.5)
plt.savefig(plots_dir + 'model_comparison_table.png', bbox_inches='tight')
plt.close()

# 6. Model summary and equation visualization
with open(plots_dir + 'sarima_model_summary.txt', 'w') as f:
    f.write(str(best_model.summary()))

plt.figure(figsize=(10, 6))
plt.text(0.5, 0.7, f"$SARIMA{best_order}x{best_seasonal_order}$ model:", fontsize=16, ha='center')
plt.axis('off')
plt.tight_layout()
plt.savefig(plots_dir + 'model_equation.png')
plt.close()

# 7. Coefficient significance table
params = best_model.params
bse = best_model.bse
zvalues = best_model.tvalues
pvalues = best_model.pvalues
coef_table = [['Parameter', 'Coef', 'Std Err', 'z-value', 'p-value']]
for i in range(len(params)):
    coef_table.append([params.index[i], f"{params[i]:.4f}", f"{bse[i]:.4f}", f"{zvalues[i]:.4f}", f"{pvalues[i]:.4f}"])
fig, ax = plt.figure(figsize=(10, 3)), plt.axes(frameon=False)
ax.set_title('Coefficient Significance', fontsize=14, pad=20)
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
table = ax.table(cellText=coef_table, loc='center', cellLoc='center', colColours=['#f2f2f2']*5)
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.5)
plt.savefig(plots_dir + 'coefficient_tests.png', bbox_inches='tight')
plt.close()

# 8. Model diagnostics
residuals = best_model.resid
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes[0, 0].plot(residuals.index, residuals, color='blue')
axes[0, 0].set_title('(a) Time Plot of Residuals', fontsize=14)
axes[0, 0].set_xlabel('Date', fontsize=12)
axes[0, 0].set_ylabel('Residual Value', fontsize=12)
axes[0, 0].grid(True, linestyle='--', alpha=0.7)
axes[0, 1].hist(residuals, bins=30, density=True, color='blue', alpha=0.7)
xmin, xmax = axes[0, 1].get_xlim()
x = np.linspace(xmin, xmax, 100)
axes[0, 1].plot(x, np.exp(-(x**2)/2)/(np.sqrt(2*np.pi)), 'r', linewidth=2)
axes[0, 1].set_title('(b) Histogram of Residuals with Normal Curve', fontsize=14)
axes[0, 1].set_xlabel('Residual Value', fontsize=12)
axes[0, 1].set_ylabel('Density', fontsize=12)
axes[0, 1].grid(True, linestyle='--', alpha=0.7)
plot_acf(residuals, ax=axes[1, 0], lags=40, alpha=0.05)
axes[1, 0].set_title('(c) ACF of Residuals', fontsize=14)
axes[1, 0].grid(True, linestyle='--', alpha=0.7)
stats.probplot(residuals, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('(d) Q-Q Plot of Residuals', fontsize=14)
axes[1, 1].grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(plots_dir + 'figure3_residual_diagnostics.png')
plt.close()

# 9. Ljung-Box and Jarque-Bera tests
diag_table_data = [
    ['Test', 'Statistic', 'p-value', 'Conclusion'],
]
lb_test = acorr_ljungbox(residuals, lags=10)
jb_test = stats.jarque_bera(residuals)
diag_table_data.append(['Ljung-Box (lag 10)', f"{lb_test.iloc[0, 0]:.4f}", f"{lb_test.iloc[0, 1]:.8f}", 'Reject H0' if lb_test.iloc[0, 1] < 0.05 else 'Fail to reject H0'])
diag_table_data.append(['Jarque-Bera', f"{jb_test[0]:.4f}", f"{jb_test[1]:.8f}", 'Reject H0' if jb_test[1] < 0.05 else 'Fail to reject H0'])
fig, ax = plt.figure(figsize=(10, 2)), plt.axes(frameon=False)
ax.set_title('Diagnostic Tests', fontsize=14, pad=20)
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
table = ax.table(cellText=diag_table_data[1:], colLabels=diag_table_data[0], loc='center', cellLoc='center', colColours=['#f2f2f2']*4)
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.5)
plt.savefig(plots_dir + 'diagnostic_tests.png', bbox_inches='tight')
plt.close()

# 10. Forecasting
forecast_steps = len(test)
forecast = best_model.get_forecast(steps=forecast_steps)
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()
plt.figure(figsize=(15, 7))
plt.plot(forecast_mean.index, forecast_mean, 'r-', label='Point Forecast')
plt.fill_between(forecast_mean.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='pink', alpha=0.3)
plt.plot(test.index, test, 'b-', label='Actual Values')
plt.title('Air Traffic Forecast (Differenced Series)', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Total Flights', fontsize=14)
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.7)
mse = mean_squared_error(test, forecast_mean)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((test.values - forecast_mean.values) / test.values)) * 100
error_text = f"MSE: {mse:.2f}\nRMSE: {rmse:.2f}\nMAPE: {mape:.2f}%"
plt.text(0.02, 0.96, "Error Metrics:", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
plt.text(0.02, 0.90, error_text, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
plt.tight_layout()
plt.savefig(plots_dir + 'figure8_point_forecast.png')
plt.close()

# 11. Theil's U statistic
if d_order > 0:
    # Naive forecast (previous value)
    naive_forecast = test.shift(1).fillna(method='bfill')
    naive_mse = mean_squared_error(test, naive_forecast)
    theil_u = np.sqrt(mse) / np.sqrt(naive_mse)
    plt.figure(figsize=(8, 6))
    plt.text(0.5, 0.8, "Theil's U Statistic", fontsize=16, ha='center')
    plt.text(0.5, 0.6, f"U = {theil_u:.4f}", fontsize=14, ha='center')
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

print("\nSARIMA analysis complete!") 