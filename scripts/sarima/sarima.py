import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

#read the csv files
# df2016 = pd.read_csv("airtraffic/airport_traffic_2016.csv")
# df2017 = pd.read_csv("airtraffic/airport_traffic_2017.csv")
# df2018 = pd.read_csv("airtraffic/airport_traffic_2018.csv")
# df2019 = pd.read_csv("airtraffic/airport_traffic_2019.csv")
# df2020 = pd.read_csv("airtraffic/airport_traffic_2020.csv")
# df2021 = pd.read_csv("airtraffic/airport_traffic_2021.csv")
# df2022 = pd.read_csv("airtraffic/airport_traffic_2022.csv")
# df2023 = pd.read_csv("airtraffic/airport_traffic_2023.csv")
# df2024 = pd.read_csv("airtraffic/airport_traffic_2024.csv")


# df = pd.concat([df2016, df2017, df2018, df2019, df2020, df2021, df2022, df2023, df2024])

#save the merged dataframe to a csv file
# df.to_csv("airtraffic/air_traffic_merged.csv", index=False)


main_data_set = pd.read_csv("datasets/airtraffic-part2/air_traffic_merged.csv")

# Convert FLT_DATE to datetime and set as index
main_data_set['FLT_DATE'] = pd.to_datetime(main_data_set['FLT_DATE'], format='%d-%m-%y')
main_data_set = main_data_set.set_index('FLT_DATE')

# Aggregate by date, summing FLT_TOT_1
time_series_data = main_data_set.groupby('FLT_DATE')['FLT_TOT_1'].sum().to_frame()

def adf_test(series, title=''):
    print(f"\nAugmented Dickey-Fuller Test for {title}")
    result = adfuller(series.dropna())
    labels = ['ADF Test Statistic', 'p-value', '# Lags Used', '# Observations Used']
    for value, label in zip(result, labels):
        print(f'{label} : {value}')
    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis (stationary)")
        return True
    else:
        print("Weak evidence against the null hypothesis (non-stationary)")
        return False

def main():
    # Plot original time series
    plt.figure(figsize=(12, 6))
    plt.plot(time_series_data)
    plt.title('Total Daily Flights Over Time')
    plt.xlabel('Date')
    plt.ylabel('Total Flights')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/sarima/airtraffic_time_series.png')
    plt.close()

    # ADF test and differencing
    is_stationary = adf_test(time_series_data['FLT_TOT_1'], title='Original Series')
    d_order = 0
    series = time_series_data['FLT_TOT_1']
    if not is_stationary:
        # First difference
        series_diff = series.diff().dropna()
        plt.figure(figsize=(12, 6))
        plt.plot(series_diff)
        plt.title('First Difference of Total Daily Flights')
        plt.xlabel('Date')
        plt.ylabel('Difference')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('plots/airtraffic_first_diff.png')
        plt.close()
        is_stationary = adf_test(series_diff, title='First Difference')
        if is_stationary:
            d_order = 1
            series = series_diff
        else:
            # Second difference
            series_diff2 = series_diff.diff().dropna()
            plt.figure(figsize=(12, 6))
            plt.plot(series_diff2)
            plt.title('Second Difference of Total Daily Flights')
            plt.xlabel('Date')
            plt.ylabel('Second Difference')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('plots/airtraffic_second_diff.png')
            plt.close()
            is_stationary = adf_test(series_diff2, title='Second Difference')
            if is_stationary:
                d_order = 2
                series = series_diff2
            else:
                print("Warning: Series may still be non-stationary after second differencing.")
                d_order = 2
                series = series_diff2
    print(f"\nUsing d={d_order} for SARIMA.")

    # ACF and PACF plots
    plt.figure(figsize=(12, 10))
    plt.subplot(211)
    plot_acf(series, ax=plt.gca(), lags=40)
    plt.title('ACF')
    plt.subplot(212)
    plot_pacf(series, ax=plt.gca(), lags=40)
    plt.title('PACF')
    plt.tight_layout()
    plt.savefig('plots/airtraffic_acf_pacf.png')
    plt.close()

    # Split into train/test
    train_size = int(len(series) * 0.8)
    train, test = series[:train_size], series[train_size:]

    # SARIMA grid search (simple, for demo)
    p = d = q = range(0, 2)
    P = D = Q = range(0, 2)
    s = 7  # Weekly seasonality (adjust if needed)
    best_aic = float('inf')
    best_order = None
    best_seasonal_order = None
    best_model = None
    print("\nGrid search for SARIMA parameters...")
    for i in p:
        for j in d:
            for k in q:
                for si in P:
                    for sj in D:
                        for sk in Q:
                            try:
                                model = SARIMAX(train, order=(i, d_order, k), seasonal_order=(si, 1, sk, s), enforce_stationarity=False, enforce_invertibility=False)
                                results = model.fit(disp=False)
                                if results.aic < best_aic:
                                    best_aic = results.aic
                                    best_order = (i, d_order, k)
                                    best_seasonal_order = (si, 1, sk, s)
                                    best_model = results
                                print(f"SARIMA{(i, d_order, k)}x{(si, 1, sk, s)} - AIC:{results.aic:.2f}")
                            except Exception as e:
                                continue
    print(f"\nBest SARIMA order: {best_order} Seasonal: {best_seasonal_order} AIC: {best_aic:.2f}")
    print(best_model.summary())
    with open('plots/airtraffic_sarima_model_summary.txt', 'w') as f:
        f.write(str(best_model.summary()))

    # Diagnostics
    best_model.plot_diagnostics(figsize=(12, 8))
    plt.tight_layout()
    plt.savefig('plots/airtraffic_sarima_diagnostics.png')
    plt.close()

    # Forecast
    forecast_steps = len(test)
    pred = best_model.get_forecast(steps=forecast_steps)
    pred_mean = pred.predicted_mean
    pred_ci = pred.conf_int()

    # Plot forecast vs actual
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
    plt.savefig('plots/airtraffic_sarima_forecast.png')
    plt.close()

    # Forecast accuracy
    mse = mean_squared_error(test, pred_mean)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((test.values - pred_mean.values) / test.values)) * 100
    print(f"\nForecast Accuracy:\nMSE: {mse:.2f}\nRMSE: {rmse:.2f}\nMAPE: {mape:.2f}%")
    with open('plots/airtraffic_sarima_forecast_accuracy.txt', 'w') as f:
        f.write(f"MSE: {mse:.2f}\nRMSE: {rmse:.2f}\nMAPE: {mape:.2f}%\n")

if __name__ == "__main__":
    main()




