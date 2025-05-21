import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

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


main_data_set = pd.read_csv("airtraffic/air_traffic_merged.csv")

# Convert FLT_DATE to datetime and set as index
main_data_set['FLT_DATE'] = pd.to_datetime(main_data_set['FLT_DATE'], format='%d-%m-%y')
main_data_set = main_data_set.set_index('FLT_DATE')

# Aggregate by date, summing FLT_TOT_1
time_series_data = main_data_set.groupby('FLT_DATE')['FLT_TOT_1'].sum().to_frame()

# Apply non-seasonal differencing (d=1)
differenced_series = time_series_data['FLT_TOT_1'].diff(periods=1).dropna()

# Apply seasonal differencing (D=1, s=365)
differenced_series = differenced_series.diff(periods=365).dropna()

def main():
    
    # Plot the original time series data
    plt.figure(figsize=(12, 6))
    plt.plot(time_series_data)
    plt.title('Total Daily Flights Over Time (Original)')
    plt.xlabel('Date')
    plt.ylabel('Total Flights')
    plt.grid(True)
    plt.show()

    # Plot the differenced time series data
    plt.figure(figsize=(12, 6))
    plt.plot(differenced_series)
    plt.title('Total Daily Flights Over Time (Differenced)')
    plt.xlabel('Date')
    plt.ylabel('Total Flights Differenced')
    plt.grid(True)
    plt.show()

    # Perform Augmented Dickey-Fuller test on differenced series
    print("\nAugmented Dickey-Fuller Test on Differenced Series:")
    adf_test_diff = adfuller(differenced_series)
    print(f'ADF Statistic: {adf_test_diff[0]}')
    print(f'p-value: {adf_test_diff[1]}')
    print('Critical Values:')
    for key, value in adf_test_diff[4].items():
        print(f'   {key}: {value}')

    # print("\nOriginal DataFrame Head:")
    # print(main_data_set.head())

    # print("\nOriginal DataFrame Info:")
    # print(main_data_set.info())

    # print("\nTime Series Data Head:")
    # print(time_series_data.head())

    # print("\nTime Series Data Info:")
    # print(time_series_data.info())

main()




