import pandas as pd
from sktime.forecasting.theta import ThetaForecaster
# from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.statsforecast import StatsForecastAutoARIMA
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.utils.plotting import plot_series
import matplotlib.pyplot as plt
import mplcyberpunk
from sktime.forecasting.bats import BATS
import holidays
# from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.exp_smoothing import ExponentialSmoothing


import numpy as np



def create_forecast_for_target(df: pd.DataFrame, target_column: str, sp_id: str,Month_train: int  ,Year_train: int , predict_start_date: str, predict_end_date: str):
    """
    Function to predict a time series for a specific sp_id and target column.
    
    :param df: DataFrame with 'ds', 'sp_id', and target_column
    :param target_column: Column to forecast (e.g., 'utilization')
    :param sp_id: Specific sp_id to filter and forecast
    :param predict_start_date: Start date for prediction
    :param predict_end_date: End date for prediction
    """
    # Filter data for the specific sp_id
    print(Month_train)
    df_whole=df.copy()
    df = df[(df['year'] < Year_train ) | ((df['year'] == Year_train ) & (df['Month'] < Month_train))]
    print("=============",df.unique_id.unique())
    data = df[df['unique_id'] == sp_id].copy()
    if data.empty:
        print(f"No data found for sp_id '{sp_id}', skipping . . .")
        return
    print(len(df))
    # Ensure 'ds' is datetime
    data['ds'] = pd.to_datetime(data['ds'], errors='coerce')

    # Check for negative values
    if (data[target_column] < 0).any():
        print(f"Target '{target_column}' for sp_id '{sp_id}' contained negative values, skipping . . .")
        return

    # Replace 0 with 0.001
    if data[target_column].min() == 0:
        print(f"Target '{target_column}' for sp_id '{sp_id}' contained values <= 0, adjusting . . .")
        data[target_column] = data[target_column].replace(0, 0.001)

    # Set 'ds' as index and convert to PeriodIndex
    data = data.set_index('ds')
    df_whole = df_whole.set_index('ds')
    data.index = data.index.to_period('W-SUN')
    df_whole.index = df_whole.index.to_period('W-SUN')

    # Train/test split
    # train, test = temporal_train_test_split(data[target_column], train_size=1.0)
    train = data[target_column]
    print(f"Last training date: {train.index.max()}")

    # Dynamic sp logic based on data length (weekly data)
    data_length = len(train)
    print(f"Data length for sp_id {sp_id}: {data_length}")

    # Define possible seasonal periods (weekly data)
    possible_sp = [104,52, 26, 13, 4]  # Yearly, half-yearly, quarterly, monthly approximations
    sp = None
    forecaster = None

    # Select largest sp with at least 2 cycles
    for period in possible_sp:
        if data_length >= 2 * period:
            sp = period
            forecaster = ExponentialSmoothing(trend="additive", seasonal="additive", sp=sp)#ThetaForecaster(sp=sp)
            print(f"Using ThetaForecaster with sp={sp} for {data_length} observations")
            break  # Weekly data, annual seasonality
    # forecaster = NaiveForecaster(strategy="drift")
    forecaster.fit(train)

    # Create forecast horizon
    future_timeseries = pd.period_range(start=predict_start_date, end=predict_end_date, freq="W-SUN")
    forecast_horizon = ForecastingHorizon(future_timeseries, is_relative=False)

    # Generate predictions
    pred = forecaster.predict(forecast_horizon)
    # print(pred)
    # Convert indices to DatetimeIndex for plotting
    data.index = data.index.to_timestamp()
    df_whole.index = df_whole.index.to_timestamp()
    pred.index = pred.index.to_timestamp()


    predicted_dates = pred.index
    filtered_actual_data = df_whole[df_whole.index.isin(predicted_dates)]
    
    # Now merge actual and predicted data on the index (date)
    combined_df = filtered_actual_data[['utilization']].rename(columns={'utilization': 'actual'})  # Rename 'utilization' to 'actual'
    combined_df['predicted'] = pred.values
    # print("combined_df", combined_df)
    # Plot
    # fig, ax = plt.subplots()
    # plot_series(
    #     data[target_column],
    #     pred,
    #     title=f"Predicted Data for sp_id {sp_id} - {target_column.capitalize()}",
    #     x_label="Date",
    #     y_label=f"{target_column.capitalize()} Level",
    #     labels=["Truth", "Predicted"],
    #     colors=["C0", "yellow"],
    #     markers=[None, None],
    #     ax=ax
    # )
    # # Add holiday lines (optional, if holidays are still in df)
    # if 'holiday' in df.columns:
    #     holiday_dates = df[(df['unique_id'] == sp_id) & (df['holiday'] == 1)]['ds']
    #     for date in holiday_dates:
    #         ax.axvline(x=date, color='Green', linestyle='--', alpha=0.3)
    # # mplcyberpunk.make_lines_glow(ax=ax, n_glow_lines=20, diff_linewidth=1.01, alpha_line=0.4)
    # mplcyberpunk.make_lines_glow(ax=ax, n_glow_lines=20, diff_linewidth=1.01, alpha_line=0.4)
    # plt.show()

    # Use plot_series to create the plot
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    plot_series(
        df_whole[target_column],
        pred,
        title=f"Predicted Data for sp_id {sp_id} - {target_column.capitalize()}",
        x_label="Date",
        y_label=f"{target_column.capitalize()} Level",
        labels=["Truth", "Predicted"],
        colors=["C0", "yellow"],
        markers=[None, None],
        ax=ax
    )
    # Add neon glow effect
    mplcyberpunk.make_lines_glow(
        ax=fig.axes[0],
        n_glow_lines=20,
        diff_linewidth=1.01,
        alpha_line=0.4
    )

    # Optional: Add holiday lines (if 'holiday' column exists in df)
    # if 'holiday' in df.columns:
    #     holiday_dates = df[(df['unique_id'] == sp_id) & (df['holiday'] == 1)]['ds']
    #     for date in holiday_dates:
    #         ax.axvline(x=date, color='green', linestyle='--', alpha=0.3)

    # # Apply cyberpunk style
    # mplcyberpunk.make_lines_glow(ax=ax, n_glow_lines=20, diff_linewidth=1.01, alpha_line=0.4)

    plt.tight_layout()
    # print(data)
    return fig, combined_df


# df=df[['ds','unique_id','week', 'year', 'Month','weekday', 'quarter',
#        'dayofyear', 'sin_day', 'cos_day', 'holiday', 'season', 'is_weekend',
#        'major_holiday','utilization']].copy()

# df=df[df['unique_id']==30007]
# Get unique sp_ids
# df['unique_id']=df['unique_id'].astype(int)
# sp_ids = df['unique_id'].unique()
# print(f"Found sp_ids: {sp_ids}")

# Forecast for each sp_id and target column
# for sp_id in sp_ids:
#     new=create_forecast_for_target(
#         df=df,
#         target_column='utilization',
#         sp_id=sp_id,
#         predict_start_date="2024-11-01",
#         predict_end_date="2025-01-01")
#     print("dfskvdfv",new.y)



def create_city_forecast_for_target(df: pd.DataFrame, target_column: str, sp_id: str,Month_train: int  ,Year_train: int , predict_start_date: str, predict_end_date: str):
    """
    Function to predict a time series for a specific sp_id and target column.
    
    :param df: DataFrame with 'ds', 'sp_id', and target_column
    :param target_column: Column to forecast (e.g., 'utilization')
    :param sp_id: Specific sp_id to filter and forecast
    :param predict_start_date: Start date for prediction
    :param predict_end_date: End date for prediction
    """
    # Filter data for the specific sp_id
    print(Month_train)
    df_whole=df.copy()
    df = df[(df['year'] < Year_train ) | ((df['year'] == Year_train ) & (df['Month'] < Month_train))]
    print("=============",df.city.unique())
    data = df[df['city'] == sp_id].copy()
    if data.empty:
        print(f"No data found for sp_id '{sp_id}', skipping . . .")
        return
    print(len(df))
    # Ensure 'ds' is datetime
    data['ds'] = pd.to_datetime(data['ds'], errors='coerce')

    # Check for negative values
    if (data[target_column] < 0).any():
        print(f"Target '{target_column}' for sp_id '{sp_id}' contained negative values, skipping . . .")
        return

    # Replace 0 with 0.001
    if data[target_column].min() == 0:
        print(f"Target '{target_column}' for sp_id '{sp_id}' contained values <= 0, adjusting . . .")
        data[target_column] = data[target_column].replace(0, 0.001)

    # Set 'ds' as index and convert to PeriodIndex
    data = data.set_index('ds')
    df_whole = df_whole.set_index('ds')
    data.index = data.index.to_period('W-SUN')
    df_whole.index = df_whole.index.to_period('W-SUN')

    # Train/test split
    # train, test = temporal_train_test_split(data[target_column], train_size=1.0)
    train = data[target_column]
    print(f"Last training date: {train.index.max()}")

    # Dynamic sp logic based on data length (weekly data)
    data_length = len(train)
    print(f"Data length for sp_id {sp_id}: {data_length}")

    # Define possible seasonal periods (weekly data)
    possible_sp = [104,52, 26, 13, 4]  # Yearly, half-yearly, quarterly, monthly approximations
    sp = None
    forecaster = None

    # Select largest sp with at least 2 cycles
    for period in possible_sp:
        if data_length >= 2 * period:
            sp = period
            forecaster = ExponentialSmoothing(trend="additive", seasonal="additive", sp=sp)#ThetaForecaster(sp=sp)
            print(f"Using ThetaForecaster with sp={sp} for {data_length} observations")
            break  # Weekly data, annual seasonality
    # forecaster = NaiveForecaster(strategy="drift")
    forecaster.fit(train)

    # Create forecast horizon
    future_timeseries = pd.period_range(start=predict_start_date, end=predict_end_date, freq="W-SUN")
    forecast_horizon = ForecastingHorizon(future_timeseries, is_relative=False)

    # Generate predictions
    pred = forecaster.predict(forecast_horizon)
    # print(pred)
    # Convert indices to DatetimeIndex for plotting
    data.index = data.index.to_timestamp()
    df_whole.index = df_whole.index.to_timestamp()
    pred.index = pred.index.to_timestamp()


    predicted_dates = pred.index
    filtered_actual_data = df_whole[df_whole.index.isin(predicted_dates)]
    
    # Now merge actual and predicted data on the index (date)
    combined_df = filtered_actual_data[[target_column]].rename(columns={target_column: 'actual'})  # Rename 'utilization' to 'actual'
    combined_df['predicted'] = pred.values
    # print("combined_df", combined_df)
    # Plot
    # fig, ax = plt.subplots()
    # plot_series(
    #     data[target_column],
    #     pred,
    #     title=f"Predicted Data for sp_id {sp_id} - {target_column.capitalize()}",
    #     x_label="Date",
    #     y_label=f"{target_column.capitalize()} Level",
    #     labels=["Truth", "Predicted"],
    #     colors=["C0", "yellow"],
    #     markers=[None, None],
    #     ax=ax
    # )
    # # Add holiday lines (optional, if holidays are still in df)
    # if 'holiday' in df.columns:
    #     holiday_dates = df[(df['city'] == sp_id) & (df['holiday'] == 1)]['ds']
    #     for date in holiday_dates:
    #         ax.axvline(x=date, color='Green', linestyle='--', alpha=0.3)
    # # mplcyberpunk.make_lines_glow(ax=ax, n_glow_lines=20, diff_linewidth=1.01, alpha_line=0.4)
    # mplcyberpunk.make_lines_glow(ax=ax, n_glow_lines=20, diff_linewidth=1.01, alpha_line=0.4)
    # plt.show()

    # Use plot_series to create the plot
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    plot_series(
        df_whole[target_column],
        pred,
        title=f"Predicted Data for sp_id {sp_id} - {target_column.capitalize()}",
        x_label="Date",
        y_label=f"{target_column.capitalize()} Level",
        labels=["Truth", "Predicted"],
        colors=["C0", "yellow"],
        markers=[None, None],
        ax=ax
    )
    # Add neon glow effect
    mplcyberpunk.make_lines_glow(
        ax=fig.axes[0],
        n_glow_lines=20,
        diff_linewidth=1.01,
        alpha_line=0.4
    )

    # Optional: Add holiday lines (if 'holiday' column exists in df)
    # if 'holiday' in df.columns:
    #     holiday_dates = df[(df['city'] == sp_id) & (df['holiday'] == 1)]['ds']
    #     for date in holiday_dates:
    #         ax.axvline(x=date, color='green', linestyle='--', alpha=0.3)

    # # Apply cyberpunk style
    # mplcyberpunk.make_lines_glow(ax=ax, n_glow_lines=20, diff_linewidth=1.01, alpha_line=0.4)

    plt.tight_layout()
    # print(data)
    return fig, combined_df