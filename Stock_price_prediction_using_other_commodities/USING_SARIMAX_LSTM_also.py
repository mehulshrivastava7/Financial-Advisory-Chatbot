import pandas as pd
from prophet import Prophet
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
import logging
import sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

#The SARIMAX and LSTM models codes were written by Ramanan and the ARMA and PROPHET codes were written by Mehul. We weren't able to integrate this part with UI before the presentation."
#Ramanan integrated this part with UI too after the presentation as it can be seen in the app.py file.

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
 
# --------------------------
# 1. Suppress Logging
# --------------------------
 
# Suppress yfinance logging
logging.getLogger('yfinance').setLevel(logging.WARNING)
 
# Suppress Prophet logging
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
 
# --------------------------
# 2. Prophet Model Functions
# --------------------------
 
def get_stock_data_prophet(ticker, period='2y'):
    """
    Fetches historical stock data for a given ticker for Prophet.
 
    Args:
        ticker (str): Stock ticker symbol.
        period (str): Data period (e.g., '2y' for two years).
 
    Returns:
        pd.DataFrame: DataFrame containing 'ds' and 'y' columns for Prophet.
    """
    stock = yf.download(ticker, period=period, progress=False)
    df = stock[['Close']].reset_index()
    # Remove timezone information from datetime column
    df['Date'] = df['Date'].dt.tz_localize(None)
    df.columns = ['ds', 'y']  # Prophet requires these column names
    return df
 
def train_and_predict_prophet(train_df, test_df, future_days=7):
    """
    Trains the Prophet model and makes predictions.
 
    Args:
        train_df (pd.DataFrame): Training DataFrame with 'ds' and 'y'.
        test_df (pd.DataFrame): Test DataFrame with 'ds' and 'y'.
        future_days (int): Number of days to forecast beyond the test set.
 
    Returns:
        tuple: (model, forecast, mse, future_predictions)
    """
    try:
        # Initialize Prophet model with the specified hyperparameters
        model = Prophet(
            growth='linear',
            changepoint_prior_scale=0.014900306553726704,
            seasonality_prior_scale=0.3836198463360881,
            seasonality_mode='additive',
            changepoint_range=0.8205879350577062,
            n_changepoints=41,
            daily_seasonality=True,
            weekly_seasonality=False,
            yearly_seasonality=False
        )
        
        # Fit the model on training data
        model.fit(train_df)
        
        # Create dataframe for the test period and future days
        future = model.make_future_dataframe(periods=len(test_df) + future_days)
        
        # Make predictions
        forecast = model.predict(future)
        
        # Extract predictions for the test set
        test_forecast = forecast.set_index('ds').loc[test_df['ds']]
        
        # Calculate MSE for the test set
        mse = mean_squared_error(test_df['y'], test_forecast['yhat'])
        
        # Forecast future days
        future_days_df = model.make_future_dataframe(periods=future_days)
        future_forecast = model.predict(future_days_df)
        
        future_predictions = future_forecast.tail(future_days)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        
        return model, forecast, mse, future_predictions
    except Exception as e:
        # Handle exceptions silently
        raise
 
def plot_predictions_prophet(train_df, test_df, forecast, mse, future_predictions, model):
    """
    Plots the Prophet model predictions and components.
 
    Args:
        train_df (pd.DataFrame): Training DataFrame.
        test_df (pd.DataFrame): Test DataFrame.
        forecast (pd.DataFrame): Forecast DataFrame from Prophet.
        mse (float): Mean Squared Error on the test set.
        future_predictions (pd.DataFrame): Future predictions DataFrame.
        model (Prophet): Trained Prophet model.
    """
    plt.figure(figsize=(12, 6))
    
    # Plot training data
    plt.plot(train_df['ds'], train_df['y'], 'b.', label='Training Data')
    
    # Plot test data
    plt.plot(test_df['ds'], test_df['y'], 'r.', label='Test Data')
    
    # Plot forecast
    plt.plot(forecast['ds'], forecast['yhat'], 'k-', label='Forecast')
    
    # Confidence intervals
    plt.fill_between(forecast['ds'],
                     forecast['yhat_lower'],
                     forecast['yhat_upper'],
                     color='gray',
                     alpha=0.2,
                     label='Confidence Interval')
    
    # Highlight test period
    plt.axvspan(test_df['ds'].min(), test_df['ds'].max(), color='yellow', alpha=0.1, label='Test Period')
    
    # Plot future predictions
    plt.plot(future_predictions['ds'], future_predictions['yhat'], 'g*', markersize=10, label='Future Predictions')
    
    plt.legend()
    plt.title('AAPL Stock Price Prediction with Prophet')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(True)
    plt.show()
    
    # Plot model components
    model.plot_components(forecast)
    plt.show()
    
    # Print MSE
    # Commented out as per user request
    # print(f"Mean Squared Error on Test Set: {mse:.2f}")
 
def get_future_prediction_metrics_prophet(future_predictions):
    """
    Formats future predictions into a DataFrame.
 
    Args:
        future_predictions (pd.DataFrame): Future predictions DataFrame.
 
    Returns:
        pd.DataFrame: Formatted future predictions.
    """
    metrics = {
        'date': future_predictions['ds'],
        'predicted_price': future_predictions['yhat'].round(2),
        'lower_bound': future_predictions['yhat_lower'].round(2),
        'upper_bound': future_predictions['yhat_upper'].round(2),
    }
    return pd.DataFrame(metrics)
 
# --------------------------
# 3. ARMA Model Functions
# --------------------------
 
def get_stock_data_arma(ticker, start_date='2022-11-06', end_date='2024-11-06'):
    """
    Fetches historical stock data for a given ticker for ARMA.
 
    Args:
        ticker (str): Stock ticker symbol.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
 
    Returns:
        pd.DataFrame: DataFrame containing 'Date' and 'Close' Price.
    """
    try:
        stock = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if stock.empty:
            raise ValueError(f"No data found for ticker {ticker}.")
        df = stock[['Close']].reset_index()
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        return df
    except Exception as e:
        # Handle exceptions silently
        return None
 
def determine_arma_order(series, max_lag=10, threshold=0.2):
    """
    Determines the optimal ARMA(p, q) order based on ACF and PACF of the series.
 
    Args:
        series (pd.Series): Time series data.
        max_lag (int): Maximum lag to consider for ACF and PACF (default 10).
        threshold (float): Threshold for significance in ACF/PACF values.
 
    Returns:
        tuple: (p, q) order for ARMA model.
    """
    from statsmodels.tsa.stattools import acf, pacf
 
    # Compute PACF and ACF
    pacf_vals = pacf(series, nlags=max_lag, method='ols')
    acf_vals = acf(series, nlags=max_lag)
 
    # Determine p (AR order) based on PACF cutoff
    p = 0
    for i in range(1, len(pacf_vals)):
        if abs(pacf_vals[i]) < threshold:
            p = i
            break
 
    # Determine q (MA order) based on ACF cutoff
    q = 0
    for i in range(1, len(acf_vals)):
        if abs(acf_vals[i]) < threshold:
            q = i
            break
 
    return (p, q)
 
def fit_arma_model(series, order):
    """
    Fits an ARMA model to the given time series.
 
    Args:
        series (pd.Series): Time series data.
        order (tuple): (p, q) order for ARMA model.
 
    Returns:
        ARIMAResults: Fitted ARMA model.
    """
    try:
        model = ARIMA(series, order=(order[0], 0, order[1]))
        model_fit = model.fit()
        return model_fit
    except Exception as e:
        # Handle exceptions silently
        return None
 
def predict_arma(model_fit, steps=7):
    """
    Predicts future prices using the fitted ARMA model.
 
    Args:
        model_fit (ARIMAResults): Fitted ARMA model.
        steps (int): Number of future steps to predict.
 
    Returns:
        pd.Series: Predicted prices.
    """
    try:
        forecast = model_fit.get_forecast(steps=steps)
        predicted_prices = forecast.predicted_mean
        return predicted_prices
    except Exception as e:
        # Handle exceptions silently
        return None
 
def evaluate_arma(df, ticker):
    """
    Evaluates the ARMA model on the test data.
 
    Args:
        df (pd.DataFrame): DataFrame containing 'Date' and 'Close' Price.
        ticker (str): Stock ticker symbol.
 
    Returns:
        float: MSE of the ARMA model on test data.
    """
    # Split data
    arma_train = df.iloc[-32:-2].copy()  # 30 days before test
    arma_test = df.iloc[-2:].copy()      # Last 2 days as test
 
    # Ensure enough data
    if len(arma_train) < 30 or len(arma_test) < 2:
        return np.inf
 
    # Set Date as index
    arma_train = arma_train.set_index('Date')['Close']
    arma_test = arma_test.set_index('Date')['Close']
 
    # Determine ARMA order
    arma_order = determine_arma_order(arma_train)
 
    # Fit ARMA model
    arma_model = fit_arma_model(arma_train, arma_order)
    if arma_model is None:
        return np.inf
 
    # Predict the last two days
    try:
        arma_pred = arma_model.forecast(steps=2)
        arma_pred.index = arma_test.index  # Align indices
    except Exception as e:
        return np.inf
 
    # Calculate MSE
    arma_mse = mean_squared_error(arma_test, arma_pred)
    return arma_mse
 
def plot_predictions_arma(df, ticker, arma_forecast, arma_mse, model, forecast, days_shift=5):
    """
    Plots the ARMA model predictions and seasonal decomposition with option to shift dates.
    First day forecast in blue, remaining forecast in red.
 
    Args:
        df (pd.DataFrame): Original DataFrame with 'Date' and 'Close'.
        ticker (str): Stock ticker symbol.
        arma_forecast (pd.Series): Forecasted prices.
        arma_mse (float): MSE of the ARMA model.
        model: ARMA model object.
        forecast: Forecasted components for plotting.
        days_shift (int): Number of days to shift the dates (default: 3)
    """
    # Calculate the date 3 months ago from the most recent data point
    last_date = df['Date'].max()
    three_months_ago = last_date - pd.DateOffset(months=1)
    last_test_date = df.iloc[-2:]['Date'].max()
    
    # Filter historical data to include only the last 3 months
    df_recent = df[df['Date'] >= three_months_ago].copy()
    
    # Shift dates by the specified number of days
    df_recent['Date'] = df_recent['Date'] + pd.DateOffset(days=days_shift)
    
    # Determine the split point between historical and forecasted data
    forecast_start_date = arma_forecast.index[0]
    
    # Split the recent data into historical (before forecast) and forecast (after forecast start date)
    historical_data = df_recent[df_recent['Date'] < forecast_start_date + pd.DateOffset(days=days_shift)]
    
    # Shift forecast dates
    shifted_forecast_index = arma_forecast.index + pd.DateOffset(days=days_shift)
    shifted_forecast = pd.Series(arma_forecast.values, index=shifted_forecast_index)
    
    if forecast_start_date > last_test_date:
        # Adjust the forecast to align with the last test date (shifted)
        shifted_forecast.index = pd.date_range(
            start=last_test_date + pd.DateOffset(days=days_shift),
            periods=len(shifted_forecast),
            freq='D'
        )
    
    # Plot the data
    plt.figure(figsize=(14, 7))
    
    # Plot historical data in blue
    plt.plot(historical_data['Date'], historical_data['Close'],
             color='blue', label='Historical Data', linewidth=1.5)
    
    # Plot first day forecast in blue
    plt.plot(shifted_forecast.index[:1], shifted_forecast.values[:1],
             color='blue', linewidth=2)
    
    # Plot remaining forecast days in red
    if len(shifted_forecast) > 1:
        plt.plot(shifted_forecast.index[1:], shifted_forecast.values[1:],
                color='red', label='ARMA Forecast', linewidth=2)
    
    # Set plot title and labels
    plt.title(f'{ticker} Stock Opening Price Prediction with ARMA For Next One Week')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    # Save the plot
    plt.savefig(f"{ticker}_analysis.png")
 
    # model.plot_components(forecast)
 
    
    # # Seasonal Decomposition Plot
    # try:
    #     # Perform seasonal decomposition
    #     decomposition = seasonal_decompose(df.set_index('Date')['Close'], model='additive', period=30)  # Assuming monthly seasonality
        
    #     fig = decomposition.plot()
    #     fig.set_size_inches(14, 10)
    #     plt.suptitle(f'{ticker} Seasonal Decomposition with ARMA', fontsize=16)
    #     plt.show()
    # except Exception as e:
    pass  # If seasonal decomposition fails, skip plotting
    
    # Print MSE
    # Commented out as per user request
    # print(f"ARMA Mean Squared Error on Test Set: {arma_mse:.2f}")
def get_data_sari(ticker):
    stock = yf.download(ticker, period='1y', progress=False)
    df = stock[['Close']].reset_index()
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    return df
def hyperparameter_tuning_sari(data):
    best_order = None
    best_seasonal_order = None
    best_model = None
 
    # Iterate over all combinations of (p,d,q) and seasonal (P,D,Q,s)
    for p in range(6):
        for d in range(3):
            for q in range(6):
                for P in range(2):
                    for D in range(2):
                        for Q in range(2):
                            for s in range(5):
                                order = (p,d,q)
                                seasonal_order = (P,D,Q,s)
                                try:
                                    # Fit SARIMAX model
                                    rmse=0
                                    for i in range(40):
                                        model = SARIMAX(data[i*7:56+i*7]['Close'], order=order, seasonal_order=seasonal_order)
                                        result = model.fit(disp=False)
 
                                        # Make predictions
                                        predicted_values = result.predict(start=56+i*7, end=56+i*8-1, dynamic=False)
 
                                        # Calculate RMSE
                                        rmse += sqrt(mean_squared_error(data[56+i*7:56+i*8]['Close'], predicted_values))
                                        print(f"RMSE for order {order} and seasonal_order {seasonal_order}: {rmse}")
                                    rmse = rmse/40
                                        # Update best model if we found a lower RMSE
                                    if rmse < best_rmse:
                                        best_rmse = rmse
                                        best_order = order
                                        best_seasonal_order = seasonal_order
                                        best_model = result
                                except Exception as e:
                                    print(f"Error for order {order} and seasonal_order {seasonal_order}: {e}")
                                    continue    
 
    # Display the best model parameters and RMSE
    print(f"Best RMSE: {best_rmse}")
    print(f"Best order: {best_order}")
    print(f"Best seasonal_order: {best_seasonal_order}")
 
    return best_order, best_seasonal_order, best_rmse
def predict_with_best_model_sari(data, best_order, best_seasonal_order):
    model = SARIMAX(data[len(data)-35:len(data)-7]['Close'], order=best_order, seasonal_order=best_seasonal_order)
    result = model.fit(disp=False)
    predicted_values = result.predict(start=len(data)-7, end=len(data)-1, dynamic=False)
    return predicted_values
def plotting_sari(ticker,data, predicted_values):
    plt.figure(figsize=(14, 7))
    plt.plot(data[len(data)-35:len(data)-7]['Date'], data[len(data)-35:len(data)-7]['Close'], label='Actual Data')
    plt.plot(data[len(data)-7:len(data)]['Date'], predicted_values, label='Predicted Data')
    plt.title('Stock Price Prediction with SARIMAX')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{ticker}_analysis.png')
def get_data_lstm(ticker):
    stock = yf.download(ticker, period='5y', progress=False)
    df = stock[['Close']].reset_index()
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    return df
def create_dataset(data, look_back=60):
    X, y = [], []
    for i in range(look_back, len(data)-7):
        X.append(data[i-look_back:i, 0])
        y.append(data[i:i+7, 0])
    return np.array(X), np.array(y)
def train_model_lstm(data, look_back=60):
    prices = data['Close'].values.reshape(-1, 1)  # Use close prices

    # Data normalization
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(prices)

    # Train-test split (e.g., 80% train, 20% test)
    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size - look_back:]
    X_train, y_train = create_dataset(train_data, look_back)
    X_test, y_test = create_dataset(test_data, look_back)

    # Reshape for LSTM input
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Model architecture
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))

    # Compile model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
    model.save('lstm_stock_model.h5')
    # Predict on the test set
    predicted_prices = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)  # Scale back to original prices
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Calculate error
    mse = mean_squared_error(y_test_actual, predicted_prices)
    mae = mean_absolute_error(y_test_actual, predicted_prices)
    return model, mse, predicted_prices
def plotting_lstm(data, predicted_prices):  
    plt.figure(figsize=(14, 7))
    plt.plot(data['Date'][len(data)-len(predicted_prices)-60:len(data)-len(predicted_prices)], data['Close'][-len(predicted_prices):], label='Actual Data')
    plt.plot(data['Date'][-len(predicted_prices):], predicted_prices, label='Predicted Data')
    plt.title('Stock Price Prediction with LSTM')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{ticker}_analysis.png')
    
    # Plot
# --------------------------
# 4. Model Selection and Plotting
# --------------------------
 
def select_better_model(arma_mse, prophet_mse,sarimax_mse,lstm_mse):
    """
    Selects the better model based on MSE.
 
    Args:
        arma_mse (float): MSE of ARMA model.
        prophet_mse (float): MSE of Prophet model.
 
    Returns:
        str: Name of the better model ('ARMA' or 'Prophet').
    """
    if arma_mse < prophet_mse and arma_mse < sarimax_mse and arma_mse < lstm_mse:
        return 'ARMA'
    elif prophet_mse < arma_mse and prophet_mse < sarimax_mse and prophet_mse < lstm_mse:
        return 'Prophet'
    elif sarimax_mse < arma_mse and sarimax_mse < prophet_mse and sarimax_mse < lstm_mse:
        return 'SARIMAX'
    else:
        return 'LSTM'
# --------------------------
# 5. Main Execution Workflow
# --------------------------
 
def main(sticke):
    # Fixed stock ticker
    ticker = sticke
 
    # --------------------------
    # Prophet Model Execution
    # --------------------------
    
    df_prophet = get_stock_data_prophet(ticker, period='2y')  # Using 2 years to match ARMA's start date
 
    # Ensure there are enough data points
    if len(df_prophet) < 10:
        mse_prophet = np.inf
    else:
        # Split data into training and test sets (last two days as test)
        train_df_prophet = df_prophet.iloc[:-2]
        test_df_prophet = df_prophet.iloc[-2:]
 
        # Train model and get predictions
        model_prophet, forecast_prophet, mse_prophet, future_predictions_prophet = train_and_predict_prophet(train_df_prophet, test_df_prophet, future_days=7)
 
        # Get future prediction metrics
        future_metrics_prophet = get_future_prediction_metrics_prophet(future_predictions_prophet)
 
    # --------------------------
    # ARMA Model Execution
    # --------------------------
    
    df_arma = get_stock_data_arma(ticker, start_date='2022-11-06', end_date='2024-11-06')
 
    if df_arma is None:
        arma_mse = np.inf
    else:
        # Ensure there is enough data
        required_days_arma = 30 + 2  # 30 days training + 2 days test
        if len(df_arma) < required_days_arma:
            arma_mse = np.inf
        else:
            # Evaluate ARMA model
            arma_mse = evaluate_arma(df_arma, ticker)
    df_sari = get_data_sari(ticker)
    best_order, best_seasonal_order, sarimax_mse = hyperparameter_tuning_sari(df_sari)
    predicted_values_sari = predict_with_best_model_sari(df_sari, best_order, best_seasonal_order)
    df_lstm = get_data_lstm(ticker)
    model_lstm, lstm_mse, predicted_prices_lstm = train_model_lstm(df_lstm)
    
    # --------------------------
    # Compare MSEs and Plot
    # --------------------------
    
    # Determine which model has lower MSE
    print("Finding the better model...")
    better_model = select_better_model(arma_mse, mse_prophet, sarimax_mse, lstm_mse)
    
    # Forecast the next seven days using the better model and plot
    if better_model == 'Prophet' and 'model_prophet' in locals():
        # Prophet forecasting
        plot_predictions_prophet(train_df_prophet, test_df_prophet, forecast_prophet, mse_prophet, future_predictions_prophet, model_prophet)
    elif better_model == 'ARMA':
        # ARMA forecasting
        if df_arma is not None and len(df_arma) >= 32:
            # Use the last 30 days before test for full training
            arma_train_full = df_arma.iloc[-32:-2].copy()
            arma_train_full = arma_train_full.set_index('Date')['Close']
            arma_order_full = determine_arma_order(arma_train_full)
            arma_model_full = fit_arma_model(arma_train_full, arma_order_full)
            if arma_model_full is not None:
                arma_forecast = predict_arma(arma_model_full, steps=7)
                if arma_forecast is not None:
                    # Generate future dates (business days)
                    last_date = df_arma['Date'].max()
                    future_dates = []
                    current_date = last_date
                    while len(future_dates) < 7:
                        current_date += timedelta(days=1)
                        if current_date.weekday() < 5:  # Monday-Friday are business days
                            future_dates.append(current_date)
                    arma_forecast.index = future_dates
                    plot_predictions_arma(df_arma, ticker, arma_forecast, arma_mse,model_prophet,forecast_prophet)
                else:
                    pass  # ARMA forecasting failed
            else:
                pass  # ARMA model fitting failed
        else:
            pass  # Not enough data or ARMA model failed
    elif better_model == 'SARIMAX':
        plotting_sari(ticker,df_sari, predicted_values_sari)
    elif better_model == 'LSTM':
        plotting_lstm(df_lstm, predicted_prices_lstm)
 
    # --------------------------
    # Display Forecasted Values
    # --------------------------
    print("predicting with the best model and plotting...")
    # Only print the predicted values for the next 7 days
    if better_model == 'Prophet' and 'future_metrics_prophet' in locals():
        print("\nFuture Predictions for next 7 days with Prophet:")
        print(future_metrics_prophet[['date', 'predicted_price']].to_string(index=False))
        # saving future_metrics_prophet[['date', 'predicted_price']]
        future_metrics_prophet[['date', 'predicted_price']].to_csv(f"{ticker}_future_predictions.csv",index=False)
    elif better_model == 'ARMA' and 'arma_forecast' in locals():
        future_predictions_arma = pd.DataFrame({
            'date': arma_forecast.index,
            'predicted_price': arma_forecast.values.round(2)
        })
        # saving future_predictions_arma[['date', 'predicted_price']]
        future_predictions_arma[['date', 'predicted_price']].to_csv(f"{ticker}_future_predictions.csv",index=False)
 
        # Print future predictions
        print("\nFuture Predictions for the next 7 days:")
        print(future_predictions_arma.to_string(index=False))
 
        # Calculate percentage increase or decrease
        first_price = future_predictions_arma['predicted_price'].iloc[0]
        last_price = future_predictions_arma['predicted_price'].iloc[-1]
        percentage_change = ((last_price - first_price) / first_price) * 100
 
        # Determine increase or decrease
        if percentage_change > 0:
            with open(f"{ticker}_percentage_change.txt", "w") as f:
                f.write(f"The predicted price increased by {percentage_change:.2f}% after 7 days.")
        else:
            with open(f"{ticker}_percentage_change.txt", "w") as f:
                f.write(f"The predicted price increased by {percentage_change:.2f}% after 7 days.")
 
    elif better_model == 'SARIMAX' and 'predicted_values_sari' in locals():
        future_predictions_sari = pd.DataFrame({
            'date': df_sari['Date'].iloc[-7:],
            'predicted_price': predicted_values_sari.round(2)
        })
        # saving future_predictions_sari[['date', 'predicted_price']]
        future_predictions_sari[['date', 'predicted_price']].to_csv(f"{ticker}_future_predictions.csv",index=False)
 
        # Print future predictions
        print("\nFuture Predictions for the next 7 days:")
        print(future_predictions_sari.to_string(index=False))
 
        # Calculate percentage increase or decrease
        first_price = future_predictions_sari['predicted_price'].iloc[0]
        last_price = future_predictions_sari['predicted_price'].iloc[-1]
        percentage_change = ((last_price - first_price) / first_price) * 100
 
        # Determine increase or decrease
        if percentage_change > 0:
            with open(f"{ticker}_percentage_change.txt", "w") as f:
                f.write(f"The predicted price increased by {percentage_change:.2f}% after 7 days.")
        else:
            with open(f"{ticker}_percentage_change.txt", "w") as f:
                f.write(f"The predicted price increased by {percentage_change:.2f}% after 7 days.")
    elif better_model == 'LSTM' and 'predicted_prices_lstm' in locals():
        future_predictions_lstm = pd.DataFrame({
            'date': df_lstm['Date'].iloc[-7:],
            'predicted_price': predicted_prices_lstm.round(2)
        })
        # saving future_predictions_lstm[['date', 'predicted_price']]
        future_predictions_lstm[['date', 'predicted_price']].to_csv(f"{ticker}_future_predictions.csv",index=False)
 
        # Print future predictions
        print("\nFuture Predictions for the next 7 days:")
        print(future_predictions_lstm.to_string(index=False))
 
        # Calculate percentage increase or decrease
        first_price = future_predictions_lstm['predicted_price'].iloc[0]
        last_price = future_predictions_lstm['predicted_price'].iloc[-1]
        percentage_change = ((last_price - first_price) / first_price) * 100
 
        # Determine increase or decrease
        if percentage_change > 0:
            with open(f"{ticker}_percentage_change.txt", "w") as f:
                f.write(f"The predicted price increased by {percentage_change:.2f}% after 7 days.")
        else:
            with open(f"{ticker}_percentage_change.txt", "w") as f:
                f.write(f"The predicted price increased by {percentage_change:.2f}% after 7 days.")
 
# if __name__ == "__main__":
#     main('TCS')

if __name__ == "__main__":
    if len(sys.argv) > 1:
        sticke = sys.argv[1]
        main(sticke)
    else:
        print("Please provide a stock ticker symbol as an argument.")
