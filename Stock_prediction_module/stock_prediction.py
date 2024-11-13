"""This code is by mehul"""
import pandas as pd
from prophet import Prophet
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
import logging
import sys
 
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
 """We can change the start and end date according to the current date"""
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
 
def plot_predictions_arma(df, ticker, arma_forecast, arma_mse, forecast):
    """
    Plots the ARMA model predictions and seasonal decomposition.
    First day forecast in blue, remaining forecast in red.
 
    Args:
        df (pd.DataFrame): Original DataFrame with 'Date' and 'Close'.
        ticker (str): Stock ticker symbol.
        arma_forecast (pd.Series): Forecasted prices.
        arma_mse (float): MSE of the ARMA model.
        forecast: Forecasted components for plotting.
    """
    # Calculate the date 3 months ago from the most recent data point
    last_date = df['Date'].max()
    three_months_ago = last_date - pd.DateOffset(months=1)
    last_test_date = df.iloc[-2:]['Date'].max()
    
    # Filter historical data to include only the last 3 months
    df_recent = df[df['Date'] >= three_months_ago].copy()
    
    # Determine the split point between historical and forecasted data
    forecast_start_date = arma_forecast.index[0]
    
    # Split the recent data into historical (before forecast)
    historical_data = df_recent[df_recent['Date'] < forecast_start_date]
    
    if forecast_start_date > last_test_date:
        # Adjust the forecast to align with the last test date
        arma_forecast.index = pd.date_range(
            start=last_test_date,
            periods=len(arma_forecast),
            freq='D'
        )
    
    # Plot the data
    plt.figure(figsize=(14, 7))
    
    # Plot historical data in blue
    plt.plot(historical_data['Date'], historical_data['Close'],
             color='blue', label='Historical Data', linewidth=1.5)
    
    # Plot first day forecast in blue
    plt.plot(arma_forecast.index[:1], arma_forecast.values[:1],
             color='blue', linewidth=2)
    
    # Plot remaining forecast days in red
    if len(arma_forecast) > 1:
        plt.plot(arma_forecast.index[1:], arma_forecast.values[1:],
                color='red', label='ARMA Forecast', linewidth=2)
    
    # Set plot title and labels
    plt.title(f'{ticker} Stock Opening Price Prediction with ARMA For Next One Week')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    # Save the plot
    plt.savefig(f"{ticker}_analysis.png") 


 """The below is not required now"""
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
 
# --------------------------
# 4. Model Selection and Plotting
# --------------------------


"""Check the test mse"""
def select_better_model(arma_mse, prophet_mse):
    """
    Selects the better model based on MSE.
 
    Args:
        arma_mse (float): MSE of ARMA model.
        prophet_mse (float): MSE of Prophet model.
 
    Returns:
        str: Name of the better model ('ARMA' or 'Prophet').
    """
    if arma_mse < prophet_mse:
        return 'ARMA'
    else:
        return 'Prophet'
 
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
        prophet_mse = np.inf
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
 
    # --------------------------
    # Compare MSEs and Plot
    # --------------------------
    
    # Determine which model has lower MSE
    print("Finding the better model...")
    if 'mse_prophet' in locals() and 'arma_mse' in locals():
        better_model = select_better_model(arma_mse, mse_prophet)
    else:
        better_model = 'ARMA' if arma_mse < np.inf else 'Prophet'
    
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
                    plot_predictions_arma(df_arma, ticker, arma_forecast, arma_mse,forecast_prophet)
                else:
                    pass  # ARMA forecasting failed
            else:
                pass  # ARMA model fitting failed
        else:
            pass  # Not enough data or ARMA model failed
    else:
        pass  # No valid model to forecast and plot
 
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
 
    else:
        print("\nNo predictions available.")
 
# if __name__ == "__main__":
#     main('TCS')

if __name__ == "__main__":
    if len(sys.argv) > 1:
        sticke = sys.argv[1]
        main(sticke)
    else:
        print("Please provide a stock ticker symbol as an argument.")
