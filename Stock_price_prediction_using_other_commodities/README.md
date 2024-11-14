'''The code and research regarding which models to use in this module was done by me, Ramanan.
For stock price prediction as can be seen in this experiment https://www.researchgate.net/publication/365833612_Profit_Prediction_Using_ARIMA_SARIMA_and_LSTM_Models_in_Time_Series_Forecasting_A_Comparison
LSTMs and SARIMAX models have 97 and 94.38 percentage accuracies and were better than the ARIMA model with 93.84 percent accuracy. Even from a theoretical perspective the
SARIMAX models are better than the ARIMA models for stock prediction as stock prices show good seasonality like Mondays showing lower returns and Fridays showing higher returns
 and seasonality is not accounted for in the ARIMA models whereas it's done in the SARIMAX models. But it wasn't clear as to which one of the LSTM and SARIMAX models are 
 better from a theoretical perspective, so I thought I'll code both and give the user whichever model gives a lesser error for the paricular ticker given by the user.
But my teammates had already written the code for the prophet and ARMA models, so we decided we'll predict using all the 4 models and give whichever model has the lesser 
error for the ticker to the user.

The integration with UI had already been done for the ARMA and Prophet models by my teammates, and the integration of the LSTM and SARIMAX
models with UI was done by me, Ramanan.

In the commodities_price_prediction.py, it takes the gold prices for the last 7 days and predicts how much the nifty and the sp500 stock prices will change. I decided to use the xgboost
regressor model as xgboost is one of the best models for any classification or regression task with tabular data. I also created columns for the price difference between the 
1st and 7th day of the week and the mean returns for the week and the variance for the week as it would be relevant for predicting the future stock prices. In this file
it also gives the user how much the prices of stocks of other commodities like like Oil, 
