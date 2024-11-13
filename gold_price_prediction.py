"""Addedd later, were not able to integrate it with the UI"""
"""This code was written by ramanan"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import yfinance as yf
import os
#This file was written by Ramanan

PROJECT_DIR = "/home/ramanant/myenv/project/"
model1 = xgb.XGBRegressor()
model2=xgb.XGBRegressor()
MODEL1_PATH = os.path.join(PROJECT_DIR, 'xgboost_model_india.json')
MODEL2_PATH = os.path.join(PROJECT_DIR, 'xgboost_model_us.json')
# Load the model from the JSON file
model1.load_model(MODEL1_PATH)
model2.load_model(MODEL2_PATH)
gold_ticker = "GC=F"

# Download the last 7 days of gold prices
gold_data = yf.download('GC=F', period='1mo')

# Ensure the data is ordered by date (in case it's not)
gold_data = gold_data.sort_index()

# Select the last 7 days of data
gold_data_last = gold_data['Close'].tail(7)
gold_data_last.to_csv(PROJECT_DIR+'gold.csv', index=True)
# Create a DataFrame with the latest data
latest_data = gold_data
latest_data['Gold'] = latest_data['Close']
latest_data=latest_data[['Gold']]
latest_data=latest_data.tail(7)
    
# Create lagged features for the past 7 days
for lag in range(1, 7):
    latest_data[f'Gold_Price_Lag_{lag}'] = latest_data['Gold'].shift(lag)

# Drop rows with NaN values (only the first row will have NaNs in this case)
latest_data.dropna(inplace=True)

# Calculate custom features
latest_data['Gold_Price_Avg'] = latest_data[[f'Gold_Price_Lag_{i}' for i in range(1, 7)]].mean(axis=1)
latest_data['Gold_Price_Variance'] = latest_data[[f'Gold_Price_Lag_{i}' for i in range(1, 7)]].var(axis=1)
latest_data['Gold_Price_diff'] = latest_data['Gold'] - latest_data['Gold_Price_Lag_6']

# Step 3: Select the features required by the model
X_latest = latest_data[[f'Gold_Price_Lag_{i}' for i in range(1, 7)] +['Gold', 'Gold_Price_Avg', 'Gold_Price_Variance', 'Gold_Price_diff']]
predictions1 = model1.predict(X_latest)
predictions2 = model2.predict(X_latest)
pred_ind_path= os.path.join(PROJECT_DIR+ 'pred_india.txt')
pred_us_path= os.path.join(PROJECT_DIR+ 'pred_us.txt')
with open(pred_ind_path, 'w') as f:
    f.write(f"The predicted Nifty 50 returns for tomorrow using gold prices are as follows:{predictions1[0]}\n")
with open(pred_us_path, 'w') as f:
    f.write(f"The predicted S&P 500 returns for tomorrow using gold prices are as follows:{predictions2[0]}\n")
oil_path = os.path.join(PROJECT_DIR, 'oil.txt')
oil_data = yf.download('CL=F', period="1mo", interval="1d")
if len(oil_data) >= 2:
    oil_oldest_price = oil_data['Close'].iloc[-7]
    oil_latest_price = oil_data['Close'].iloc[-1]
    oil_price_difference = oil_latest_price.iloc[0] - oil_oldest_price.iloc[0]
    oil_trend = "increasing" if oil_price_difference > 0 else "decreasing"
    with open(oil_path, 'w') as f:
        f.write(f"Oil price is {oil_trend} with price difference in $: {oil_price_difference:.2f}, so oil selling companies' stock prices tend to be {oil_trend}.Whereas it's the other way around for oil buying companies like airlines and transportations.\n")
else:
    print("Not enough data for Oil.")
copper_path = os.path.join(PROJECT_DIR, 'copper.txt')
nickel_path = os.path.join(PROJECT_DIR, 'nickel.txt')
lithium_path = os.path.join(PROJECT_DIR, 'lithium.txt')
cobalt_path = os.path.join(PROJECT_DIR, 'cobalt.txt')
# Download copper data (last 7 days)
copper_data = yf.download('HG=F', period="1mo", interval="1d")
if len(copper_data) >= 2:
    copper_oldest_price = copper_data['Close'].iloc[-7]
    copper_latest_price = copper_data['Close'].iloc[-1]
    copper_price_difference = copper_latest_price.iloc[0] - copper_oldest_price.iloc[0]
    copper_trend = "increasing" if copper_price_difference > 0 else "decreasing"
    with open(copper_path, 'w') as f:
        f.write(f"Copper price is {copper_trend} with price difference in $: {copper_price_difference:.2f}, so copper selling companies' stock prices tend to be {copper_trend}.Whereas it's the other way around for copper buying companies like automotive, construction and electronics companies.\n")
else:
    print("Not enough data for Copper.")

# Download nickel data (last 7 days)


# Download lithium data (last 7 days, using LIT as a proxy)
lithium_data = yf.download('LIT', period="1mo", interval="1d")
if len(lithium_data) >= 2:
    lithium_oldest_price = lithium_data['Close'].iloc[-7]
    lithium_latest_price = lithium_data['Close'].iloc[-1]
    lithium_price_difference = lithium_latest_price.iloc[0] - lithium_oldest_price.iloc[0]
    lithium_trend = "increasing" if lithium_price_difference > 0 else "decreasing"
    with open(lithium_path, 'w') as f:
        f.write(f"Lithium price is {lithium_trend} with price difference in $: {lithium_price_difference:.2f},so lithium selling companies' stock prices tend to be {lithium_trend}.Whereas it's the other way aroud for lithim buying companies like EV batteries producng companies.\n")
else:
    print("Not enough data for Lithium.")

# Download cobalt data (last 7 days, using BATT as a proxy)
cobalt_data = yf.download('BATT', period="1mo", interval="1d")
if len(cobalt_data) >= 2:
    cobalt_oldest_price = cobalt_data['Close'].iloc[-7]
    cobalt_latest_price = cobalt_data['Close'].iloc[-1]
    cobalt_price_difference = cobalt_latest_price.iloc[0] - cobalt_oldest_price.iloc[0]
    cobalt_trend = "increasing" if cobalt_price_difference > 0 else "decreasing"
    with open(cobalt_path, 'w') as f:
        f.write(f"Cobalt price is {cobalt_trend} with price difference in $: {cobalt_price_difference:.2f},so cobalt selling companies' stock prices tend to be {cobalt_trend}.Whereas it's the other way around for cobalt buying companies like battery manufacturing and EV vehicle companies.\n")

else:
    print("Not enough data for Cobalt.")
all_data = pd.DataFrame({
    'Oil trade price in $': oil_data['Close'].tail(7).values.ravel(),
    'Copper trade price in $': copper_data['Close'].tail(7).values.ravel(),
    'Lithium trade price in $ (estimate)': lithium_data['Close'].tail(7).values.ravel(),
    'Cobalt trade price in $ (estimate)': cobalt_data['Close'].tail(7).values.ravel(),
},index=gold_data_last.index)
all_data=all_data.tail(7)
all_data.to_csv(PROJECT_DIR+'all_data.csv', index=True)
if __name__ == "__main__":
    try:
        pass
    except Exception as e:
        print(f"An error occurred: {str(e)}")
