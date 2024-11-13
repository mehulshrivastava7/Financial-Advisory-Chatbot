"""Couldn't integrate on time"""
"""This is also for Stock Prediction"""

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
