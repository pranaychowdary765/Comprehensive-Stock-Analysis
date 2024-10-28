import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from datetime import datetime

# Fetching stock data using yfinance
def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

# Preparing the data for LSTM model
def prepare_data(stock_data):
    # Using the 'Close' price as the feature for prediction
    close_prices = stock_data['Close'].values
    close_prices = close_prices.reshape(-1, 1)

    # Normalize the data using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)

    return scaled_data, scaler

# Creating dataset with look-back
def create_dataset(data, look_back=60):
    x_data, y_data = [], []
    for i in range(look_back, len(data)):
        x_data.append(data[i-look_back:i, 0])
        y_data.append(data[i, 0])
    return np.array(x_data), np.array(y_data)

# Building the LSTM model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))  # Output layer
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Predicting the next day's stock price movement
def predict_movement(model, data, scaler, look_back=60):
    last_days = data[-look_back:]
    last_days_scaled = scaler.transform(last_days)
    X_test = np.array([last_days_scaled])
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    predicted_price = model.predict(X_test)
    predicted_price = scaler.inverse_transform(predicted_price)

    return predicted_price

# Main function to execute stock market analysis
def stock_analysis(ticker, start_date, end_date):
    stock_data = fetch_stock_data(ticker, start_date, end_date)
    scaled_data, scaler = prepare_data(stock_data)

    # Creating the dataset with look-back window
    look_back = 60  # Looking back 60 days to predict the next day's price
    X, y = create_dataset(scaled_data, look_back)

    # Reshaping X to be compatible with LSTM [samples, time steps, features]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Splitting the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Building and training the LSTM model
    model = build_lstm_model(input_shape=(X_train.shape[1], 1))
    model.fit(X_train, y_train, epochs=10, batch_size=32)

    # Predicting the next day's stock movement
    predicted_price = predict_movement(model, stock_data['Close'].values.reshape(-1, 1), scaler)

    # Comparing the predicted price with the last known closing price
    last_price = stock_data['Close'].values[-1]
    print(f"Last closing price: {last_price}")
    print(f"Predicted price for next day: {predicted_price[0][0]}")

    # Analyzing if the stock is predicted to increase or decrease
    if predicted_price > last_price:
        print(f"The stock price of {ticker} is predicted to INCREASE in the next 24 hours.")
    else:
        print(f"The stock price of {ticker} is predicted to DECREASE in the next 24 hours.")

# Plotting the results for better visualization
def plot_predictions(stock_data, predicted_price, scaler):
    plt.figure(figsize=(12,6))
    plt.plot(stock_data['Close'], label='Actual Stock Price')
    plt.plot(range(len(stock_data)-1, len(stock_data)+1), [stock_data['Close'].values[-1], predicted_price], color='red', label='Predicted Stock Price')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

# Run the analysis for a specific stock
ticker = "ITC"
start_date = "2020-01-01"
end_date = datetime.now().strftime('%Y-%m-%d')  # Fetch data up to today

stock_analysis(ticker, start_date, end_date)
