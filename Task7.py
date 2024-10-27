# Required installations:
# pip install numpy matplotlib pandas tensorflow scikit-learn yfinance statsmodels tweepy nltk

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import tweepy
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Dense, Dropout
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime

# Twitter API credentials
bearer_token = 'AAAAAAAAAAAAAAAAAAAAAE6OwgEAAAAAOJ6ZnqCZC6vF9O06T%2Bmt4QvcWAE%3DzxIcLzpRLTub4qr9AouKzul7CesBXuTD1ev4d5PyCiK5SAjJhb'

# Initialize Twitter API v2 client and VADER sentiment analyzer
client = tweepy.Client(bearer_token=bearer_token)
sentiment_analyzer = SentimentIntensityAnalyzer()

# Fetch sentiment data for a given ticker and date range using Twitter API v2
def get_daily_sentiment_v2(ticker, start_date, end_date):
    dates = pd.date_range(start_date, end_date)
    sentiment_data = []

    for date in dates:
        daily_sentiment_score = 0
        tweet_count = 0
        query = f"{ticker} lang:en -is:retweet"  # Query without retweets

        try:
            # Fetch tweets for the specific date
            tweets = client.search_recent_tweets(query=query, max_results=100, end_time=date.strftime('%Y-%m-%dT00:00:00Z'))

            if tweets.data:
                for tweet in tweets.data:
                    sentiment = sentiment_analyzer.polarity_scores(tweet.text)
                    daily_sentiment_score += sentiment['compound']
                    tweet_count += 1

            avg_sentiment = daily_sentiment_score / tweet_count if tweet_count > 0 else 0
            sentiment_data.append((date, avg_sentiment))

        except tweepy.errors.Forbidden as e:
            print(f"Access denied. Error: {e}")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            break

    return pd.DataFrame(sentiment_data, columns=['Date', 'SentimentScore']).set_index('Date')

# Load stock and sentiment data, combine, and preprocess
def load_data_with_sentiment(ticker, start_date, end_date, split_date, scale_features=True):
    # Load stock price data
    raw_data = yf.download(ticker, start=start_date, end=end_date).ffill()
    
    # Fetch sentiment data
    sentiment_df = get_daily_sentiment_v2(ticker, start_date, end_date)
    
    # Merge and fill missing sentiment values with 0
    raw_data = raw_data.merge(sentiment_df, left_index=True, right_index=True, how='left').fillna(0)
    
    # Select features and target
    X = raw_data[['Open', 'High', 'Low', 'Volume', 'Adj Close', 'SentimentScore']]
    y = raw_data['Close']
    
    # Scale data
    scalers = {}
    if scale_features:
        for column in X.columns:
            scaler = MinMaxScaler()
            X[column] = scaler.fit_transform(X[column].values.reshape(-1, 1))
            scalers[column] = scaler
        y_scaler = MinMaxScaler()
        y = y_scaler.fit_transform(y.values.reshape(-1, 1))
        scalers['target'] = y_scaler
    
    # Split data
    X_train, X_test = X[:split_date], X[split_date:]
    y_train, y_test = y[:len(X_train)], y[len(X_train):]

    return X_train, X_test, y_train, y_test, scalers, raw_data

# Create sequences for model training
def create_sequences(X, y, time_steps=60):
    Xs, ys = [], []
    for i in range(time_steps, len(X)):
        Xs.append(X[i-time_steps:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

# Build RNN model
def create_rnn_model(layer_type, input_shape, units=50, optimizer='adam'):
    model = Sequential()
    model.add(layer_type(units=units, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

# Train the model
def train_rnn_model(model, x_train, y_train, epochs=10, batch_size=32):
    return model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

# ARIMA model training and prediction
def train_arima_model(train_data, order=(5,1,0)):
    model = ARIMA(train_data, order=order)
    model_fit = model.fit()
    return model_fit

def predict_arima_model(model_fit, steps):
    return model_fit.forecast(steps=steps)

# Random Forest model training and prediction
def train_random_forest_model(x_train, y_train, n_estimators=100, max_depth=10):
    rf_model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    rf_model.fit(x_train, y_train)
    return rf_model

def predict_random_forest_model(rf_model, x_test):
    return rf_model.predict(x_test)

# Ensemble Methods
def ensemble_average(predictions_list):
    return np.mean(predictions_list, axis=0)

# Evaluation
def evaluate_model(y_test, predictions, y_scaler):
    predictions = y_scaler.inverse_transform(predictions)
    y_test = y_scaler.inverse_transform(y_test.reshape(-1, 1))
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    return mse, mae

# Plot predictions
def plot_predictions(y_test, predictions, title='Model Predictions vs Actual'):
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label='Actual', color='black', linestyle='--')
    plt.plot(predictions, label='Predicted')
    plt.title(title)
    plt.xlabel('Time Steps')
    plt.ylabel('Closing Price')
    plt.legend()
    plt.show()

# Main function
if __name__ == "__main__":
    ticker = "AAPL"
    start_date = '2015-01-01'
    end_date = '2020-01-01'
    split_date = '2018-01-01'
    TIME_STEPS = 60

    # Load and preprocess data
    X_train, X_test, y_train, y_test, scalers, raw_data = load_data_with_sentiment(
        ticker, start_date, end_date, split_date, scale_features=True)
    
    X_train_seq, y_train_seq = create_sequences(X_train.values, y_train, TIME_STEPS)
    X_test_seq, y_test_seq = create_sequences(X_test.values, y_test, TIME_STEPS)

    # Inverse-transform the test data for plotting and evaluation
    y_test_rescaled = scalers['target'].inverse_transform(y_test_seq.reshape(-1, 1)).flatten()

    # Ensemble configurations
    ensemble_configs = [
        {'name': 'ARIMA + LSTM', 'models': ['arima', 'lstm']},
        {'name': 'Random Forest + LSTM', 'models': ['random_forest', 'lstm']},
    ]

    for ensemble in ensemble_configs:
        predictions_list = []
        config_idx = ensemble_configs.index(ensemble)

        for model_name in ensemble['models']:
            if model_name == 'arima':
                y_train_arima = scalers['target'].inverse_transform(y_train.flatten().reshape(-1, 1)).flatten()
                arima_model_fit = train_arima_model(y_train_arima)
                arima_pred = predict_arima_model(arima_model_fit, len(y_test_rescaled))
                predictions_list.append(arima_pred)
            elif model_name == 'lstm':
                lstm_model = create_rnn_model(LSTM, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]))
                train_rnn_model(lstm_model, X_train_seq, y_train_seq, epochs=10, batch_size=32)
                lstm_pred = lstm_model.predict(X_test_seq)
                lstm_pred = scalers['target'].inverse_transform(lstm_pred).flatten()
                predictions_list.append(lstm_pred)
            elif model_name == 'random_forest':
                x_train_rf = X_train_seq.reshape((X_train_seq.shape[0], -1))
                x_test_rf = X_test_seq.reshape((X_test_seq.shape[0], -1))
                rf_model = train_random_forest_model(x_train_rf, y_train_seq.flatten())
                rf_pred = predict_random_forest_model(rf_model, x_test_rf)
                rf_pred = scalers['target'].inverse_transform(rf_pred.reshape(-1,1)).flatten()
                predictions_list.append(rf_pred)
            else:
                print(f"Model {model_name} is not supported.")

        # Create Ensemble Prediction
        ensemble_pred = ensemble_average(predictions_list)
        mse, mae = evaluate_model(y_test_seq, ensemble_pred.reshape(-1, 1), scalers['target'])
        print(f"{ensemble['name']} - MSE: {mse:.4f}, MAE: {mae:.4f}")

        # Plot results
        plot_predictions(y_test_rescaled, ensemble_pred, title=f'{ensemble["name"]} - Actual vs Predicted')
