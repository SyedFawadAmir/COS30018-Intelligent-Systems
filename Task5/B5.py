# File: stock_prediction_task5_complete.py
# Authors: Bao Vo and Cheong Koo
# Date: 14/07/2021(v1); 19/07/2021(v2); 02/07/2024(v3); Updated for Task 5

# Code modified from:
# Title: Predicting Stock Prices with Python
# YouTube link: https://www.youtube.com/watch?v=PuZY9q-aKLw
# By: NeuralNine

# Required installations (best in a virtual env):
# pip install numpy matplotlib pandas tensorflow scikit-learn yfinance mplfinance

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import mplfinance as mpf
import os
import yfinance as yf
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, RNN, SimpleRNNCell, Bidirectional
from tensorflow.keras.optimizers import Adam, RMSprop, SGD, Adadelta
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

#------------------------------------------------------------------------------
# Load Data Function
def load_data(ticker, start_date=None, end_date=None, handle_nan='drop', split_method='ratio', split_ratio=0.8,
              split_date=None, shuffle=True, save_local=True, file_path=None, scale_features=True):
    """
    Loads stock data from Yahoo Finance, handles NaN values, and splits the data into
    training and testing sets using the specified method (by ratio, by date, or randomly),
    and optionally saves/loads the data locally.

    Parameters:
    - ticker (str): The ticker symbol of the stock (e.g., "AAPL").
    - start_date (str): Start date for data in the format 'YYYY-MM-DD'. Defaults to None.
    - end_date (str): End date for data in the format 'YYYY-MM-DD'. Defaults to None.
    - handle_nan (str): 'drop' to drop NaNs, 'fill' to fill NaNs with column mean. Defaults to 'drop'.
    - split_method (str): 'ratio', 'date', or 'random'. Determines how to split the data.
    - split_ratio (float): Ratio for train/test split if `split_method='ratio'`. Defaults to 0.8.
    - split_date (str): Date for splitting the data if `split_method='date'. Defaults to None.
    - shuffle (bool): Whether to shuffle the data before splitting. Defaults to True.
    - save_local (bool): If True, the data will be saved locally. Defaults to True.
    - file_path (str): Path to the file where the data will be saved/loaded. Defaults to None.
    - scale_features (bool): If True, scale the feature columns. Defaults to True.

    Returns:
    - X_train, X_test (pd.DataFrame): Training and testing sets for features.
    - y_train, y_test (np.ndarray): Training and testing sets for the target variable.
    - scalers (dict): A dictionary storing the scalers for each feature column (if scaled).
    - raw_data (pd.DataFrame): The raw stock data used for plotting charts.
    """
    # Set a default file path if not provided
    if file_path is None:
        file_path = f"{ticker}_data.csv"

    # Check if the local file exists
    if os.path.exists(file_path):
        raw_data = pd.read_csv(file_path, index_col='Date', parse_dates=True)
        print(f"Data loaded from local file: {file_path}")
    else:
        if start_date is None:
            start_date = dt.datetime(2010, 1, 1).strftime('%Y-%m-%d')
        
        if end_date is None:
            end_date = dt.datetime.now().strftime('%Y-%m-%d')

        raw_data = yf.download(ticker, start=start_date, end=end_date)

        if raw_data.empty:
            raise ValueError(f"No data found for {ticker} between {start_date} and {end_date}.")

        if save_local:
            raw_data.to_csv(file_path)
            print(f"Data saved locally to {file_path}")

    # Handle NaN values
    if handle_nan == 'drop':
        raw_data = raw_data.dropna()
        print("Dropped rows with NaN values.")
    elif handle_nan == 'fill':
        raw_data = raw_data.fillna(raw_data.mean())
        print("Filled NaN values with column mean.")

    # Features and target variable (using 'Close' as the target)
    X = raw_data[['Open', 'High', 'Low', 'Volume', 'Adj Close']]  # Use all relevant features
    y = raw_data['Close']  # Target remains 'Close'

    # Dictionary to store scalers for each feature
    scalers = {}

    # Scale the features if requested
    if scale_features:
        for column in X.columns:
            scaler = MinMaxScaler()
            X[column] = scaler.fit_transform(X[column].values.reshape(-1, 1))
            scalers[column] = scaler  # Store the scaler for future use
        print("Features scaled using MinMaxScaler.")

        # Scale the target variable y
        y_scaler = MinMaxScaler()
        y = y_scaler.fit_transform(y.values.reshape(-1, 1))
        scalers['target'] = y_scaler
        print("Target variable scaled using MinMaxScaler.")

    # Splitting the data
    if split_method == 'ratio':
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - split_ratio, shuffle=shuffle)
        print(f"Data split by ratio: {split_ratio*100}% training, {(1-split_ratio)*100}% testing.")

    elif split_method == 'date':
        if split_date is None:
            raise ValueError("split_date must be provided when split_method='date'.")

        X_train = X.loc[:split_date]
        X_test = X.loc[split_date:]
        y_train = y[:len(X_train)]
        y_test = y[len(X_train):]
        print(f"Data split by date: {split_date}.")

    elif split_method == 'random':
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - split_ratio, shuffle=shuffle)
        print("Data split randomly.")

    else:
        raise ValueError("Invalid split_method. Choose from 'ratio', 'date', or 'random'.")

    return X_train, X_test, y_train, y_test, scalers, raw_data

#------------------------------------------------------------------------------
# Prepare Data Functions
def create_sequences(X, y, time_steps=60):
    Xs, ys = [], []
    for i in range(time_steps, len(X)):
        Xs.append(X[i-time_steps:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

def create_multistep_sequences(X, y, time_steps=60, future_steps=1):
    """
    Creates sequences for multistep prediction.

    Parameters:
    - X (np.ndarray): Feature data.
    - y (np.ndarray): Target data.
    - time_steps (int): Number of past time steps to consider.
    - future_steps (int): Number of future steps to predict.

    Returns:
    - Xs (np.ndarray): Input sequences.
    - ys (np.ndarray): Output sequences for future steps.
    """
    Xs, ys = [], []
    for i in range(len(X) - time_steps - future_steps + 1):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[(i + time_steps):(i + time_steps + future_steps)])
    return np.array(Xs), np.array(ys)

def prepare_multivariate_data(X_train, y_train, X_test, y_test, time_steps=60, future_steps=1):
    """
    Prepares multivariate data sequences for training/testing.

    Parameters:
    - X_train (np.ndarray): Training feature data.
    - y_train (np.ndarray): Training target data.
    - X_test (np.ndarray): Testing feature data.
    - y_test (np.ndarray): Testing target data.
    - time_steps (int): Number of past time steps to consider.
    - future_steps (int): Number of future steps to predict.

    Returns:
    - x_train (np.ndarray): Training input sequences.
    - y_train_seq (np.ndarray): Training target sequences.
    - x_test (np.ndarray): Testing input sequences.
    - y_test_seq (np.ndarray): Testing target sequences.
    """
    x_train, y_train_seq = create_multistep_sequences(X_train, y_train, time_steps, future_steps=future_steps)
    x_test, y_test_seq = create_multistep_sequences(X_test, y_test, time_steps, future_steps=future_steps)
    return x_train, y_train_seq, x_test, y_test_seq

#------------------------------------------------------------------------------
# Model Creation Functions
def create_model(units, layer_types, input_shape, output_size, n_layers=3, dropout_rate=0.3, loss='mean_absolute_error', optimizer='adam', bidirectional=False, metrics=['mean_absolute_error']):
    """
    Creates and compiles a deep learning model.

    Parameters:
    - units (list): List of integers specifying the number of units in each layer.
    - layer_types (list): List of layer types (e.g., [LSTM, GRU]).
    - input_shape (tuple): Shape of the input data.
    - output_size (int): Size of the output layer.
    - n_layers (int): Number of layers in the model.
    - dropout_rate (float): Dropout rate for regularization.
    - loss (str): Loss function to use.
    - optimizer (str): Optimizer to use.
    - bidirectional (bool): Whether to use bidirectional layers.
    - metrics (list): List of metrics to evaluate during training.

    Returns:
    - model: Compiled Keras model.
    """
    model = Sequential()
    for i in range(n_layers):
        if i == 0:
            # First layer with input shape
            if bidirectional:
                model.add(Bidirectional(layer_types[i](units=units[i], return_sequences=True), input_shape=input_shape))
            else:
                model.add(layer_types[i](units=units[i], return_sequences=True, input_shape=input_shape))
        else:
            if i == n_layers - 1:
                # Last layer without return_sequences
                if bidirectional:
                    model.add(Bidirectional(layer_types[i](units=units[i])))
                else:
                    model.add(layer_types[i](units=units[i]))
            else:
                if bidirectional:
                    model.add(Bidirectional(layer_types[i](units=units[i], return_sequences=True)))
                else:
                    model.add(layer_types[i](units=units[i], return_sequences=True))
        model.add(Dropout(dropout_rate))

    # Output layer
    model.add(Dense(units=output_size))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model

def train_model(model, x_train, y_train, epochs, batch_size):
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    return history

#------------------------------------------------------------------------------
# Visualization Functions
def plot_candlestick_chart(data, n=1, title='Candlestick Chart', save=False, filename='candlestick_chart.png'):
    """
    Plots a candlestick chart for the provided stock market data.
    """
    # (Function code remains unchanged)
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Data must be a pandas DataFrame.")
    
    required_columns = {'Open', 'High', 'Low', 'Close', 'Volume'}
    if not required_columns.issubset(data.columns):
        raise ValueError(f"Data must contain the following columns: {required_columns}")
    
    if n < 1 or not isinstance(n, int):
        raise ValueError("Parameter 'n' must be an integer greater than or equal to 1.")
    
    if n > 1:
        resample_rule = f'{n}D'
        aggregated_data = data.resample(resample_rule).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
    else:
        aggregated_data = data.copy()
    
    style = mpf.make_mpf_style(base_mpf_style='yahoo', rc={'font.size': 10})

    if save:
        mpf.plot(
            aggregated_data,
            type='candle',
            style=style,
            title=title,
            ylabel='Price',
            volume=True,
            mav=(n),
            show_nontrading=False,
            savefig=filename
        )
        print(f"Candlestick chart saved as '{filename}'.")
    else:
        mpf.plot(
            aggregated_data,
            type='candle',
            style=style,
            title=title,
            ylabel='Price',
            volume=True,
            mav=(n),
            show_nontrading=True
        )

def plot_boxplot_chart(data, n=5, title='Boxplot Chart', save=False, filename='boxplot_chart.png'):
    """
    Plots a boxplot chart for the provided stock market closing price data over a moving window.
    """
    # (Function code remains unchanged)
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Data must be a pandas DataFrame.")
    
    if 'Close' not in data.columns:
        raise ValueError("Data must contain the 'Close' column.")
    
    if n < 1 or not isinstance(n, int):
        raise ValueError("Parameter 'n' must be an integer greater than or equal to 1.")
    
    boxplot_data = []
    labels = []
    total_windows = len(data) - n + 1
    
    if total_windows <= 0:
        raise ValueError("Not enough data points for the specified window size.")
    
    for i in range(total_windows):
        window_data = data['Close'].iloc[i:i+n]
        boxplot_data.append(window_data.values)
        window_end_date = data.index[i + n - 1].strftime('%Y-%m-%d')
        labels.append(window_end_date)
    
    plt.figure(figsize=(12, 6))
    boxprops = dict(linestyle='-', linewidth=2, color='darkblue')
    medianprops = dict(linestyle='-', linewidth=2, color='firebrick')
    
    plt.boxplot(boxplot_data, labels=labels, boxprops=boxprops, medianprops=medianprops, patch_artist=True)
    plt.title(title)
    plt.xlabel('End Date of Window')
    plt.ylabel('Closing Price')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    
    if save:
        plt.savefig(filename)
        print(f"Boxplot chart saved as '{filename}'.")
    else:
        plt.show()
    
    plt.close()

#------------------------------------------------------------------------------
# Multistep Prediction Function
def multistep_prediction(data, k_days, sequence_length=60):
    """
    Performs multistep prediction using univariate data (only 'Close' prices).

    Parameters:
    - data (pd.DataFrame): The stock data containing 'Close' prices.
    - k_days (int): Number of future days to predict.
    - sequence_length (int): Number of past days to consider for making predictions.

    Returns:
    - model: Trained model.
    - predictions: Predicted values for the test set.
    - y_test_rescaled: Actual values for the test set.
    """
    # Extract the 'Close' price
    close_data = data[['Close']]

    # Scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(close_data)

    # Split the data
    split_index = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:split_index]
    test_data = scaled_data[split_index:]

    # Prepare sequences
    def create_sequences(data, seq_length, future_steps):
        X, y = [], []
        for i in range(len(data) - seq_length - future_steps + 1):
            X.append(data[i:(i + seq_length)])
            y.append(data[(i + seq_length):(i + seq_length + future_steps)])
        return np.array(X), np.array(y).reshape(-1, future_steps)

    x_train, y_train = create_sequences(train_data, sequence_length, k_days)
    x_test, y_test = create_sequences(test_data, sequence_length, k_days)

    # Build the model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=k_days))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    history = model.fit(x_train, y_train, epochs=25, batch_size=32, validation_split=0.2)

    # Make predictions
    predictions = model.predict(x_test)
    predictions_rescaled = scaler.inverse_transform(predictions)
    y_test_rescaled = scaler.inverse_transform(y_test)

    # Plotting
    for i in range(min(5, len(predictions_rescaled))):
        plt.figure(figsize=(10, 4))
        plt.plot(range(k_days), y_test_rescaled[i], label='Actual')
        plt.plot(range(k_days), predictions_rescaled[i], label='Predicted')
        plt.title(f'Multistep Prediction Sample {i+1}')
        plt.xlabel('Future Days')
        plt.ylabel('Closing Price')
        plt.legend()
        plt.show()

    return model, predictions_rescaled, y_test_rescaled

#------------------------------------------------------------------------------
# Multivariate Prediction Function
def multivariate_prediction(data, k_days, features=None, sequence_length=60):
    """
    Performs multivariate prediction using multiple features.

    Parameters:
    - data (pd.DataFrame): The stock data containing all features.
    - k_days (int): Number of days ahead to predict (for this function, we can set k_days=1).
    - features (list): List of feature column names to use as input.
    - sequence_length (int): Number of past days to consider for making predictions.

    Returns:
    - model: Trained model.
    - predictions: Predicted values for the test set.
    - y_test_rescaled: Actual values for the test set.
    """
    if features is None:
        # Default to using all six features
        features = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    
    num_of_features = len(features)
    
    # Select features
    data_features = data[features].values
    data_target = data['Close'].values.reshape(-1, 1)

    # Scale features and target
    scaler_X = MinMaxScaler()
    scaled_input_data = scaler_X.fit_transform(data_features)
    scaler_y = MinMaxScaler()
    scaled_output_data = scaler_y.fit_transform(data_target)

    # Split data
    split_index = int(len(scaled_input_data) * 0.8)
    X_train = scaled_input_data[:split_index]
    y_train = scaled_output_data[:split_index]
    X_test = scaled_input_data[split_index:]
    y_test = scaled_output_data[split_index:]

    # Prepare sequences
    def create_sequences_multivariate(X, y, seq_length):
        X_seq, y_seq = [], []
        for i in range(len(X) - seq_length):
            X_seq.append(X[i:(i + seq_length)])
            y_seq.append(y[i + seq_length])
        return np.array(X_seq), np.array(y_seq)

    x_train, y_train_seq = create_sequences_multivariate(X_train, y_train, sequence_length)
    x_test, y_test_seq = create_sequences_multivariate(X_test, y_test, sequence_length)

    # Build the model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], num_of_features)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    history = model.fit(x_train, y_train_seq, epochs=25, batch_size=32, validation_split=0.2)

    # Make predictions
    predictions = model.predict(x_test)
    predictions_rescaled = scaler_y.inverse_transform(predictions)
    y_test_rescaled = scaler_y.inverse_transform(y_test_seq)

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_rescaled, color='blue', label='Actual Closing Price')
    plt.plot(predictions_rescaled, color='red', label='Predicted Closing Price')
    plt.title('Multivariate Prediction of Closing Price')
    plt.xlabel('Time')
    plt.ylabel('Closing Price')
    plt.legend()
    plt.show()

    return model, predictions_rescaled, y_test_rescaled

#------------------------------------------------------------------------------
# Multivariate Multistep Prediction Function
def multivariate_multistep_prediction(data, k_days, features=None, sequence_length=60):
    """
    Performs multistep prediction using multivariate data.

    Parameters:
    - data (pd.DataFrame): The stock data containing all features.
    - k_days (int): Number of future days to predict.
    - features (list): List of feature column names to use as input.
    - sequence_length (int): Number of past days to consider for making predictions.

    Returns:
    - model: Trained model.
    - predictions: Predicted values for the test set.
    - y_test_seq_rescaled: Actual values for the test set.
    """
    if features is None:
        # Default to using all six features
        features = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    
    num_of_features = len(features)
    
    # Select features
    data_features = data[features].values
    data_target = data['Close'].values.reshape(-1, 1)

    # Scale features and target
    scaler_X = MinMaxScaler()
    scaled_input_data = scaler_X.fit_transform(data_features)
    scaler_y = MinMaxScaler()
    scaled_output_data = scaler_y.fit_transform(data_target)

    # Split data
    split_index = int(len(scaled_input_data) * 0.8)
    X_train = scaled_input_data[:split_index]
    y_train = scaled_output_data[:split_index]
    X_test = scaled_input_data[split_index:]
    y_test = scaled_output_data[split_index:]

    # Prepare sequences
    def create_multistep_sequences(X, y, seq_length, future_steps):
        Xs, ys = [], []
        for i in range(len(X) - seq_length - future_steps + 1):
            Xs.append(X[i:(i + seq_length)])
            ys.append(y[(i + seq_length):(i + seq_length + future_steps)])
        return np.array(Xs), np.array(ys).reshape(-1, future_steps)

    x_train, y_train_seq = create_multistep_sequences(X_train, y_train, sequence_length, k_days)
    x_test, y_test_seq = create_multistep_sequences(X_test, y_test, sequence_length, k_days)

    # Build the model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], num_of_features)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=k_days))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    history = model.fit(x_train, y_train_seq, epochs=25, batch_size=32, validation_split=0.2)

    # Make predictions
    predictions = model.predict(x_test)
    predictions_rescaled = scaler_y.inverse_transform(predictions)
    y_test_seq_rescaled = scaler_y.inverse_transform(y_test_seq)

    # Plotting
    for i in range(min(5, len(predictions_rescaled))):
        plt.figure(figsize=(10, 4))
        plt.plot(range(k_days), y_test_seq_rescaled[i], label='Actual')
        plt.plot(range(k_days), predictions_rescaled[i], label='Predicted')
        plt.title(f'Multivariate Multistep Prediction Sample {i+1}')
        plt.xlabel('Future Days')
        plt.ylabel('Closing Price')
        plt.legend()
        plt.show()

    return model, predictions_rescaled, y_test_seq_rescaled

#------------------------------------------------------------------------------
# Main Execution
if __name__ == "__main__":
    # Parameters
    ticker = "AAPL"
    start_date = '2015-01-01'
    end_date = '2020-01-01'
    split_method = 'date'  # Options: 'ratio', 'date', 'random'
    split_date = '2018-01-01'  # Used if split_method='date'
    file_path = "AAPL_data.csv"  # Local file path for storing/loading data
    scale_features = True  # Enable feature scaling

    # Load data
    data = yf.download(ticker, start=start_date, end=end_date)
    data = data.dropna()

    # Multistep Prediction
    k_days = 5  # Number of future days to predict
    sequence_length = 60  # Number of past days to consider
    model_multistep, predictions_multistep, y_test_multistep = multistep_prediction(data, k_days, sequence_length)

    # Multivariate Prediction
    selected_features = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']  # You can specify different features here
    model_multivariate, predictions_multivariate, y_test_multivariate = multivariate_prediction(
        data, k_days=1, features=selected_features, sequence_length=60)

    # Multivariate Multistep Prediction
    model_mv_ms, predictions_mv_ms, y_test_mv_ms = multivariate_multistep_prediction(
        data, k_days, features=selected_features, sequence_length=60)

    #------------------------------------------------------------------------------
    # Generate the Candlestick Chart using data from load_data function
    plot_candlestick_chart(data, n=5, title=f'{ticker} Candlestick Chart', save=False)

    # Generate the Boxplot Chart using data from load_data function
    plot_boxplot_chart(data, n=10, title=f'{ticker} Boxplot Chart', save=False)
