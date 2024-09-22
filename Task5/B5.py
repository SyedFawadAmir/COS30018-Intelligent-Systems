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
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, RNN, SimpleRNNCell
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

def create_multistep_sequences(X, y, time_steps=60, step_size=1, future_steps=1):
    """
    Creates sequences for multistep prediction.

    Parameters:
    - X (np.ndarray): Feature data.
    - y (np.ndarray): Target data.
    - time_steps (int): Number of past time steps to consider.
    - step_size (int): Step size between sequences.
    - future_steps (int): Number of future steps to predict.

    Returns:
    - Xs (np.ndarray): Input sequences.
    - ys (np.ndarray): Output sequences for future steps.
    """
    Xs, ys = [], []
    for i in range(0, len(X) - time_steps - future_steps + 1, step_size):
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
def create_dl_model(layer_type, num_layers, layer_sizes, input_shape, output_size, dropout_rate=0.2, optimizer='adam', loss='mean_squared_error'):
    model = Sequential()

    # Handle RNN differently, as it needs a cell type
    if layer_type == RNN:
        first_layer = layer_type(SimpleRNNCell(layer_sizes[0]), input_shape=input_shape, return_sequences=(num_layers > 1))
    else:
        first_layer = layer_type(units=layer_sizes[0], input_shape=input_shape, return_sequences=(num_layers > 1))
    
    model.add(first_layer)
    model.add(Dropout(dropout_rate))

    # Add hidden layers
    for i in range(1, num_layers - 1):
        if layer_type == RNN:
            model.add(layer_type(SimpleRNNCell(layer_sizes[i]), return_sequences=True))
        else:
            model.add(layer_type(units=layer_sizes[i], return_sequences=True))
        model.add(Dropout(dropout_rate))

    # Final layer (without return_sequences)
    if num_layers > 1:
        if layer_type == RNN:
            model.add(layer_type(SimpleRNNCell(layer_sizes[-1])))
        else:
            model.add(layer_type(units=layer_sizes[-1]))
        model.add(Dropout(dropout_rate))

    # Output layer
    model.add(Dense(units=output_size))

    # Compile the model
    optimizer_instance = {
        'adam': Adam(),
        'rmsprop': RMSprop(),
        'sgd': SGD(),
        'adadelta': Adadelta()
    }.get(optimizer, Adam())

    model.compile(optimizer=optimizer_instance, loss=loss, metrics=['mean_absolute_error'])

    return model

def train_model(model, x_train, y_train, epochs, batch_size):
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    return history

#------------------------------------------------------------------------------
# Visualization Functions
def plot_candlestick_chart(data, n=1, title='Candlestick Chart', save=False, filename='candlestick_chart.png'):
    """
    Plots a candlestick chart for the provided stock market data.

    Parameters:
        data (pd.DataFrame): DataFrame containing stock market data with columns ['Open', 'High', 'Low', 'Close', 'Volume'] and DateTimeIndex.
        n (int): Number of trading days to aggregate per candlestick. Must be >= 1. Default is 1.
        title (str): Title of the candlestick chart. Default is 'Candlestick Chart'.
        save (bool): If True, saves the chart as a PNG file. Default is False.
        filename (str): Filename for the saved chart if save is True. Default is 'candlestick_chart.png'.

    Returns:
        None
    """
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

    Parameters:
        data (pd.DataFrame): DataFrame containing stock market data with at least the 'Close' column and DateTimeIndex.
        n (int): Size of the moving window in trading days. Must be >= 1. Default is 5.
        title (str): Title of the boxplot chart. Default is 'Boxplot Chart'.
        save (bool): If True, saves the chart as a PNG file. Default is False.
        filename (str): Filename for the saved chart if save is True. Default is 'boxplot_chart.png'.

    Returns:
        None
    """
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
# Experimentation Function
def experiment_with_configurations(raw_data, layer_types, layer_configs, epochs_configs, batch_sizes, optimizers, prediction_days=60, future_steps=1):
    results = []
    for layer_type in layer_types:
        for num_layers, layer_sizes in layer_configs:
            for epochs in epochs_configs:
                for batch_size in batch_sizes:
                    for optimizer in optimizers:
                        print(f"Training model with {layer_type.__name__}, layers: {num_layers}, sizes: {layer_sizes}, epochs: {epochs}, batch size: {batch_size}, optimizer: {optimizer}")
                        try:
                            # Prepare data
                            X = raw_data[['Open', 'High', 'Low', 'Volume', 'Adj Close']].values
                            y = raw_data['Close'].values.reshape(-1, 1)
                            # Scale data
                            scaler_X = MinMaxScaler()
                            X_scaled = scaler_X.fit_transform(X)
                            scaler_y = MinMaxScaler()
                            y_scaled = scaler_y.fit_transform(y)

                            # Split data
                            split_index = int(len(X_scaled) * 0.8)
                            X_train_exp = X_scaled[:split_index]
                            y_train_exp = y_scaled[:split_index]
                            X_test_exp = X_scaled[split_index:]
                            y_test_exp = y_scaled[split_index:]

                            # Prepare sequences
                            x_train_exp, y_train_exp_seq = create_multistep_sequences(X_train_exp, y_train_exp, prediction_days, future_steps=future_steps)
                            x_test_exp, y_test_exp_seq = create_multistep_sequences(X_test_exp, y_test_exp, prediction_days, future_steps=future_steps)

                            # Create the model
                            model = create_dl_model(
                                layer_type=layer_type,
                                num_layers=num_layers,
                                layer_sizes=layer_sizes,
                                input_shape=(x_train_exp.shape[1], x_train_exp.shape[2]),
                                output_size=future_steps,
                                optimizer=optimizer
                            )

                            # Train the model
                            history = train_model(model, x_train_exp, y_train_exp_seq, epochs=epochs, batch_size=batch_size)

                            # Evaluate the model
                            loss, mae = model.evaluate(x_test_exp, y_test_exp_seq)
                            results.append({
                                'layer_type': layer_type.__name__,
                                'num_layers': num_layers,
                                'layer_sizes': layer_sizes,
                                'epochs': epochs,
                                'batch_size': batch_size,
                                'optimizer': optimizer,
                                'loss': loss,
                                'mean_absolute_error': mae
                            })
                        except Exception as e:
                            print(f"An error occurred: {e}")
                            continue
    return results

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
    X_train, X_test, y_train, y_test, scalers, raw_data = load_data(
        ticker, start_date, end_date, handle_nan='fill', split_method=split_method,
        split_ratio=0.8, split_date=split_date, save_local=True,
        file_path=file_path, scale_features=scale_features)

    # Parameters for prediction
    PREDICTION_DAYS = 60  # Number of past days to use for prediction
    FUTURE_STEPS = 5      # Number of future days to predict

    # Prepare multivariate, multistep data
    X_train_values = X_train.values
    y_train_values = y_train.reshape(-1)
    X_test_values = X_test.values
    y_test_values = y_test.reshape(-1)

    x_train, y_train_seq, x_test, y_test_seq = prepare_multivariate_data(
        X_train_values, y_train_values, X_test_values, y_test_values,
        time_steps=PREDICTION_DAYS, future_steps=FUTURE_STEPS)

    print(f"x_train shape: {x_train.shape}")
    print(f"y_train_seq shape: {y_train_seq.shape}")
    print(f"x_test shape: {x_test.shape}")
    print(f"y_test_seq shape: {y_test_seq.shape}")

    # Build the model
    model = create_dl_model(
        layer_type=LSTM,
        num_layers=3,
        layer_sizes=[50, 50, 50],
        input_shape=(x_train.shape[1], x_train.shape[2]),
        output_size=FUTURE_STEPS,
        dropout_rate=0.2,
        optimizer='adam',
        loss='mean_squared_error'
    )

    # Train the model
    history = train_model(
        model=model,
        x_train=x_train,
        y_train=y_train_seq,
        epochs=25,
        batch_size=32
    )

    # Evaluate the model
    loss, mae = model.evaluate(x_test, y_test_seq)
    print(f"Test Loss: {loss}, Test MAE: {mae}")

    # Make predictions
    predicted_seq = model.predict(x_test)
    predicted_seq_rescaled = scalers['target'].inverse_transform(predicted_seq)
    y_test_seq_rescaled = scalers['target'].inverse_transform(y_test_seq)

    # Plot the first prediction sequence vs actual
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_seq_rescaled[0], label='Actual')
    plt.plot(predicted_seq_rescaled[0], label='Predicted')
    plt.title('Multistep Prediction for First Test Sample')
    plt.xlabel('Future Time Steps')
    plt.ylabel('Closing Price')
    plt.legend()
    plt.show()

    #------------------------------------------------------------------------------
    # Generate the Candlestick Chart using data from load_data function
    plot_candlestick_chart(raw_data, n=5, title=f'{ticker} Candlestick Chart', save=False)

    # Generate the Boxplot Chart using data from load_data function
    plot_boxplot_chart(raw_data, n=10, title=f'{ticker} Boxplot Chart', save=False)

    #------------------------------------------------------------------------------
    # Predict next day (using the last data point in x_test)
    real_data = x_test[-1]
    real_data = np.expand_dims(real_data, axis=0)  # Reshape to match the input shape

    prediction = model.predict(real_data)
    prediction = scalers['target'].inverse_transform(prediction)
    print(f"Prediction for the next {FUTURE_STEPS} days: {prediction[0]}")

    #------------------------------------------------------------------------------
    # Define your experiment configurations
    layer_types = [LSTM, GRU, RNN]  # Test different types of layers
    layer_configs = [(2, [64, 32]), (3, [128, 64, 32])]  # Different layer configurations (num_layers, layer_sizes)
    epochs_configs = [10, 25]  # Number of epochs
    batch_sizes = [32, 64]  # Batch sizes
    optimizers = ['adam', 'rmsprop']  # Different optimizers

    # Uncomment the following code if you want to run experiments
    # Note: Running experiments can take a significant amount of time

    # results = experiment_with_configurations(raw_data, layer_types, layer_configs, epochs_configs, batch_sizes, optimizers, prediction_days=PREDICTION_DAYS, future_steps=FUTURE_STEPS)

    # # Print results for analysis
    # for result in results:
    #     print(result)
