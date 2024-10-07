# File: stock_prediction_multiple_ensembles.py
# Date: 2024-04-27

# Required installations (best in a virtual environment):
# pip install numpy matplotlib pandas tensorflow scikit-learn yfinance statsmodels

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import yfinance as yf
import datetime as dt
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, SimpleRNN
from tensorflow.keras.optimizers import Adam, RMSprop
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ------------------------------------------------------------------------------
# Load Data Function
def load_data(ticker, start_date, end_date, split_date, scale_features=True):
    """
    Loads stock data from Yahoo Finance, handles NaN values, scales features, and splits the data.

    Parameters:
    - ticker (str): The ticker symbol of the stock (e.g., "AAPL").
    - start_date (str): Start date for data in the format 'YYYY-MM-DD'.
    - end_date (str): End date for data in the format 'YYYY-MM-DD'.
    - split_date (str): Date for splitting the data into training and testing sets.
    - scale_features (bool): If True, scales the features and target variable.

    Returns:
    - X_train, X_test (pd.DataFrame): Training and testing sets for features.
    - y_train, y_test (np.ndarray): Training and testing sets for the target variable.
    - scalers (dict): A dictionary storing the scalers for each feature column (if scaled).
    - raw_data (pd.DataFrame): The raw stock data used for plotting charts.
    """
    # Download data from Yahoo Finance
    raw_data = yf.download(ticker, start=start_date, end=end_date)

    if raw_data.empty:
        raise ValueError(f"No data found for {ticker} between {start_date} and {end_date}.")

    # Fill NaN values with column mean
    raw_data = raw_data.fillna(raw_data.mean())
    print("Filled NaN values with column mean.")

    # Features and target variable (using 'Close' as the target)
    X = raw_data[['Open', 'High', 'Low', 'Volume', 'Adj Close']]  # Use relevant features
    y = raw_data['Close']  # Target variable

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

    # Splitting the data by date
    X_train = X.loc[:split_date]
    X_test = X.loc[split_date:]
    y_train = y[:len(X_train)]
    y_test = y[len(X_train):]
    print(f"Data split by date: {split_date}.")

    return X_train, X_test, y_train, y_test, scalers, raw_data

# ------------------------------------------------------------------------------
# Prepare Data Functions
def create_sequences(X, y, time_steps=60):
    """
    Creates input-output sequences for training/testing.

    Parameters:
    - X (np.ndarray): Feature data.
    - y (np.ndarray): Target data.
    - time_steps (int): Number of past time steps to consider.

    Returns:
    - Xs (np.ndarray): Input sequences.
    - ys (np.ndarray): Output values.
    """
    Xs, ys = [], []
    for i in range(time_steps, len(X)):
        Xs.append(X[i-time_steps:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

# ------------------------------------------------------------------------------
# Model Creation Functions
def create_rnn_model(layer_type, input_shape, units=50, optimizer='adam'):
    """
    Creates an RNN model based on the specified layer type.

    Parameters:
    - layer_type: Type of RNN layer (LSTM, GRU, SimpleRNN).
    - input_shape (tuple): Shape of the input data (time_steps, features).
    - units (int): Number of units in the RNN layer.
    - optimizer (str): Optimizer to use (e.g., 'adam', 'rmsprop').

    Returns:
    - model: Compiled Keras model.
    """
    model = Sequential()
    model.add(layer_type(units=units, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

# ------------------------------------------------------------------------------
# Training and Prediction Functions
def train_rnn_model(model, x_train, y_train, epochs=10, batch_size=32):
    """
    Trains the given RNN model on the training data.

    Parameters:
    - model: Keras model to train.
    - x_train (np.ndarray): Training input sequences.
    - y_train (np.ndarray): Training target sequences.
    - epochs (int): Number of training epochs.
    - batch_size (int): Batch size for training.

    Returns:
    - history: Training history.
    """
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    return history

def predict_rnn_model(model, x_test):
    """
    Makes predictions using the trained RNN model.

    Parameters:
    - model: Trained RNN model.
    - x_test (np.ndarray): Testing input sequences.

    Returns:
    - predictions (np.ndarray): Predicted values.
    """
    predictions = model.predict(x_test)
    return predictions

def train_arima_model(train_data, order=(5,1,0)):
    """
    Trains an ARIMA model on the training data.

    Parameters:
    - train_data (np.ndarray): Training target data (inverse-scaled).

    Returns:
    - model_fit: Fitted ARIMA model.
    """
    model = ARIMA(train_data, order=order)
    model_fit = model.fit()
    print(f"ARIMA model trained with order={order}")
    return model_fit

def predict_arima_model(model_fit, steps):
    """
    Makes predictions using the trained ARIMA model.

    Parameters:
    - model_fit: Trained ARIMA model.
    - steps (int): Number of future steps to predict.

    Returns:
    - predictions (np.ndarray): Predicted values.
    """
    predictions = model_fit.forecast(steps=steps)
    return predictions

def train_random_forest_model(x_train, y_train, n_estimators=100, max_depth=10):
    """
    Trains a Random Forest Regressor on the training data.

    Parameters:
    - x_train (np.ndarray): Training features.
    - y_train (np.ndarray): Training target.
    - n_estimators (int): Number of trees in the forest.
    - max_depth (int): Maximum depth of the tree.

    Returns:
    - rf_model: Trained Random Forest model.
    """
    rf_model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    rf_model.fit(x_train, y_train)
    print("Random Forest model trained.")
    return rf_model

def predict_random_forest_model(rf_model, x_test):
    """
    Makes predictions using the trained Random Forest model.

    Parameters:
    - rf_model: Trained Random Forest model.
    - x_test (np.ndarray): Testing features.

    Returns:
    - predictions (np.ndarray): Predicted values.
    """
    predictions = rf_model.predict(x_test)
    return predictions

# ------------------------------------------------------------------------------
# Ensemble Methods
def ensemble_average(predictions_list):
    """
    Simple averaging ensemble method.

    Parameters:
    - predictions_list (list of np.ndarray): List of prediction arrays from different models.

    Returns:
    - ensemble_pred (np.ndarray): Averaged predictions.
    """
    ensemble_pred = np.mean(predictions_list, axis=0)
    return ensemble_pred

# ------------------------------------------------------------------------------
# Visualization Functions
def plot_predictions(actual, prediction, title='Model Predictions vs Actual'):
    """
    Plots actual vs. predicted values.

    Parameters:
    - actual (np.ndarray): Actual target values.
    - prediction (np.ndarray): Predicted values.
    - title (str): Title of the plot.

    Returns:
    - None
    """
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label='Actual', color='black', linestyle='--')
    plt.plot(prediction, label='Predicted')
    plt.title(title)
    plt.xlabel('Time Steps')
    plt.ylabel('Closing Price')
    plt.legend()
    plt.grid(True)
    plt.show()

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
    import mplfinance as mpf
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

    plt.figure(figsize=(14, 7))
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

# ------------------------------------------------------------------------------
# Main Execution
if __name__ == "__main__":
    # --------------------- Parameters ---------------------
    ticker = "AAPL"
    start_date = '2010-01-01'
    end_date = '2020-01-01'
    split_date = '2018-01-01'  # Used to split data into training and testing
    TIME_STEPS = 60  # Number of past days to use for prediction

    # --------------------- Load Data ---------------------
    X_train, X_test, y_train, y_test, scalers, raw_data = load_data(
        ticker, start_date, end_date, split_date, scale_features=True)

    # --------------------- Prepare Data ---------------------
    X_train_values = X_train.values
    y_train_values = y_train.flatten()
    X_test_values = X_test.values
    y_test_values = y_test.flatten()

    x_train, y_train_seq = create_sequences(X_train_values, y_train_values, TIME_STEPS)
    x_test, y_test_seq = create_sequences(X_test_values, y_test_values, TIME_STEPS)

    # Inverse transform y_test_seq for evaluation
    y_test_rescaled = scalers['target'].inverse_transform(y_test_seq.reshape(-1, 1)).flatten()

    # --------------------- Model Ensembles ---------------------
    ensemble_configs = [
        {'name': 'ARIMA + LSTM', 'models': ['arima', 'lstm']},
        {'name': 'ARIMA + Random Forest', 'models': ['arima', 'random_forest']},
        {'name': 'Random Forest + LSTM', 'models': ['random_forest', 'lstm']},
        {'name': 'ARIMA + Simple RNN', 'models': ['arima', 'simple_rnn']},
        {'name': 'ARIMA + GRU', 'models': ['arima', 'gru']}
    ]

    # Hyperparameters for models
    # ARIMA Orders to experiment with
    arima_orders = [(5, 1, 0), (3, 1, 2)]

    # LSTM Hyperparameters
    lstm_units = [50, 100]
    lstm_epochs = [10]
    lstm_batch_size = [32]

    # GRU Hyperparameters
    gru_units = [50, 100]
    gru_epochs = [10]
    gru_batch_size = [32]

    # Random Forest Hyperparameters
    rf_n_estimators = [100]
    rf_max_depth = [10]

    # Simple RNN Hyperparameters
    rnn_units = [50]
    rnn_epochs = [10]
    rnn_batch_size = [32]

    # Function to select hyperparameters based on model type
    def get_hyperparameters(model_name, config_idx):
        """
        Returns hyperparameters for a given model and configuration index.

        Parameters:
        - model_name (str): Name of the model ('arima', 'lstm', 'gru', 'simple_rnn', 'random_forest').
        - config_idx (int): Index to select a hyperparameter set.

        Returns:
        - dict: Dictionary of hyperparameters.
        """
        if model_name == 'arima':
            return {'order': arima_orders[config_idx % len(arima_orders)]}
        elif model_name == 'lstm':
            return {
                'units': lstm_units[config_idx % len(lstm_units)],
                'epochs': lstm_epochs[0],
                'batch_size': lstm_batch_size[0]
            }
        elif model_name == 'gru':
            return {
                'units': gru_units[config_idx % len(gru_units)],
                'epochs': gru_epochs[0],
                'batch_size': gru_batch_size[0]
            }
        elif model_name == 'simple_rnn':
            return {
                'units': rnn_units[0],
                'epochs': rnn_epochs[0],
                'batch_size': rnn_batch_size[0]
            }
        elif model_name == 'random_forest':
            return {
                'n_estimators': rf_n_estimators[0],
                'max_depth': rf_max_depth[0]
            }
        else:
            return {}

    for ensemble in ensemble_configs:
        ensemble_name = ensemble['name']
        models = ensemble['models']
        predictions_list = []
        hyperparameter_sets = []

        print(f"\nTraining Ensemble: {ensemble_name}")

        # Determine the configuration index based on ensemble name for hyperparameter selection
        config_idx = ensemble_configs.index(ensemble)

        # Train and predict with each model in the ensemble
        for model_name in models:
            if model_name == 'arima':
                # Select ARIMA hyperparameters
                arima_params = get_hyperparameters('arima', config_idx)
                arima_order = arima_params['order']
                # Prepare ARIMA data (inverse-scaled)
                y_train_arima = scalers['target'].inverse_transform(y_train_values.reshape(-1, 1)).flatten()
                arima_model_fit = train_arima_model(y_train_arima, order=arima_order)
                steps = len(y_test_rescaled)
                arima_pred = predict_arima_model(arima_model_fit, steps=steps)
                predictions_list.append(arima_pred)
            elif model_name == 'lstm':
                # Select LSTM hyperparameters
                lstm_params = get_hyperparameters('lstm', config_idx)
                lstm_units_selected = lstm_params['units']
                lstm_epochs_selected = lstm_params['epochs']
                lstm_batch_size_selected = lstm_params['batch_size']
                # Create and train LSTM model
                lstm_model = create_rnn_model(LSTM, input_shape=(x_train.shape[1], x_train.shape[2]),
                                              units=lstm_units_selected, optimizer='adam')
                train_rnn_model(lstm_model, x_train, y_train_seq,
                                epochs=lstm_epochs_selected, batch_size=lstm_batch_size_selected)
                # Predict on test data
                lstm_pred_scaled = predict_rnn_model(lstm_model, x_test)
                lstm_pred = scalers['target'].inverse_transform(lstm_pred_scaled).flatten()
                predictions_list.append(lstm_pred)
            elif model_name == 'gru':
                # Select GRU hyperparameters
                gru_params = get_hyperparameters('gru', config_idx)
                gru_units_selected = gru_params['units']
                gru_epochs_selected = gru_params['epochs']
                gru_batch_size_selected = gru_params['batch_size']
                # Create and train GRU model
                gru_model = create_rnn_model(GRU, input_shape=(x_train.shape[1], x_train.shape[2]),
                                            units=gru_units_selected, optimizer='adam')
                train_rnn_model(gru_model, x_train, y_train_seq,
                                epochs=gru_epochs_selected, batch_size=gru_batch_size_selected)
                # Predict on test data
                gru_pred_scaled = predict_rnn_model(gru_model, x_test)
                gru_pred = scalers['target'].inverse_transform(gru_pred_scaled).flatten()
                predictions_list.append(gru_pred)
            elif model_name == 'simple_rnn':
                # Select Simple RNN hyperparameters
                rnn_params = get_hyperparameters('simple_rnn', config_idx)
                rnn_units_selected = rnn_params['units']
                rnn_epochs_selected = rnn_params['epochs']
                rnn_batch_size_selected = rnn_params['batch_size']
                # Create and train Simple RNN model
                rnn_model = create_rnn_model(SimpleRNN, input_shape=(x_train.shape[1], x_train.shape[2]),
                                            units=rnn_units_selected, optimizer='adam')
                train_rnn_model(rnn_model, x_train, y_train_seq,
                                epochs=rnn_epochs_selected, batch_size=rnn_batch_size_selected)
                # Predict on test data
                rnn_pred_scaled = predict_rnn_model(rnn_model, x_test)
                rnn_pred = scalers['target'].inverse_transform(rnn_pred_scaled).flatten()
                predictions_list.append(rnn_pred)
            elif model_name == 'random_forest':
                # Select Random Forest hyperparameters
                rf_params = get_hyperparameters('random_forest', config_idx)
                rf_n_estimators_selected = rf_params['n_estimators']
                rf_max_depth_selected = rf_params['max_depth']
                # Prepare data by flattening sequences
                x_train_rf = x_train.reshape((x_train.shape[0], -1))
                x_test_rf = x_test.reshape((x_test.shape[0], -1))
                y_train_rf = y_train_seq.flatten()
                # Train Random Forest model
                rf_model = train_random_forest_model(x_train_rf, y_train_rf,
                                                     n_estimators=rf_n_estimators_selected,
                                                     max_depth=rf_max_depth_selected)
                # Predict on test data
                rf_pred_scaled = predict_random_forest_model(rf_model, x_test_rf)
                rf_pred = scalers['target'].inverse_transform(rf_pred_scaled.reshape(-1,1)).flatten()
                predictions_list.append(rf_pred)
            else:
                print(f"Model {model_name} is not supported.")

        # Create Ensemble Prediction
        ensemble_pred = ensemble_average(predictions_list)

        # Evaluate Ensemble
        mse = mean_squared_error(y_test_rescaled, ensemble_pred)
        mae = mean_absolute_error(y_test_rescaled, ensemble_pred)
        print(f"{ensemble_name} - MSE: {mse:.4f}, MAE: {mae:.4f}")

        # Plot Predictions
        plot_predictions(y_test_rescaled, ensemble_pred, title=f'{ensemble_name} - Actual vs Predicted')

    # --------------------- Additional Visualizations ---------------------
    # Generate Candlestick Chart
    plot_candlestick_chart(raw_data, n=5, title=f'{ticker} Candlestick Chart', save=False)

    # Generate Boxplot Chart
    plot_boxplot_chart(raw_data, n=10, title=f'{ticker} Boxplot Chart', save=False)
