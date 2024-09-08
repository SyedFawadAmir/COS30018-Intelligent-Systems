# File: stock_prediction.py
# Authors: Bao Vo and Cheong Koo
# Date: 14/07/2021(v1); 19/07/2021 (v2); 02/07/2024 (v3)

# Code modified from:
# Title: Predicting Stock Prices with Python
# YouTube link: https://www.youtube.com/watch?v=PuZY9q-aKLw
# By: NeuralNine

# Need to install the following (best in a virtual env):
# pip install numpy
# pip install matplotlib
# pip install pandas
# pip install tensorflow
# pip install scikit-learn
# pip install pandas-datareader
# pip install yfinance
# pip install mplfinance

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
import tensorflow as tf
import mplfinance as mpf
import os
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, InputLayer
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import LSTM, GRU, RNN, Dense, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop, SGD, Adadelta
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")


#------------------------------------------------------------------------------
# Load Data
def load_data(ticker, start_date=None, end_date=None, handle_nan='drop', split_method='ratio', split_ratio=0.8, split_date=None, shuffle=True, save_local=True, file_path=None, scale_features=True):
    """
    Loads stock data from Yahoo Finance, handles NaN values, and splits the data into 
    training and testing sets using the specified method (by ratio, by date, or randomly).
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
    - y_train, y_test (pd.DataFrame): Training and testing sets for the target variable.
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

    # Splitting the data
    if split_method == 'ratio':
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - split_ratio, shuffle=shuffle)
        print(f"Data split by ratio: {split_ratio*100}% training, {(1-split_ratio)*100}% testing.")

    elif split_method == 'date':
        if split_date is None:
            raise ValueError("split_date must be provided when split_method='date'.")
        
        X_train = X.loc[:split_date]
        X_test = X.loc[split_date:]
        y_train = y.loc[:split_date]
        y_test = y.loc[split_date:]
        print(f"Data split by date: {split_date}.")

    elif split_method == 'random':
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - split_ratio, shuffle=shuffle)
        print("Data split randomly.")

    else:
        raise ValueError("Invalid split_method. Choose from 'ratio', 'date', or 'random'.")

    return X_train, X_test, y_train, y_test, scalers, raw_data

# Example usage
ticker = "AAPL"
start_date = '2015-01-01'
end_date = '2020-01-01'
split_method = 'date'  # Options: 'ratio', 'date', 'random'
split_ratio = 0.8  # Used if split_method='ratio'
split_date = '2018-01-01'  # Used if split_method='date'
file_path = "AAPL_data.csv"  # Local file path for storing/loading data
scale_features = True  # Enable feature scaling

# Call the new load_data function with scaling
X_train, X_test, y_train, y_test, scalers, raw_data = load_data(ticker, start_date, end_date, handle_nan='fill', split_method=split_method, split_ratio=split_ratio, split_date=split_date, save_local=True, file_path=file_path, scale_features=scale_features)

# printing x train and y train to check correct splitting
print(X_train)  
print(y_train)

print("Scalers for future use:", scalers)

#------------------------------------------------------------------------------
# DATA_SOURCE = "yahoo"
COMPANY = 'CBA.AX'

TRAIN_START = '2020-01-01'     # Start date to read
TRAIN_END = '2023-08-01'       # End date to read

# data = web.DataReader(COMPANY, DATA_SOURCE, TRAIN_START, TRAIN_END) # Read data using yahoo


import yfinance as yf

# Get the data for the stock AAPL
data = yf.download(COMPANY,TRAIN_START,TRAIN_END)

#------------------------------------------------------------------------------
# Prepare Data
## To do:
# 1) Check if data has been prepared before. 
# If so, load the saved data
# If not, save the data into a directory
# 2) Use a different price value eg. mid-point of Open & Close
# 3) Change the Prediction days
#------------------------------------------------------------------------------
PRICE_VALUE = "Close"

scaler = MinMaxScaler(feature_range=(0, 1)) 
# Note that, by default, feature_range=(0, 1). Thus, if you want a different 
# feature_range (min,max) then you'll need to specify it here
scaled_data = scaler.fit_transform(data[PRICE_VALUE].values.reshape(-1, 1)) 
# Flatten and normalise the data
# First, we reshape a 1D array(n) to 2D array(n,1)
# We have to do that because sklearn.preprocessing.fit_transform()
# requires a 2D array
# Here n == len(scaled_data)
# Then, we scale the whole array to the range (0,1)
# The parameter -1 allows (np.)reshape to figure out the array size n automatically 
# values.reshape(-1, 1) 
# https://stackoverflow.com/questions/18691084/what-does-1-mean-in-numpy-reshape'
# When reshaping an array, the new shape must contain the same number of elements 
# as the old shape, meaning the products of the two shapes' dimensions must be equal. 
# When using a -1, the dimension corresponding to the -1 will be the product of 
# the dimensions of the original array divided by the product of the dimensions 
# given to reshape so as to maintain the same number of elements.

# Number of days to look back to base the prediction
PREDICTION_DAYS = 60 # Original

# To store the training data
x_train = []
y_train = []

scaled_data = scaled_data[:,0] # Turn the 2D array back to a 1D array
# Prepare the data
for x in range(PREDICTION_DAYS, len(scaled_data)):
    x_train.append(scaled_data[x-PREDICTION_DAYS:x])
    y_train.append(scaled_data[x])

# Convert them into an array
x_train, y_train = np.array(x_train), np.array(y_train)
# Now, x_train is a 2D array(p,q) where p = len(scaled_data) - PREDICTION_DAYS
# and q = PREDICTION_DAYS; while y_train is a 1D array(p)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
# We now reshape x_train into a 3D array(p, q, 1); Note that x_train 
# is an array of p inputs with each input being a 2D array 

#------------------------------------------------------------------------------
# Build the Model
## TO DO:
# 1) Check if data has been built before. 
# If so, load the saved data
# If not, save the data into a directory
# 2) Change the model to increase accuracy?

#------------------------------------------------------------------------------
# Function to dynamically create a DL model
from tensorflow.keras.layers import SimpleRNNCell

def create_dl_model(layer_type, num_layers, layer_sizes, input_shape, output_size, dropout_rate=0.2, optimizer='adam', loss='mean_squared_error'):
    model = Sequential()

    # Handle RNN differently, as it needs a cell type
    if layer_type == RNN:
        first_layer = layer_type(SimpleRNNCell(layer_sizes[0]), input_shape=input_shape, return_sequences=True)
    else:
        first_layer = layer_type(units=layer_sizes[0], input_shape=input_shape, return_sequences=True)
    
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
    if layer_type == RNN:
        model.add(layer_type(SimpleRNNCell(layer_sizes[-1])))
    else:
        model.add(layer_type(units=layer_sizes[-1]))
    
    model.add(Dropout(dropout_rate))

    # Output layer
    model.add(Dense(units=output_size))

    # Compile the model
    if optimizer == 'adam':
        optimizer_instance = Adam()
    elif optimizer == 'rmsprop':
        optimizer_instance = RMSprop()
    elif optimizer == 'sgd':
        optimizer_instance = SGD()
    elif optimizer == 'adadelta':
        optimizer_instance = Adadelta()
    else:
        raise ValueError("Unsupported optimizer type")

    model.compile(optimizer=optimizer_instance, loss=loss, metrics=['mean_absolute_error'])

    return model



# Function to train the model with given configurations
def train_model(model, x_train, y_train, epochs, batch_size):
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    return history

# Load and preprocess the data (scaling, reshaping, etc.)
def load_and_prepare_data(data, feature_column='Close', prediction_days=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[feature_column].values.reshape(-1, 1))

    x_train, y_train = [], []
    for i in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[i-prediction_days:i])
        y_train.append(scaled_data[i])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))  # Reshape to be 3D for LSTM
    return x_train, y_train, scaler

# Experiment with different configurations
def experiment_with_configurations(data, layer_types, layer_configs, epochs_configs, batch_sizes, optimizers, prediction_days=60):
    x_train, y_train, scaler = load_and_prepare_data(data, prediction_days=prediction_days)

    results = []
    for layer_type in layer_types:
        for num_layers, layer_sizes in layer_configs:
            for epochs in epochs_configs:
                for batch_size in batch_sizes:
                    for optimizer in optimizers:
                        # Create the model
                        model = create_dl_model(layer_type, num_layers, layer_sizes, input_shape=(x_train.shape[1], 1), output_size=1, optimizer=optimizer)

                        # Train the model
                        history = train_model(model, x_train, y_train, epochs=epochs, batch_size=batch_size)

                        # Collect the results
                        results.append({
                            'layer_type': layer_type.__name__,
                            'num_layers': num_layers,
                            'layer_sizes': layer_sizes,
                            'epochs': epochs,
                            'batch_size': batch_size,
                            'optimizer': optimizer,
                            'loss': history.history['loss'][-1],
                            'val_loss': history.history['val_loss'][-1]
                        })

                        # Save model for reusability
                        # model.save(f'model_{layer_type.__name__}_{num_layers}layers_{epochs}epochs_{batch_size}batch_{optimizer}.h5')

    return results


# Create an LSTM model using the provided create_dl_model function
model = create_dl_model(
    layer_type=LSTM,           # Using LSTM layers
    num_layers=3,              # Number of layers in the model
    layer_sizes=[50, 50, 50],  # Sizes for each layer (as specified in the earlier example)
    input_shape=(x_train.shape[1], 1),  # Input shape as per the earlier code
    output_size=1,             # One output unit in the Dense layer
    dropout_rate=0.2,          # Dropout rate
    optimizer='adam',          # Optimizer to use
    loss='mean_squared_error'  # Loss function
)

history = train_model(
    model=model,
    x_train=x_train,           # Training data
    y_train=y_train,           # Target data
    epochs=25,                 # Number of epochs to train the model
    batch_size=32              # Batch size
)

#------------------------------------------------------------------------------
# Test the model accuracy on existing data
#------------------------------------------------------------------------------
# Load the test data
TEST_START = '2023-08-02'
TEST_END = '2024-07-02'

# test_data = web.DataReader(COMPANY, DATA_SOURCE, TEST_START, TEST_END)

test_data = yf.download(COMPANY,TEST_START,TEST_END)


# The above bug is the reason for the following line of code
# test_data = test_data[1:]

actual_prices = test_data[PRICE_VALUE].values

total_dataset = pd.concat((data[PRICE_VALUE], test_data[PRICE_VALUE]), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - PREDICTION_DAYS:].values
# We need to do the above because to predict the closing price of the first
# PREDICTION_DAYS of the test period [TEST_START, TEST_END], we'll need the 
# data from the training period

model_inputs = model_inputs.reshape(-1, 1)
# TO DO: Explain the above line

model_inputs = scaler.transform(model_inputs)
# We again normalize our closing price data to fit them into the range (0,1)
# using the same scaler used above 
# However, there may be a problem: scaler was computed on the basis of
# the Max/Min of the stock price for the period [TRAIN_START, TRAIN_END],
# but there may be a lower/higher price during the test period 
# [TEST_START, TEST_END]. That can lead to out-of-bound values (negative and
# greater than one)
# We'll call this ISSUE #2

# TO DO: Generally, there is a better way to process the data so that we 
# can use part of it for training and the rest for testing. You need to 
# implement such a way

#------------------------------------------------------------------------------
# Make predictions on test data
#------------------------------------------------------------------------------
x_test = []
for x in range(PREDICTION_DAYS, len(model_inputs)):
    x_test.append(model_inputs[x - PREDICTION_DAYS:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
# TO DO: Explain the above 5 lines

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)
# Clearly, as we transform our data into the normalized range (0,1),
# we now need to reverse this transformation 
#------------------------------------------------------------------------------
# Plot the test predictions
plt.plot(actual_prices, color="black", label=f"Actual {COMPANY} Price")
plt.plot(predicted_prices, color="green", label=f"Predicted {COMPANY} Price")
plt.title(f"{COMPANY} Share Price")
plt.xlabel("Time")
plt.ylabel(f"{COMPANY} Share Price")
plt.legend()
plt.show()

#------------------------------------------------------------------------------
# Candlestick Chart Function
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

#------------------------------------------------------------------------------
# Boxplot Chart Function
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
# Generate the Candlestick Chart using data from load_data function
plot_candlestick_chart(raw_data, n=5, title=f'{COMPANY} Candlestick Chart', save=False)

# Generate the Boxplot Chart using data from load_data function
plot_boxplot_chart(raw_data, n=10, title=f'{COMPANY} Boxplot Chart', save=False)

#------------------------------------------------------------------------------
# Predict next day
real_data = [model_inputs[len(model_inputs) - PREDICTION_DAYS:, 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(f"Prediction: {prediction}")

# Define your experiment configurations
layer_types = [LSTM, GRU, RNN]  # Test different types of layers
layer_configs = [(2, [64, 32]), (3, [128, 64, 32])]  # Different layer configurations (num_layers, layer_sizes)
epochs_configs = [10, 25]  # Number of epochs
batch_sizes = [32, 64]  # Batch sizes
optimizers = ['adam', 'rmsprop']  # Different optimizers

# Call the experiment function
results = experiment_with_configurations(data, layer_types, layer_configs, epochs_configs, batch_sizes, optimizers, prediction_days=60)

# Print results for analysis
for result in results:
    print(result)


# A few concluding remarks here:
# 1. The predictor is quite bad, especially if you look at the next day 
# prediction, it missed the actual price by about 10%-13%
# Can you find the reason?
# 2. The code base at
# https://github.com/x4nth055/pythoncode-tutorials/tree/master/machine-learning/stock-prediction
# gives a much better prediction. Even though on the surface, it didn't seem 
# to be a big difference (both use Stacked LSTM)
# Again, can you explain it?
# A more advanced and quite different technique use CNN to analyse the images
# of the stock price changes to detect some patterns with the trend of
# the stock price:
# https://github.com/jason887/Using-Deep-Learning-Neural-Networks-and-Candlestick-Chart-Representation-to-Predict-Stock-Market
# Can you combine these different techniques for a better prediction??