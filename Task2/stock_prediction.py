# File: stock_prediction.py
# Authors: Bao Vo and Cheong Koo
# Date: 14/07/2021(v1); 19/07/2021 (v2); 02/07/2024 (v3)

# Code modified from:
# Title: Predicting Stock Prices with Python
# Youtuble link: https://www.youtube.com/watch?v=PuZY9q-aKLw
# By: NeuralNine

# Need to install the following (best in a virtual env):
# pip install numpy
# pip install matplotlib
# pip install pandas
# pip install tensorflow
# pip install scikit-learn
# pip install pandas-datareader
# pip install yfinance

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, InputLayer
from sklearn.model_selection import train_test_split
import yfinance as yf
import os

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
    - split_date (str): Date for splitting the data if `split_method='date'`. Defaults to None.
    - shuffle (bool): Whether to shuffle the data before splitting. Defaults to True.
    - save_local (bool): If True, the data will be saved locally. Defaults to True.
    - file_path (str): Path to the file where the data will be saved/loaded. Defaults to None.
    - stores the scalers for future use. Uses multiple features (Open, High, Low, Volume, AdjClose).
    Parameters:
    - scale_features (bool): If True, scale the feature columns. Defaults to True.
    Other parameters remain the same.

    Returns:
    - X_train, X_test (pd.DataFrame): Training and testing sets for features.
    - y_train, y_test (pd.DataFrame): Training and testing sets for the target variable.
    - scalers (dict): A dictionary storing the scalers for each feature column (if scaled).
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

    return X_train, X_test, y_train, y_test, scalers

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
X_train, X_test, y_train, y_test, scalers = load_data(ticker, start_date, end_date, handle_nan='fill', split_method=split_method, split_ratio=split_ratio, split_date=split_date, save_local=True, file_path=file_path, scale_features=scale_features)

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
model = Sequential() # Basic neural network
# See: https://www.tensorflow.org/api_docs/python/tf/keras/Sequential
# for some useful examples

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
# This is our first hidden layer which also spcifies an input layer. 
# That's why we specify the input shape for this layer; 
# i.e. the format of each training example
# The above would be equivalent to the following two lines of code:
# model.add(InputLayer(input_shape=(x_train.shape[1], 1)))
# model.add(LSTM(units=50, return_sequences=True))
# For som eadvances explanation of return_sequences:
# https://machinelearningmastery.com/return-sequences-and-return-states-for-lstms-in-keras/
# https://www.dlology.com/blog/how-to-use-return_state-or-return_sequences-in-keras/
# As explained there, for a stacked LSTM, you must set return_sequences=True 
# when stacking LSTM layers so that the next LSTM layer has a 
# three-dimensional sequence input. 

# Finally, units specifies the number of nodes in this layer.
# This is one of the parameters you want to play with to see what number
# of units will give you better prediction quality (for your problem)

model.add(Dropout(0.2))
# The Dropout layer randomly sets input units to 0 with a frequency of 
# rate (= 0.2 above) at each step during training time, which helps 
# prevent overfitting (one of the major problems of ML). 

model.add(LSTM(units=50, return_sequences=True))
# More on Stacked LSTM:
# https://machinelearningmastery.com/stacked-long-short-term-memory-networks/

model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))

model.add(Dense(units=1)) 
# Prediction of the next closing value of the stock price

# We compile the model by specify the parameters for the model
# See lecture Week 6 (COS30018)
model.compile(optimizer='adam', loss='mean_squared_error')
# The optimizer and loss are two important parameters when building an 
# ANN model. Choosing a different optimizer/loss can affect the prediction
# quality significantly. You should try other settings to learn; e.g.
    
# optimizer='rmsprop'/'sgd'/'adadelta'/...
# loss='mean_absolute_error'/'huber_loss'/'cosine_similarity'/...

# Now we are going to train this model with our training data 
# (x_train, y_train)
model.fit(x_train, y_train, epochs=25, batch_size=32)
# Other parameters to consider: How many rounds(epochs) are we going to 
# train our model? Typically, the more the better, but be careful about
# overfitting!
# What about batch_size? Well, again, please refer to 
# Lecture Week 6 (COS30018): If you update your model for each and every 
# input sample, then there are potentially 2 issues: 1. If you training 
# data is very big (billions of input samples) then it will take VERY long;
# 2. Each and every input can immediately makes changes to your model
# (a souce of overfitting). Thus, we do this in batches: We'll look at
# the aggreated errors/losses from a batch of, say, 32 input samples
# and update our model based on this aggregated loss.

# TO DO:
# Save the model and reload it
# Sometimes, it takes a lot of effort to train your model (again, look at
# a training data with billions of input samples). Thus, after spending so 
# much computing power to train your model, you may want to save it so that
# in the future, when you want to make the prediction, you only need to load
# your pre-trained model and run it on the new input for which the prediction
# need to be made.

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
# We need to do the above because to predict the closing price of the fisrt
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
## To do:
# 1) Candle stick charts
# 2) Chart showing High & Lows of the day
# 3) Show chart of next few days (predicted)
#------------------------------------------------------------------------------

plt.plot(actual_prices, color="black", label=f"Actual {COMPANY} Price")
plt.plot(predicted_prices, color="green", label=f"Predicted {COMPANY} Price")
plt.title(f"{COMPANY} Share Price")
plt.xlabel("Time")
plt.ylabel(f"{COMPANY} Share Price")
plt.legend()
plt.show()

#------------------------------------------------------------------------------
# Predict next day
#------------------------------------------------------------------------------


real_data = [model_inputs[len(model_inputs) - PREDICTION_DAYS:, 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(f"Prediction: {prediction}")

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