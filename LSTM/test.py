#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 13:24:25 2022

@author: victor
"""

import math
import pandas_datareader as web
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.metrics import MeanSquaredError
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD

plt.style.use('fivethirtyeight')

# Load the stock data
df = web.DataReader('^N100', data_source='stooq', start='2015-01-01', end='2021-01-05')
print('Number of rows and columns: ', df.shape)
print(df.head())
print("Checking if any null values are present\n", df.isna().sum())

# Plot stock prices
plt.figure(figsize=(12, 6))
plt.plot(df["Open"])
plt.plot(df["High"])
plt.plot(df["Low"])
plt.plot(df["Close"])
plt.title('Stock Price History')
plt.ylabel('Price (USD)')
plt.xlabel('Days')
plt.legend(['Open', 'High', 'Low', 'Close'], loc='upper left')
plt.show()

# Plot volume
plt.figure(figsize=(12, 6))
plt.plot(df["Volume"])
plt.title('Stock Volume History')
plt.ylabel('Volume')
plt.xlabel('Days')
plt.show()

# Prepare target data
data_target = df.filter(['Close'])
target = data_target.values

# Split dataset
training_data_len = math.ceil(len(target) * 0.75)

# Normalize data
sc = MinMaxScaler(feature_range=(0, 1))
training_scaled_data = sc.fit_transform(target)

# Create training set
train_data = training_scaled_data[0:training_data_len, :]
X_train = []
y_train = []

for i in range(180, len(train_data)):
    X_train.append(train_data[i - 180:i, 0])
    y_train.append(train_data[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
print('Training shape:', X_train.shape)

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1), activation='tanh'))
model.add(Dropout(0.2))
model.add(LSTM(units=50, activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(units=1))

# Compile the model
optimizer = SGD(learning_rate=0.01, decay=1e-7, momentum=0.9, nesterov=False)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=64, verbose=0)

# Prepare test data
test_data = training_scaled_data[training_data_len - 180:, :]
X_test = []
y_test = target[training_data_len:, :]

for i in range(180, len(test_data)):
    X_test.append(test_data[i - 180:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
print('Test shape:', X_test.shape)

# Make predictions
predicted_stock_price = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
predicted_stock = np.reshape(predicted_stock_price, (-1,))

# Simulated trading strategy
budget = 1000
for x in range(len(predicted_stock) - 1):
    budget = budget * (predicted_stock[x + 1] / predicted_stock[x])
    print("Day %s : %d" % (x, budget))

profit = budget - 1000
print("Made profit/loss of %s" % (profit))

# Visualize results
train = data_target[:training_data_len]
valid = data_target[training_data_len:].copy()
valid['Predictions'] = predicted_stock_price

plt.figure(figsize=(10, 5))
plt.title('Model')
plt.xlabel('Date', fontsize=8)
plt.ylabel('Close Price USD ($)', fontsize=12)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()

# Evaluate model
m = MeanSquaredError()
m.update_state(np.array(valid['Close']), np.array(valid['Predictions']))
print("Final MSE:", m.result().numpy())
