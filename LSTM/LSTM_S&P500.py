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
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from keras.metrics import MeanSquaredError
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD


plt.style.use('fivethirtyeight')
    
df = web.DataReader('^N100', data_source = 'stooq', start = '2015-01-01', end = '2021-01-05' )
print('Number of rows and columns: ', df.shape)
print(df.head())
print("checking if any null values are present\n", df.isna().sum())
    
plt.figure(figsize = (12,6))
plt.plot(df["Open"])
plt.plot(df["High"])
plt.plot(df["Low"])
plt.plot(df["Close"])
plt.title(' stock price history')
plt.ylabel('Price (USD)')
plt.xlabel('Days')
plt.legend(['Open','High','Low','Close'], loc='upper left')
plt.show()

plt.figure(figsize = (12,6))
plt.plot(df["Volume"])
plt.title('stock volume history')
plt.ylabel('Volume')
plt.xlabel('Days')
plt.show()

# Create a dataframe with only the Close Stock Price Column
data_target = df.filter(['Close'])
    
# Convert the dataframe to a numpy array to train the LSTM model
target = data_target.values
    
# Splitting the dataset into training and test
# Target Variable: Close stock price value
    
training_data_len = math.ceil(len(target)* 0.75) # training set has 75% of the data
training_data_len
    
    # Normalizing data before model fitting using MinMaxScaler
    # Feature Scaling
    
sc = MinMaxScaler(feature_range=(0,1))
training_scaled_data = sc.fit_transform(target)
training_scaled_data
    
# Create a training dataset containing the last 180-day closing price values we want to use to estimate the 181st closing price value.
train_data = training_scaled_data[0:training_data_len  , : ]
    
X_train = []
y_train = []
for i in range(180, len(train_data)):
    X_train.append(train_data[i-180:i, 0])
    y_train.append(train_data[i, 0])
    
X_train, y_train = np.array(X_train), np.array(y_train) # converting into numpy sequences to train the LSTM model
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
print('Number of rows and columns: ', X_train.shape)  #(854 values, 180 time-steps, 1 output)

# We add the LSTM layer and later add a few Dropout layers to prevent overfitting.
# Building a LTSM model with 50 neurons and 4 hidden layers. We add the LSTM layer with the following arguments:
# 50 units which is the dimensionality of the output space
# return_sequences=True which determines whether to return the last output in the output sequence, or the full sequence input_shape as the shape of our training set.
# When defining the Dropout layers, we specify 0.2, meaning that 20% of the layers will be dropped.
# Thereafter, we add the Dense layer that specifies the output of 1 unit.
# After this, we compile our model using the popular adam optimizer and set the loss as the mean_squarred_error.

# Testing for improving 
# 
# 1 & 2: 100 epochs to 50 -- epochs WORSE
# 2 & 3: added on fourth node/hidden layer -- WORSE BY FAR
# 3 & 4: removed both nodes to two nodes in total -- BEST OVERALL RESULT SO FAR
# 4 & 5: only one node/ layer used now -- 

model = Sequential()

#Adding the first LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1), activation='tanh'))
model.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
#model.add(LSTM(units = 50, return_sequences = True), activation='tanh')
#model.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
#model.add(LSTM(units = 50, return_sequences = True), activation='tanh')
#model.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50, activation='tanh'))
model.add(Dropout(0.2))

# Adding the output layer
model.add(Dense(units = 1))

# Compiling the RNN
model.compile(optimizer=SGD(lr=0.01, decay=1e-7, momentum=0.9, nesterov=False),loss='mean_squared_error')

# Fitting the RNN to the Training set
model.fit(X_train, y_train, epochs =100, batch_size = 64, verbose = 0)

# Getting the predicted stock price
test_data = training_scaled_data[training_data_len - 180: , : ]

m = MeanSquaredError() 

#Create the x_test and y_test data sets
X_test = []
y_test =  target[training_data_len : , : ]
for i in range(180,len(test_data)):
    X_test.append(test_data[i-180:i,0])

# Convert x_test to a numpy array
X_test = np.array(X_test)

#Reshape the data into the shape accepted by the LSTM
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
print('Number of rows and columns: ', X_test.shape)

# Making predictions using the test dataset
predicted_stock_price = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
predicted_stock = np.reshape(predicted_stock_price,(384,))

# Creating budget value used for trading strategy 378
budget = 1000
for x in range(0,383):
   if x+1 > len(predicted_stock):
        pass
   else:
        budget = budget * (predicted_stock[x+1]/predicted_stock[x]) 
        print ("Day %s : %d"% (x,budget))
    
profit = budget - 1000
print("Made profit/loss of %s" % (profit))
print(m.result().numpy())
# Visualising the results
train = data_target[:training_data_len]
valid = data_target[training_data_len:]
valid['Predictions'] = predicted_stock_price
plt.figure(figsize=(10,5))
plt.title('Model')
plt.xlabel('Date', fontsize=8)
plt.ylabel('Close Price USD ($)', fontsize=12)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()

m.update_state(np.array(valid['Close']),np.array(valid['Predictions']))
print(m.result().numpy())

