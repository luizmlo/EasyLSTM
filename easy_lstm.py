import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

class EasyLSTM():
    def __init__(self, data, n_steps=4, train_size=0.8):
        self.data = data
        self.n_steps = n_steps
        self.train_size = train_size


    def build_model(self):
        model = Sequential()
        model.add(LSTM(self.n_features*16, return_sequences=True, input_shape=(self.n_steps, self.n_features)))
        model.add(Dropout(0.2))
        model.add(LSTM(self.n_features*8))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        return model


    def slice_data(self, data, n_steps, n_ahead=1):
        X_data = data.drop(['y'], axis=1)
        y_data = data['y']
        X = []
        y = []
        for i in range(len(X_data)):
            end_point = i + n_steps
            if end_point + n_ahead > len(X_data)-1:
                break
            slice_x, slice_y = X_data[i:end_point], y_data.loc[end_point]
            X.append(slice_x)
            y.append(slice_y)
        X, y = np.array(X), np.array(y)
        self.n_features = X.shape[2]
        X = X.reshape((X.shape[0], X.shape[1], X.shape[2]))
        return X, y


    def train_test_split(self, X, y):
        slice_size = int(self.train_size*len(X))

        X_train = X[:slice_size]
        X_test = X[slice_size:]
        y_train = y[:slice_size]
        y_test = y[slice_size:]

        return X_train, y_train, X_test, y_test

    def do_magic(self):
        X, y = self.slice_data(self.data, self.n_steps)
        X_train, y_train, X_test, y_test = self.train_test_split(X, y)
        model = self.build_model()

        return model, X_train, y_train, X_test, y_test
