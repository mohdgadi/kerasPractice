import pandas as pd 
from keras.models import Sequential
from keras.layers import *

model = Sequential()

model.add(Dense(30, input_dim=9, activation="relu"))
model.add(Dense(60, activation="relu"))
model.add(Dense(30, activation="relu"))
model.add(Dense(1, activation="linear"))
model.compile(optimzer='adam', loss='mse')

