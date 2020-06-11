
# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

######################## Data Preprocessing #########################

# the training set
train_dataset = pd.read_csv('Data/Google_Stock_Price_Train.csv')
train_data = train_dataset.iloc[:, 1:2].values

# Feature Rescaling using minmaxscaling
from sklearn.preprocessing import MinMaxScaler
x = MinMaxScaler(feature_range = (0, 1))
scaled_data = x.fit_transform(train_data)

# Creating a data structure with 60 timesteps and 1 output
train_x = []
train_y = []
for i in range(60, 1258):
    train_x.append(scaled_data[i-60:i, 0])
    train_y.append(scaled_data[i, 0])
train_x, train_y = np.array(train_x), np.array(train_y)

# Reshaping
train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))

######################### Building the RNN ##########################

model = Sequential()
model.add(LSTM(100, dropout=0.2,return_sequences = True, input_shape = (train_x.shape[1], 1)))
model.add(LSTM(100, dropout=0.2,return_sequences = True))
model.add(LSTM(100, dropout=0.2, return_sequences = True))
model.add(LSTM(100, dropout=0.2))
model.add(Dense(1))

# Compiling the RNN
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN
model.fit(train_x, train_y, epochs = 150, batch_size = 256)

####### Making the predictions and visualising the results ########

test_dataset = pd.read_csv('Data/Google_Stock_Price_Test.csv')
y_true = test_dataset.iloc[:, 1:2].values

# Getting the predicted stock price
dataset = pd.concat((train_dataset['Open'], test_dataset['Open']), axis = 0)
inputs = dataset[len(dataset) - len(test_dataset) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = x.transform(inputs)
test_x = []
for i in range(60, 80):
    test_x.append(inputs[i-60:i, 0])
test_x = np.array(test_x)
test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))
y_predicted = model.predict(test_x)
y_predicted = x.inverse_transform(y_predicted)

# Visualization of real vs predicted price
plt.plot(y_true, color = 'blue', label = 'Real Price')
plt.plot(y_predicted, color = 'Red', label = 'Predicted Price')
plt.title('Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()