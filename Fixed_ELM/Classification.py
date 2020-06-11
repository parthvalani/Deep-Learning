# import all the neccesary libraries
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from ELM import ELMClassifier

# data-preprocessing on UCI-iris dataset
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

# scaling  input data
data = StandardScaler()
X_train = data.fit_transform(X_train)
X_test = data.transform(X_test)

# scaling target data
target = StandardScaler()
y_train = target.fit_transform(np.expand_dims(y_train,1))
y_test = target.transform(np.expand_dims(y_test,1))  

max_y_train = max(abs(y_train))
y_train = y_train / max_y_train
y_test = y_test / max_y_train

# elm training for classification
elmc = ELMClassifier(500)
elmc.fit(X_train, y_train)

# make prediction
pred = elmc.predict(X_test)
mse = mean_squared_error(y_test, pred)

# Calculating accuracy
acc = 1- mse
print("Testing accuracy:",(acc*100))
