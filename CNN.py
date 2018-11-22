# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 12:36:42 2018

@author: Can Jozef Saul
"""

import pandas as pd
import numpy as np

data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data', header=None)

x_data_only_gender = data.iloc[:, 0]
x_data_toFS = data.iloc[:, 1:8]

y_data = data.iloc[:, 8].values

# X data proprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
x_data_only_gender = le.fit_transform(x_data_only_gender)

OHE_X = OneHotEncoder(categorical_features = 'all')
x_data_only_gender = x_data_only_gender.reshape(-1,1)
x_data_only_gender = OHE_X.fit_transform(x_data_only_gender).toarray()


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
mytest = np.array([0.455, 0.365, 0.095, 0.514, 0.2245, 0.101, 0.15])
mytest = mytest.reshape(-1,1)

print(x_data_toFS)
mytest = sc_X.fit_transform(mytest)

x_data_only_gender = pd.DataFrame(data=x_data_only_gender)
x_data_toFS = pd.DataFrame(data=x_data_toFS)

X_data = pd.concat([x_data_only_gender, x_data_toFS], axis=1, join_axes=[x_data_only_gender.index])


# y data preprocessing
new_y = []
for e in y_data:
    if e <= 9:
        new_y.append(1)
    elif e >= 18:
        new_y.append(3)
    else:
        new_y.append(2)
y_data = np.array(new_y)

y_data = y_data.reshape(-1,1)
from sklearn.preprocessing import OneHotEncoder

# encoding the independent variables
OHE_y = OneHotEncoder(categorical_features = [0])
y_data = OHE_y.fit_transform(y_data).toarray()

X_data = X_data.values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.2, random_state = 0)

X_train = np.reshape(X_train, X_train.shape + (1,))
X_test = np.reshape(X_test, X_test.shape + (1,))


from keras.models import Sequential
from keras.layers import Convolution1D
from keras.layers import MaxPooling1D
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution1D(nb_filter=128, filter_length=3, input_shape = X_train.shape[1:], activation = 'tanh'))

# Step 2 - Pooling
classifier.add(MaxPooling1D(pool_size = (2)))

# Adding a second convolutional layer
classifier.add(Convolution1D(4, kernel_size=(3), activation = 'tanh'))
classifier.add(MaxPooling1D(pool_size = (2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 3, activation = 'softmax'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

history = classifier.fit(X_train, y_train, batch_size=40, epochs=120, verbose=1, validation_data=(X_test, y_test))  # starts training
score = classifier.evaluate(X_test, y_test, batch_size=40)


import numpy as np
y_pred = classifier.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)


from sklearn.metrics import f1_score
f_1 = f1_score(y_test, y_pred, labels=None, pos_label=1, average='micro', sample_weight=None)

from sklearn.metrics import recall_score
recall = recall_score(y_test, y_pred, labels=None, pos_label=1, average='micro', sample_weight=None)

from sklearn.metrics import precision_score
precision = precision_score(y_test, y_pred, labels=None, pos_label=1, average='micro', sample_weight=None)


