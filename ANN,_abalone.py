# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 18:34:23 2018

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

'''
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
'''
y_data = y_data.reshape(-1,1)


from sklearn.preprocessing import OneHotEncoder

# encoding the independent variables
OHE_y = OneHotEncoder(categorical_features = [0])
y_data = OHE_y.fit_transform(y_data).toarray()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.2, random_state = 0)

from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.models import Sequential

model = Sequential()
model.add(Dense(units = 70, activation = "tanh", input_dim = 10, init = 'uniform'))
model.add(Dense(units = 65, init = 'uniform')) #relu
model.add(BatchNormalization())
model.add(Activation('tanh'))
model.add(Dense(units = 65, init = 'uniform', activation='tanh')) # relu
model.add(Dropout(0.4)) # relu
model.add(Dense(29, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=25, nb_epoch=120)
score = model.evaluate(X_test, y_test, batch_size=25)

sgdvals= history
adamVals = history

import matplotlib.pyplot as plt
plt.plot(sgdvals.history['acc'])
plt.plot(adamVals.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['SGD', 'Adam'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(sgdvals.history['loss'])
plt.plot(adamVals.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['SGD', 'Adam'], loc='upper left')
plt.show()


import numpy as np
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)
# Create confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.hsv): # edit color here
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

class_names = ['1', '2', '3']

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cm, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()
# current version yields an accuracy of  79.07%

import numpy as np
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)


from sklearn.metrics import f1_score
f_1 = f1_score(y_test, y_pred, labels=None, pos_label=1, average='micro', sample_weight=None)

from sklearn.metrics import recall_score
recall = recall_score(y_test, y_pred, labels=None, pos_label=1, average='micro', sample_weight=None)

from sklearn.metrics import precision_score
precision = precision_score(y_test, y_pred, labels=None, pos_label=1, average='micro', sample_weight=None)





