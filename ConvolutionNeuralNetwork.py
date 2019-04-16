import pandas as pd
import numpy as np
import time
import keras
import tensorflow as tf
# fix random seed for reproducibility
np.random.seed(7)

training_data = pd.read_csv('optdigits.tra', header = None)
testing_data = pd.read_csv('optdigits.tes', header = None)

X = training_data.iloc[:,:64].values
y = training_data.iloc[:,64].values

testing_X = testing_data.iloc[:,:64].values
testing_y = testing_data.iloc[:,64].values

# hot encoding the dependent variable
from keras.utils import np_utils
dummy_y = np_utils.to_categorical(y)
dummy_testing_y = np_utils.to_categorical(testing_y)

# Splitting the training and testing data from training_data dataframe
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, dummy_y, test_size=0.2, random_state=42)

# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 8, 8, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 8, 8, 1).astype('float32')
testing_X = testing_X.reshape(testing_X.shape[0], 8, 8, 1).astype('float32')

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense

start_time = time.time()

# Initializing the CNN
classifier = Sequential()

# Step-1 Convolution
classifier.add(Convolution2D(32, (3, 3), input_shape=(8, 8, 1), activation= 'relu'))

# Step-2 Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Using Dropout to reduce overfitting
classifier.add(Dropout(0.25))

# Step-3 Flattening
classifier.add(Flatten())

# Add fully connection layer
classifier.add(Dense(activation="relu", units=128))

# Another dropout to test the scenario.
classifier.add(Dropout(0.45))

# Output layer
classifier.add(Dense(activation="softmax", units=10))

# Compiling the classifier
# compiling the ANN
from keras import optimizers
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.90)
classifier.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

# fitting the CNN
classifier.fit(X_train, y_train, batch_size = 10, epochs = 30)
print("--- %s seconds ---" % (time.time() - start_time))

def class_accuracy(cm):
    diagonal_sum = np.diagonal(cm).sum()
    for x in range(10):
        TP = cm[x][x]
        TN = diagonal_sum - TP
        FP = cm[:,x].sum() - TP
        FN = cm[x,:].sum() - TP
        print("Accuracy of Class {0} is {1}".format(x,(TP+TN)/(TP+TN+FP+FN)))

# --------------------- Evaluate the validation data
loss, accuracy = classifier.evaluate(X_test, y_test)
print("\nLoss: {0}, Accuracy: {1}".format(loss, accuracy*100))

# --------------------- predicting the test set result for training_data
train_y_pred = classifier.predict(X_test)
train_y_pred = np.argmax(train_y_pred, axis=1)
y_test_non_category = y_test.argmax(1)

# --------------------- Making the confusion matrix for training_data
from sklearn.metrics import confusion_matrix
print("\nCONFUSION MATRIX TRAINING DATA")
cm_training = confusion_matrix(y_test_non_category,train_y_pred)
print(cm_training)

# --------------------- predicting class wise accuracy for training_data
print("\nTRAINING DATA class Accuracy")
class_accuracy(cm_training)

# --------------------- Evaluate the testing data network
loss_test, accuracy_test = classifier.evaluate(testing_X, dummy_testing_y)
print("\nTest_Loss: {0}, Test_Accuracy: {1}".format(loss_test, accuracy_test*100))

# --------------------- predicting the test set result for testing_data
test_y_pred = classifier.predict(testing_X)
test_y_pred = np.argmax(test_y_pred, axis=1)

# --------------------- Making the confusion matrix for testing_data
print("\nCONFUSION MATRIX")
cm_testing = confusion_matrix(testing_y,test_y_pred)
print(cm_testing)

# --------------------- predicting class wise accuracy for testing_data
print("\nTESTING DATA class Accuracy")
class_accuracy(cm_testing)

