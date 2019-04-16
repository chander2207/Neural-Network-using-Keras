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

# Feature Scaling
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.fit_transform(X_test)
# testing_X = sc.fit_transform(testing_X)

from keras.models import Sequential
from keras.layers import Dense

start_time = time.time()
#Initializing the ANN
classifier = Sequential()

#Adding the input layer and the first hidden layer
# classifier.add(Dense(activation="relu", input_dim=64, units=52, kernel_initializer="uniform"))
classifier.add(Dense(activation="tanh", input_dim=64, units=32, kernel_initializer="uniform"))

# Adding the second hidden layer
# classifier.add(Dense(activation="relu", units=52, kernel_initializer="uniform"))
# classifier.add(Dense(activation="tanh", units=50, kernel_initializer="uniform"))

# Adding the third hidden layer
# classifier.add(Dense(activation="relu", units=100, kernel_initializer="uniform"))
# classifier.add(Dense(activation="tanh", units=100, kernel_initializer="uniform"))

# Adding the output layer
classifier.add(Dense(activation="softmax", units=10, kernel_initializer="uniform"))

# compiling the ANN
from keras import optimizers
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.90)
# classifier.compile(optimizer='sgd',loss='mean_squared_error', metrics=['accuracy'])
classifier.compile(optimizer='sgd',loss='categorical_crossentropy', metrics=['accuracy'])

# Fitting the ANN to the training data
classifier.fit(X_train, y_train, batch_size = 5, epochs = 50)
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