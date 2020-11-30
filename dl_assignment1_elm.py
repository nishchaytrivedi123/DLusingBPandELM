import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Read the dataset
data = pd.read_csv('EEG Eye State.csv')

# Take the inputs in X variable
X = data.loc[:,'COL1':'COL14']

# Take the class labels in labels
labels = data['eyeDetection']

# Number of class variables
n_class = 2

# Reshape the labels with shape of input shape
Y = np.zeros([labels.shape[0], n_class])
for i in range(labels.shape[0]):
        Y[i][labels[i]] = 1

# Split the training and testing based on 70:30
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size= 0.3)

# print('Train size: {train}, Test size: {test}'.format(train=x_train.shape[0], test=x_test.shape[0]))

# Take the length of input and define hidden neurons
in_len = x_train.shape[1]  
hidd_neuron = 15

# Randomly initialize weights to input
in_weight = np.random.normal(size=[in_len, hidd_neuron])
# print('Input Weight shape: ', in_weight.shape)

# Calculate the dot product of inputs of weight and transfer it.
def propagate(x):
    a = np.dot(x, in_weight)
    return a

X = propagate(x_train)
X_transpose = np.transpose(X)

# The main equation of ELM, calculate weight
weight_out = np.dot(np.linalg.inv(np.dot(X_transpose, X)), np.dot(X_transpose, y_train))
# print('Output weights shape: ', weight_out.shape)


# Predict the value
def predict(test):
    test = propagate(test)
    label = np.dot(test, weight_out)
    return label

predicted_val = predict(x_test)
correct = 0
total = predicted_val.shape[0]

# Check the accuracy of model_selection
for i in range(total):
    predicted = np.argmax(predicted_val[i])
    test = np.argmax(y_test[i])
    # print(predicted, test)
    correct = correct + (1 if predicted == test else 0)
print('Accuracy: ', (correct/total))