from random import randrange
from random import random
from math import exp

# Propagate the output from previous layer to next layer
def forward_propagate(model, instance):
	inputs = instance
	for layer in model:
		layer_inputs = []
		for neuron in layer:
			activation = activation_tranfer(neuron['weights'], inputs)
			neuron['out'] = activation_func(activation)
			layer_inputs.append(neuron['out'])
		inputs = layer_inputs
	return inputs

# Pass the errors back to neurons and store those erros in neurons
def backward_propagate_error(model, expected):
	for i in reversed(range(len(model))):
		layer = model[i]
		err = []
		if i != len(model)-1:
			for j in range(len(layer)):
				error = 0.0
				for neuron in model[i + 1]:
					error += (neuron['weights'][j] * neuron['actual'])
				err.append(error)
		else:
			for j in range(len(layer)):
				neuron = layer[j]
				err.append(expected[j] - neuron['out'])
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['actual'] = err[j] * activation_derivative(neuron['out'])
			# activation_derivative(neuron['out'])

# This back_propagation performs the Back Propagation algorithm with Stochastic Gradient Decsent (SGD)
def back_propagation(train_set, test_set, learning_rate, epochs, hidden_layers):
	n_in = len(train_set[0]) - 1
	n_out = len(set([inst[-1] for inst in train_set]))
	model = define_network(n_in, hidden_layers, n_out)
	train_network(model, train_set, learning_rate, epochs, n_out)
	predictions = []
	for inst in test_set:
		predictions.append(predict(model, inst))
	return(predictions)

# Train a network for a fixed number of epochs
def train_network(model, train, learning_rate, epochs, n_out):
	for epoch in range(epochs):
		for inst in train:
			outputs = forward_propagate(model, inst)
			expected = [0 for i in range(n_out)]
			backward_propagate_error(model, expected)
			update_weights(model, inst, learning_rate)

# Create a neural network with random weights initially
def define_network(n_in, hidden_layers, n_out):
	model = []
	hidden_layer = [{'weights':[random() for i in range(n_in + 1)]} for i in range(hidden_layers)]
	model.append(hidden_layer)
	output_layer = [{'weights':[random() for i in range(hidden_layers + 1)]} for i in range(n_out)]
	model.append(output_layer)
	return model

# Activation funciton after calculating result for each layer
def activation_tranfer(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation

# Based on previous error and weights, update weight
def update_weights(model, row, learning_rate):
	for i in range(len(model)):
		inputs = row[:-1]
		if i != 0:
			inputs = [neuron['out'] for neuron in model[i - 1]]
		for neuron in model[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] += learning_rate * neuron['actual'] * inputs[j]
			neuron['weights'][-1] += learning_rate * neuron['actual']

# Calculate the derivative of an neuron output
def activation_derivative(output):
	return output * (1.0 - output)

# Predict the result on test data based on trained model
def predict(model, instance):
	outputs = forward_propagate(model, instance)
	return outputs.index(max(outputs))

# Transfer neuron activation
def activation_func(activation):
	return 1.0 / (1.0 + exp(-activation))
