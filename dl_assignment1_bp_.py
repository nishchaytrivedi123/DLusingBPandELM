import pandas as pd
from sklearn.model_selection import train_test_split
from dl_assignment1_bp_network import forward_propagate, backward_propagate_error, back_propagation, train_network, define_network

# Calculate accuracy percentage
def calc_acc(actual, predicted):
	counter = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			counter += 1
	return counter / float(len(actual)) * 100.0

# Main function to divide dataset in training and testing and perform Back Propagation
def main(data, BP, *args):	
	X = data.loc[:,'COL1':'eyeDetection']
	Y = data['eyeDetection']

	# Split dataset in 70:30
	x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size= 0.3)
	x_train_np = x_train.to_numpy().tolist()
	y_train_np = y_train.to_numpy().tolist()

	x_test_np = x_test.to_numpy().tolist()
	y_test_np = y_test.to_numpy().tolist()

	# Call Back Propagation function and predict the value
	predicted = BP(x_train_np, x_test_np, *args)
	accuracy = calc_acc(y_test_np, predicted)
	return accuracy

# Read a dataset
dataset = pd.read_csv('EEG Eye State.csv')
# dataset = dataset

# Call the main funciton to define network with 4 hidden layers and perfrom back propagation
acc = main(dataset, back_propagation, 0.0001, 500, 4)

# Final model's Accuracy
print('Accuracy: ', (acc), '%')