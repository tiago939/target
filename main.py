import numpy as np

#set the data and labels from the training and test sets
#the data must have form list of lists such as [example 1, example 2, ...]
#where each example is a list of features, such as example 1 = [pixel 1, pixel 2, ...]
training_data = np.load('data/mnistset.bin', 'r')
training_labels = np.load('data/labels.bin', 'r')
test_data = np.load('data/mnistset_test.bin', 'r')
test_labels = np.load('data/labels_test.bin', 'r')
norm = 1.0 #normalize data
check_test = True #check the performance on the testing data after each epoch
number_examples_test = 100 #number of examples in the test set
hot_label = True#true if the label need to be transformed to one-hot enconding scheme, false if the label is the output layer target. Set False for regression problems.

#set network // format: [input layer, hidden layer 1, hidden layer 2, ..., output layer]
net = [784,100,10]

#set training parameters
epochs = 30 #number of times the network will train with the training set
number_examples_training = 100 #number of examples provided to network from the training set
mini_batch_size = 1 #number of examples to calculate the gradients before update the learning parameters
suffle_training = True #shuffle the examples in the training set

#set hyperparameters
learning_rate = 0.01 #size of the gradient to update the learning parameters
momentum_rate = 0.0 #rate of the momentum for optimization, 0 if SGD
tau = 100.0 #time step size
period = 1 #time steps

######################################################################################################################
def data():
    return [norm,training_data,training_labels,test_data,test_labels,check_test,number_examples_test,hot_label]

def network():
    return [net, epochs, number_examples_training,mini_batch_size,suffle_training,learning_rate,momentum_rate,tau,period]
