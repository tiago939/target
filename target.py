import random, sys ,pickle
import numpy as np
import main, test

#load the data from the main script
norm = main.data()[0]
training_data = main.data()[1]
training_labels = main.data()[2]
check_test = main.data()[5]
hot_label = main.data()[7]

#load the parameters from the main script
net = main.network()[0]
epochs = main.network()[1]
number_examples = main.network()[2]
mini_batch_size = main.network()[3]
suffle_training = main.network()[4]
learning_rate = main.network()[5]
momentum_rate = main.network()[6]
tau = main.network()[7]
period = main.network()[8]

#activation function
def f(z):
    if z>100.0:
        return 1.0
    elif z<-100.0:
        return 0.0
    else:
        return 1.0/(1.0+np.exp(-z))

#derivative of the activation function
def fprime(z):
    return f(z)*(1.0-f(z))

x=[] #input to each neuron
y=[] #activation of each neuron
target=[] #target of each neuron
for layer in range(len(net)):
    x.append(np.array([0.0]*net[layer]))
    y.append(np.array([0.0]*net[layer]))
    target.append(np.array([0.0]*net[layer]))

#initialize weights and momentum
weights=[]
momentum=[]
for layer in range(len(net)-1):
    if layer==0:
        sigma=(48.0/(35.0*net[layer]))**0.5
    else:
        sigma=(16.0/(11.0*net[layer]))**0.5
    momentum.append(np.zeros((net[layer+1],net[layer])))
    weights.append(np.random.normal(0.0,sigma,(net[layer+1],net[layer])))

#initialize gradients
gradients=[]
for layer in range(len(net)-1):
    gradients.append(np.zeros((net[layer+1],net[layer])))

list_of_examples = []
for example in range(number_examples):
    list_of_examples.append(example)

for trials in range(epochs):
    score_training = 0.0
    example_counter = 0.0
    batch_counter = 0
    score_training = 0.0
    loss_training = 0.0
    
    if suffle_training == True:
        random.shuffle(list_of_examples)
    
    for example in list_of_examples:
        example_counter += 1.0
        batch_counter += 1
        
        if hot_label==True:
            target[len(net)-1] = np.array([0.0]*net[len(net)-1])
            target[len(net)-1][training_labels[example]] = 1.0
        else:
            target[len(net)-1] = training_labels[example]
        
        #update activation of each neuron
        y[0] = training_data[example]/norm
        for layer in range(1,len(net)):
            x[layer]=np.dot(weights[layer-1],y[layer-1])
            y[layer]=map(f,x[layer])
        
        #guess the class from classifcation problem
        if hot_label == True:
            guess = np.argmax(y[len(net)-1])
        
        #calculate online loss on training data
        loss_training += (0.5/number_examples)*(1.0/net[len(net)-1])*sum( (target[len(net)-1] - y[len(net)-1])**2.0)
        
        if batch_counter <= mini_batch_size:
            #compute gradient from the output layer
            gradient_output = (y[len(net)-1]-target[len(net)-1])*map(fprime,x[len(net)-1])
            gradients[len(net)-2] += np.outer(gradient_output,y[len(net)-2])
            
            graph = open('score','w', 0)
            #compute targets for all hidden layers
            
            for layer in range(len(net)-2,0,-1):
                target[layer] = np.array([0.0]*net[layer])
                x_hat = x[layer+1][:]
                y_hat = y[layer+1][:]
                
                for time_steps in range(period):
                    target[layer] += -tau*np.dot(np.transpose(weights[layer]),(y[layer+1]-target[layer+1])*map(fprime,x_hat))
                    x_hat = np.dot(weights[layer],target[layer])
                    y_hat = map(f,x_hat)

                #compute gradients from the hidden layers
                gradients[layer-1] += -np.outer((target[layer]-y[layer])*map(fprime,x[layer]),y[layer-1])
                
        #update the learning parameters
        if batch_counter == mini_batch_size:
            for layer in range(0,len(net)-1):
                momentum[layer] = momentum_rate*momentum[layer] - (1.0-momentum_rate)*learning_rate*gradients[layer]/mini_batch_size
                weights[layer] += momentum[layer]
            
            #reset batch counter and gradients
            batch_counter = 0
            for layer in range(len(net)-1):
                gradients[layer] = np.zeros((net[layer+1],net[layer]))
            
        #online classification performance on the training data
        if hot_label == True:
            if guess == training_labels[example]:
                score_training += 1.0
            
        sys.stdout.write("\r%f %f %i %i" % (score_training/example_counter,loss_training,example_counter,trials))
        sys.stdout.flush()
        
    #save the weights after each epoch
    address = file("weights","wb")
    pickle.dump(weights, address)
    address.close()
    
    if check_test == True:
        score_test = test.getScore()
        print (' ')
        print('epoch: ',trials, ' score test, loss: ',score_test)
