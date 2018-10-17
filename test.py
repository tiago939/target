import numpy as np
import main, pickle

#load the data from the main script
norm = main.data()[0]
test_data = main.data()[3]
test_labels = main.data()[4]
total = main.data()[6]
hot_label = main.data()[7]

#load the parameters from the main script
net = main.network()[0]

#activation function
def f(z):
    if z>100.0:
        return 1.0
    elif z<-100.0:
        return 0.0
    else:
        return 1.0/(1.0+np.exp(-z))

x=[] #input to each neuron
y=[] #activation of each neuron
for layer in range(len(net)):
    x.append(np.array([0.0]*net[layer]))
    y.append(np.array([0.0]*net[layer]))

def getScore():
    score_test = 0.0
    weights = pickle.load( open( "weights", "rb" ) )
    loss_test = 0.0
    
    for example in range(total):
        #update activation of each neuron
        y[0] = test_data[example]/norm
        for layer in range(1,len(net)):
            x[layer]=np.dot(weights[layer-1],y[layer-1])
            y[layer]=map(f,x[layer])
        
        #guess the class from classifcation problem
        guess = np.argmax(y[len(net)-1])
        
        if hot_label==True:
            target = np.array([0.0]*net[len(net)-1])
            target[test_labels[example]] = 1.0
        else:
            target = test_labels[example]
        
        #classification performance on the test data
        if hot_label == True: #for classification problems
            if guess == test_labels[example]:
                score_test += 1.0
        loss_test += (0.5/total)*(1.0/net[len(net)-1])*sum( (target - y[len(net)-1])**2.0)
    
    if hot_label == True:
        return score_test/total, loss_test
    else:
        return loss_test
