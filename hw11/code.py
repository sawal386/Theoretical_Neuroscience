#A two layer feedforward network to classify handwritten digits(MNIST)

import sys
sys.path.append("/anaconda3/lib/python3.6/site-packages/")
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as sio
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt 

## Build model (shouldn't have to change anything here)
class NN(nn.Module):

    def __init__(self, n_units,input_size):

        super(NN, self).__init__()
        self.n_units = n_units
        self.input_size = input_size
        self.build_model()

    def build_model(self):

        self.linear1 = nn.Linear(self.input_size,self.n_units)
        self.linear2 = nn.Linear(self.n_units,self.n_units)
        self.linear3 = nn.Linear(self.n_units,10)
        
    def forward(self, x):

        x = torch.tanh(self.linear1(x))
        x = torch.tanh(self.linear2(x))
        x = self.linear3(x)
        return x

## Load data (shouldn't have to change anything here)

matfile = sio.loadmat('MNISTdataset_matlab.mat')
test_data = matfile['test_data'].T/255.
test_labels = np.where(matfile['test_labels'].T)[1] # convert from one hot encoding to labels
train_data = matfile['train_data'].T/255.
train_labels = np.where(matfile['train_labels'].T)[1] # convert from one hot encoding to labels

input_size = test_data.shape[1]

## Convert data to torch tensors

test_data = torch.tensor(test_data).float()
test_labels = torch.tensor(test_labels)

train_data = torch.tensor(train_data).float()
train_labels = torch.tensor(train_labels)

batch_size=200
train_inds = np.arange(0,train_data.shape[0]/batch_size).astype('int')

def plot_figure(x_data, y_data_li, x_lab, y_lab, title, legend_li):
    
    fig = plt.figure(figsize=(15.0, 9.0))
    axes1 = fig.add_subplot(1,1,1)
    axes1.set_xlabel(x_lab)
    axes1.set_ylabel(y_lab)
    axes1.set_title(title)
    for i in range(len(y_data_li)):
        print(len(x_data))
        print(len(y_data_li[i]))
        axes1.scatter(x_data, y_data_li[i], label = legend_li[i])
    
    axes1.legend()
    plt.savefig('plot_2.png')
    plt.show()

def plot_figure2(x_data, y_data_li, x_lab, y_lab, title, legend_li):
    
    fig = plt.figure(figsize=(15.0, 9.0))
    axes1 = fig.add_subplot(1,1,1)
    axes1.set_xlabel(x_lab)
    axes1.set_ylabel(y_lab)
    axes1.set_title(title)
    print(len(x_data), len(y_data_li[0]))
    for i in range(len(y_data_li)):
        axes1.scatter(x_data, y_data_li[i][1:], label = legend_li[i % 2] + " Trial #" + str(int(i / 2) + 1))
    
    axes1.legend()
    plt.savefig('plot_3.png')
    plt.show()


## Train model (will have to make changes here!)
test_accu2 = []
train_accu2 = []
num_trials = 1
legend_names_li = []
accuracy_96_test = []
all_res = []
for i in range(num_trials):
    print("Trial Number:", i)
    n_epochs = 30 #optimal = 30
    learning_rate = 0.007 #optimal = 0.002
    n_hidden_units = 20 #optimal = 75
    
    # Build model
    net = NN(n_hidden_units,input_size)
    
    # Set up optimizer and loss
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    
    train_accuracy_ar = []
    test_accuracy_ar = []
    
    # Loop over epochs
    for i_epoch in range(n_epochs):
        np.random.shuffle(train_inds) # shuffle training indices at each step
        # Loop over training data
        
        for i_train in train_inds:
            optimizer.zero_grad()   # zero the gradient buffers
            output = net(train_data[i_train*batch_size:(i_train+1)*batch_size])
            loss = criterion(output,train_labels[i_train*batch_size:(i_train+1)*batch_size])
            loss.backward()
            
            if i_epoch>0:
                optimizer.step() # take gradient step
        # Check test accuracy
        output = net(test_data)
        predicted_labels = output.argmax(dim=1, keepdim=True).squeeze()
        test_loss = criterion(output, test_labels).item() 
        number_correct_test = torch.eq(test_labels, predicted_labels).sum().item()
        test_accuracy = number_correct_test/test_data.shape[0]
        test_accuracy_ar.append(test_accuracy)
        
        
        #Check train accuracy
        output_tr = net(train_data)
        train_pred_labels = output_tr.argmax(dim = 1, keepdim = True).squeeze()
        train_loss = criterion(output_tr, train_labels).item()
        number_correct_train = torch.eq(train_labels, train_pred_labels).sum().item()
        train_accuracy = number_correct_train / train_data.shape[0]
        train_accuracy_ar.append(train_accuracy)
        print("Test:", test_accuracy, "train:",train_accuracy)
    plot_figure(np.linspace(1, n_epochs, n_epochs), [test_accuracy_ar, train_accuracy_ar],
           "Epoch number","Accuracy", "Variation of Accuracy with Epoch",["Test", "Train"])        
        
'''
    test_accu2.append(np.average(test_accuracy_ar))
    train_accu2.append(np.average(train_accuracy_ar))
    all_res.append(test_accuracy_ar)
    all_res.append(train_accuracy_ar)
    #for question 1
    #plot_figure(np.linspace(1, n_epochs, n_epochs), [test_accuracy_ar, train_accuracy_ar],
     #      "Epoch number","Accuracy", "Variation of Accuracy with Epoch",["Test", "Train"])

    reach_96 = False
    for i in range(len(test_accuracy_ar)):
        
        if test_accuracy_ar[i] > 0.96 and not reach_96:
            accuracy_96_test.append(i)
            reach_96 = True
'''


