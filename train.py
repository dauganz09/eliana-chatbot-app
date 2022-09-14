import numpy as np
import random
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet

with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []
# loop through each sentence in our intents patterns
for intent in intents['intents']:
    tag = intent['tag']
    # add to tag list
    tags.append(tag)
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = tokenize(pattern)
        # add to our words list
        all_words.extend(w)
        # add to xy pair
        xy.append((w, tag))

# stem and lower each word
ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
# remove duplicates and sort
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)

# create training data
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    # X: bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Hyper-parameters 
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
print(input_size, output_size)

class ChatDataset(Dataset):
# Code for processing data samples can get messy and hard to maintain. We want our dataset code to be decoupled from our model training code for better readability and modularity. torch.utils.data.Dataset stores the samples and their corresponding labels. A custom Dataset class must implement three functions: __init__, __len__, and __getitem__.
    def __init__(self): #It is run once when instantiating the Dataset object.
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # support indexing such that dataset[i] can be used to get i-th sample
    # It loads and returns a sample from the dataset at the given "index"
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    # It returns the number of samples in our dataset.
    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss() #this code will apply the softnax for us kaya hindi na nilagay sa model // A loss function takes the (output, target) pair of inputs, and computes a value that estimates how far away the output is from the target. The input is expected to contain raw, unnormalized scores for each class (aka logits), so we don't need to convert it into probabilities by a softmax function.
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # Create our optimizer. We need to give it an iterable containing the parameters (model.parameters() returns the model's parameters, i.e. weights and biases) and we also specify one optimizer-specific option (the learning rate).

# Train the model (WHOLE TRAINING LOOP)
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device) #.to(device) para ma push sa gpu if it is available
        labels = labels.to(dtype=torch.long).to(device) 
        
        # Forward pass
        outputs = model(words) #code para get yung words // Get model output for the current words (*1)
        # if y would be one-hot, we must apply
        # labels = torch.max(labels, 1)[1]
        loss = criterion(outputs, labels) #and then we calculate the loss and this will get the predicted output and the actual labels // Compute the loss between actual and output values (*2).
        
        # Backward and optimize
        optimizer.zero_grad() #we use this to empty the values in the gradient attribute //  We clear calculated gradients because in PyTorch, for every mini-batch during the training phase, we need to explicitly set the gradients to zero before starting to do backpropragation (i.e., updating our weights and biases) because PyTorch accumulates the gradients on subsequent backward passes.
        loss.backward() #this will do the backpropagation // After computing the loss (how far is the output from being correct), we propagate gradients back into the network's parameters (*3).
        optimizer.step() #this will do an update step and update the parameters for us // Update parameters, typically weight = weight - learning_rate * gradient (*4)
        
    if (epoch+1) % 100 == 0: #para ma print ang loss. Every 100 step we want to print some information so lets print the current epoch and print the number of epochs and print the loss and only print four decimal value
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


print(f'final loss: {loss.item():.4f}')

data = {
"model_state": model.state_dict(), # The parameters (i.e. weights and biases) of a model are contained in the model’s parameters (accessed with model.parameters()). A state_dict is a Python dictionary object that maps each layer to its parameter tensor. 
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

FILE = "data.pth"
torch.save(data, FILE) #  Saves our model to a disk file. This function uses Python’s pickle utility for serialization.

print(f'training complete. file saved to {FILE}')
