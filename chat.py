import random
import json

import torch
import random
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

#device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state) # Loads our model's parameter dictionary.
model.eval()  # model.train() sets our model in training mode. model.eval() sets our model in evaluation or inference mode.

bot_name = "RAEL"
fallback_response = ["I didn't get that. Can you say it again?","I missed what you said. What was that?","Sorry, could you say that again?","Sorry, can you say that again?",
"Can you say that again?","Sorry, I didn't get that. Can you rephrase?","Sorry, what was that?","I didn't get that. Can you repeat?"]


#using the model; making predictions

def get_response(msg):
    sentence = tokenize(msg) # We need to tokenized the user's input.
    X = bag_of_words(sentence, all_words) # Convert it to a bag of words.
    X = X.reshape(1, X.shape[0]) # It returns a tensor with the same data and number of elements as X, with the specified shape (1, X.shape[0]). We don't need an array of zeros and ones, but a matrix with 1 row and X.shape[0] columns.
    X = torch.from_numpy(X).to(device) # It creates a tensor from a numpy.ndarray.

    output = model(X) # Get model predictions for the current sentence's bag of words. _Pytorch model returns a matrix instead of a column vector!_ 
    _, predicted = torch.max(output, dim=1) #torch.max function will return the value and the index. prediction is the class label so we dont need the first actual value // It returns the maximum value of all elements in the "output" tensor. dim is the dimension to reduce (0, columns; 1, rows). It returns a tuple (values, indices) where values is the maximum value of each row of the input tensor in the given dimension dim and indices ("predicted") is the index location of each maximum value found.

    tag = tags[predicted.item()] # predicted.item() returns the value of this tensor as a standard Python number, so this is the index that we need to find the "predicted" tag.

    probs = torch.softmax(output, dim=1) # It applies the softmax function to our model prediction to the user's input. The output of the softmax function is a probability distribution. It returns a tensor of the same dimension and shape as the "output", a matrix with 1 row and len(myPreProcessing.get_tags()) columns.
    prob = probs[0][predicted.item()] # Finally, we get the probability of this predicted tag.
    if prob.item() > 0.75: # If the probability is good enough, we will answer the user with one random response of the predicted tab.
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    
    return "Sorry, I don't understand."


if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        # sentence = "do you use credit cards?"
        sentence = input("You: ")
        if sentence == "quit": # Some keywords will trigger some functions, such as quit (quit the chat), time, etc.
            break

        resp = get_response(sentence)
        print(resp)

