import numpy as np
import matplotlib.pyplot as plt

class ReccurentNN:


    def __init__(self, char_to_idx, idx_to_char, vocab, h_size=75,
                  seq_len=20, clip_value=5, epochs=50, learning_rate=1e-2):

         self.n_h = h_size 
         self.seq_len = seq_len 
         self.clip_value = clip_value  
         self.epochs = epochs  
         self.learning_rate = learning_rate
         self.char_to_idx = char_to_idx  
         self.idx_to_char = idx_to_char  
         self.vocab = vocab  
         # smoothing loss as batch SGD is noisy
         self.smooth_loss = -np.log(1.0 / self.vocab) * self.seq_len  
         # initialize parameters
         self.params = {}
         self.params["W_xh"] = np.random.randn(self.vocab, self.n_h) * 0.01 
         self.params["W_hh"] = np.identity(self.n_h) * 0.01
         self.params["b_h"] = np.zeros((1, self.n_h))
         self.params["W_hy"] = np.random.randn(self.n_h, self.vocab) * 0.01
         self.params["b_y"] = np.zeros((1, self.vocab))
         self.h0 = np.zeros((1, self.n_h))  # value of the hidden state at time step t = -1
         # initialize gradients and memory parameters for Adagrad
         self.grads = {}
         self.m_params = {}
         for key in self.params:
             self.grads["d" + key] = np.zeros_like(self.params[key])
             self.m_params["m" + key] = np.zeros_like(self.params[key]) 

    def _encode_text(self, X):

         X_encoded = []
         for char in X:
             X_encoded.append(self.char_to_idx[char])
         return X_encoded

    def _prepare_batches(self, X, index):
         X_batch_encoded = X[index: index + self.seq_len]
         y_batch_encoded = X[index + 1: index + self.seq_len + 1]
         X_batch = []
         y_batch = []
         for i in X_batch_encoded:
             one_hot_char = np.zeros((1, self.vocab))
             one_hot_char[0][i] = 1
             X_batch.append(one_hot_char)
         for j in y_batch_encoded:
             one_hot_char = np.zeros((1, self.vocab))
             one_hot_char[0][j] = 1
             y_batch.append(one_hot_char)
         return X_batch, y_batch 

    def _softmax(self, x):
         e_x = np.exp(x - np.max(x)) 
         return e_x / np.sum(e_x)
         
    def _forward_pass(self, X):
         h = {}  # stores hidden states
         h[-1] = self.h0  # set initial hidden state at t=-1
         y_pred = {}  # stores softmax output probabilities
         # iterate over each character in the input sequence
         for t in range(self.seq_len):
             h[t] = np.tanh(
                 np.dot(X[t], self.params["W_xh"]) + np.dot(h[t - 1], self.params["W_hh"]) + self.params["b_h"])
             y_pred[t] = self._softmax(np.dot(h[t], self.params["W_hy"]) + self.params["b_y"])
         self.ho = h[t]
         return y_pred, h 

    def _backward_pass(self, X, y, y_pred, h):
         dh_next = np.zeros_like(h[0])
         for t in reversed(range(self.seq_len)):
             dy = np.copy(y_pred[t])
             dy[0][np.argmax(y[t])] -= 1  # predicted y - actual y
             self.grads["dW_hy"] += np.dot(h[t].T, dy)
             self.grads["db_y"] += dy
             dhidden = (1 - h[t] ** 2) * (np.dot(dy, self.params["W_hy"].T) + dh_next)
             dh_next = np.dot(dhidden, self.params["W_hh"].T)
             self.grads["dW_hh"] += np.dot(h[t - 1].T, dhidden)
             self.grads["dW_xh"] += np.dot(X[t].T, dhidden)
             self.grads["db_h"] += dhidden
         for grad, key in enumerate(self.grads):
             np.clip(self.grads[key], -self.clip_value, self.clip_value, out=self.grads[key])
         return 

    def _update(self):
         for key in self.params:
             self.m_params["m" + key] += self.grads["d" + key] * self.grads["d" + key]
             self.params[key] -= self.grads["d" + key] * self.learning_rate / (np.sqrt(self.m_params["m" + key]) + 1e-8) 

    def test(self, test_size, start_index):
         res = ""
         x = np.zeros((1, self.vocab))
         x[0][start_index] = 1
         for i in range(test_size):
             # forward propagation
             h = np.tanh(np.dot(x, self.params["W_xh"]) + np.dot(self.h0, self.params["W_hh"]) + self.params["b_h"])
             y_pred = self._softmax(np.dot(h, self.params["W_hy"]) + self.params["b_y"])
             # get a random index from the probability distribution of y
             index = np.random.choice(range(self.vocab), p=y_pred.ravel())
             # set x-one_hot_vector for the next character
             x = np.zeros((1, self.vocab))
             x[0][index] = 1
             # find the char with the index and concat to the output string
             char = self.idx_to_char[index]
             res += char
         return res 
    
    def train(self, X):
         loss = []
         # trim end of the text so we only get full sequences
         num_batches = len(X) // self.seq_len
         X_trimmed = X[:num_batches * self.seq_len] 
         # encode the characters to indices
         X_encoded = self._encode_text(X_trimmed) 
         for i in range(self.epochs):
             for j in range(0, len(X_encoded) - self.seq_len, self.seq_len):
                 X_batch, y_batch = self._prepare_batches(X_encoded, j)
                 y_pred, h = self._forward_pass(X_batch)
                 loss = 0
                 for t in range(self.seq_len):
                     loss += -np.log(y_pred[t][0, np.argmax(y_batch[t])])
                 self.smooth_loss = self.smooth_loss * 0.999 + loss * 0.001
                 loss.append(self.smooth_loss)
                 self._backward_pass(X_batch, y_batch, y_pred, h)
                 self._update()
             print(f'Epoch: {i + 1}\tLoss: {loss}')
             print(self.test(50,2))
         return loss, self.params 

