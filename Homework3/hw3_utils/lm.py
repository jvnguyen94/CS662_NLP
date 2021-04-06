import torch
import torch.nn as nn
import random
import numpy as np
import matplotlib.pyplot as plt
import itertools
from hw3_utils import vocab

class NameGenerator(nn.Module):
    def __init__(self, input_vocab_size, n_embedding_dims, n_hidden_dims, n_lstm_layers, output_vocab_size):
        """
        Initialize our name generator, following the equations laid out in the assignment. In other words,
        we'll need an Embedding layer, an LSTM layer, a Linear layer, and LogSoftmax layer. 
        
        Note: Remember to set batch_first=True when initializing your LSTM layer!

        Also note: When you build your LogSoftmax layer, pay attention to the dimension that you're 
        telling it to run over!
        """
        super(NameGenerator, self).__init__()
        self.lstm_dims = n_hidden_dims
        self.lstm_layers = n_lstm_layers

        ## input embedding layer
        self.input_lookup = nn.Embedding(
            num_embeddings=input_vocab_size, embedding_dim=n_embedding_dims)
        ## lstm
        self.lstm = nn.LSTM(input_size=n_embedding_dims, hidden_size=n_hidden_dims,
                            num_layers=n_lstm_layers, batch_first=True)
        ## output softmax classifier
        self.output = nn.Linear(in_features=n_hidden_dims,
                                out_features=output_vocab_size)
        ## log-softmaxing
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, history_tensor, prev_hidden_state):
        """
        Given a history, and a previous timepoint's hidden state, predict the next character. 
        
        Note: Make sure to return the LSTM hidden state, so that we can use this for
        sampling/generation in a one-character-at-a-time pattern, as in Goldberg 9.5!
        """      
        ## get embedding  
        out = self.input_lookup(history_tensor)

        ## lstm
        out, self.hidden = self.lstm(out, prev_hidden_state)
        # print(out.shape)

        ## LINEAR layer
        out = self.output(out)
        # print(out.shape)

        ## Softmax
        out = self.softmax(out)     

        return out, self.hidden


    def init_hidden(self):
        """
        Generate a blank initial history value, for use when we start predicting over a fresh sequence.
        """
        h_0 = torch.randn(self.lstm_layers, 1, self.lstm_dims)
        c_0 = torch.randn(self.lstm_layers, 1, self.lstm_dims)

### Utility functions

def train(model, epochs, training_data, c2i):
    """
    Train model for the specified number of epochs, over the provided training data.
    
    Make sure to shuffle the training data at the beginning of each epoch!
    """
    opt = torch.optim.Adam(model.parameters())

    # since our model gives negative log probs on the output side
    loss_func = torch.nn.NLLLoss()

    loss_batch_size = 100

    for i in range(epochs):

        x_train = training_data.sentence.values
        y_train = training_data.lang.values

        # There's a more pandas-ish way to do this...
        pairs = list(zip(x_train, y_train))
        random.shuffle(pairs)

        loss = 0

        for x_idx, (x, y) in enumerate(pairs):

            if x_idx % loss_batch_size == 0:
                opt.zero_grad()

            x_tens = vocab.sentence_to_tensor(x, c2i)

            y_hat = model(x_tens)

            y_tens = torch.tensor(l2i[y])

            loss += loss_func(y_hat.unsqueeze(0), y_tens.unsqueeze(0))

            if x_idx % 1000 == 0:
                print(
                    f"{x_idx}/{len(pairs)} average per-item loss: {loss / loss_batch_size}")
                if validation_data is not None:
                    acc_score, _ = eval_func(
                        model, c2i)
                    print(f"\tValidation portion accuracy: {acc_score}")

            if x_idx % loss_batch_size == 0 and x_idx > 0:
                # send back gradients:
                loss.backward()
                # now, tell the optimizer to update our weights:
                opt.step()
                loss = 0

        # now one last time:
        loss.backward()
        opt.step()

def sample(model, c2i, i2c, max_seq_len=200):
    """
    Sample a new sequence from model.
    
    The length of the resulting sequence should be < max_seq_len, and the 
    new sequence should be stripped of <bos>/<eos> symbols if necessary.
    """

    
def compute_prob(model, sentence, c2i):
    """
    Compute the negative log probability of p(sentence)
    
    Equivalent to equation 3.3 in Jurafsky & Martin.
    """
    
    nll = nn.NLLLoss(reduction='sum')
    
    with torch.no_grad():
        s_tens = vocab.sentence_to_tensor(sentence, c2i, True)
        x = s_tens[:,:-1]
        y = s_tens[:,1:]
        y_hat, _ = model(x, model.init_hidden())
        return nll(y_hat.squeeze(), y.squeeze()).item() # get rid of first dimension of each
