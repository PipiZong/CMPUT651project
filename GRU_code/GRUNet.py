import torch
import torch.nn as nn
from torch.autograd import Variable

# generate embedding layer to encode index sentences into matrices
def create_emb_layer(weights_matrix, non_trainable=False):
    num_embeddings, embedding_dim = weights_matrix.shape
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': torch.Tensor(weights_matrix)})
    #emb_layer.weight = nn.Parameter(torch.Tensor(weights_matrix))
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim

# the GRU neural network model
class GRUNet(nn.Module):
    def __init__(self, weights_matrix, hidden_size, output_size, drop):
        super(GRUNet, self).__init__()
        #create embedding layer
        self.embedding, num_embeddings, embedding_dim = create_emb_layer(weights_matrix, True)
        self.hidden_size = hidden_size
        self.out_size = output_size

        #self.gru = nn.GRU(embedding_dim, hidden_size, 1, dropout=drop, batch_first=True) # single layer GRU
        self.gru2 = nn.GRU(embedding_dim, hidden_size, 2, batch_first=True) # double layer GRU
        self.fc = nn.Linear(hidden_size, output_size) # linear layer for projection
        self.fc2 = nn.Linear(hidden_size*2, output_size) # linear layer for projection when output for 2 classes
        self.relu = nn.ReLU() # relu tranformation
        self.sig = nn.Sigmoid() # sigmoid tranformation
        self.tanh = nn.Tanh() # tanh transformation
        self.softmax = nn.Softmax() # softmax transformation
        
        # all these layers and transformations are declared to test on results with various combinations

    # forward path
    def forward(self, inp, hidden, show):
        # all of the commented lines are the approaches I have tried,
        # the final model is order of: embedding layer -> 2level GRU -> Linear
        emb = self.embedding(inp)
        # out, h = self.gru(emb, hidden)
        out, h = self.gru2(emb)
        # out, h = self.lstm(emb)
        # y_pred, hid = torch.sigmoid(torch.tensor(rgru))
        # return y_pred, hid
        # out = self.fc(self.sig(rgru[:,-1]))
        #out = torch.max(out,1)[0][:, -1].resize_((torch.max(out,1)[0][:, -1].size(0),1))
        #out = self.fc(out[:, -1])
        out = self.fc(out[:, -1, :])
        #out_max = torch.max(out,1)[0]
        #out = self.fc(out_max)
        #out_mean = torch.mean(out,1)
        #conc = torch.cat((self.relu(out_max),self.relu(out_mean)),1)
        #out = self.fc2(conc)
        
        #out = self.fc(self.relu(out[:, -1]))
        
        #y_pred = self.sig(out)
        y_pred = out
        if show:
            print('after sig:\n'+str(y_pred))
        
        #if self.out_size > 1:
        #    y_pred = self.softmax(y_pred)
        if show:
            print('after softmax:\n'+str(y_pred))
        #y_pred = torch.sigmoid(out)
        # print('p_pred0: '+str(y_pred[0]))
        return y_pred, h

    #initiate empty hidden states, but may not be needed
    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)
