import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
from sklearn.metrics import r2_score


class BobNet(nn.Module):
    """
        The BobNet model that will be used to swing.
    """
 
    def __init__(self, input_size: int, output_size: int, hidden_dim: int,
                 n_layers: int, seq_length: int, bidirectional: bool=True, drop_prob: float=0.1):
        """
            Initialize the model by setting up the layers.
        """
        super(BobNet, self).__init__()
 
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.seq_length = seq_length
        self.input_size = input_size
        self.num_directions = 2 if self.bidirectional else 1
        
        #LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_dim, n_layers, 
                            dropout=drop_prob, batch_first=False,
                            bidirectional=bidirectional)
        
        # dropout layer
        self.dropout = nn.Dropout(drop_prob)
        
        # linear and sigmoid layers
        if bidirectional:
          self.fc = nn.Linear(hidden_dim*2, output_size)
        else:
          self.fc = nn.Linear(hidden_dim, output_size)
        
        
 
    def forward(self, x, hidden):
        """
            Perform a forward pass of our model on some input and hidden state.
        """
        batch_size = x.size(0)
        assert x.shape==(batch_size,self.seq_length, self.input_size)
        # If not batch_first, switch seq_length and batch_size position.
        x.transpose_(1,0)
        lstm_out, hidden = self.lstm(x, hidden)
        h_n, c_n = hidden
        assert lstm_out.shape==(self.seq_length, batch_size, self.hidden_dim )
        assert h_n.shape==c_n.shape==(self.n_layers*self.num_directions, batch_size, self.hidden_dim)
        # dropout and fully-connected layer
        out = self.dropout(lstm_out)
        # use last seq as out
        out = out[-1]
        out = self.fc(out)

        # reshape to be batch_size first
        out = out.view(batch_size, -1)
        return out, hidden
    
    
    def init_hidden(self, batch_size: int):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        
        hidden = (weight.new(self.n_layers*self.num_directions, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers*self.num_directions, batch_size, self.hidden_dim).zero_()
                     )
        
        return hidden



if __name__ == '__main__':
    # Init model
    batch_size=8
    hidden_dim=4
    input_size=5
    seq_length=3
    n_layers=1
    output_size=1
    bob = BobNet(input_size=input_size, output_size=output_size, hidden_dim=hidden_dim,
                 n_layers=n_layers, seq_length=seq_length, bidirectional=False, drop_prob=0)
    # Init data
    input_data=np.random.uniform(0,20,size=(batch_size * 3, seq_length, input_size)).astype(np.float32)
    print('input data shape', input_data.shape)
    print('########input data########')
    print(input_data)
    print('########input data########')
    y=np.random.uniform(0,20,size=(batch_size * 3, output_size)).astype(np.float32)
    print('y shape', y.shape)
    print('########y########')
    print(y)
    print('########y########')
    input_data = torch.from_numpy(input_data)
    y = torch.from_numpy(y)
    # Init dataset
    dataset = Data.TensorDataset(input_data, y)
    train_loader = Data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Set optimizer and loss function
    optimizer = torch.optim.Adam(bob.parameters(), lr=0.0001)
    criterion = nn.MSELoss()

    loss_curve = []
    for epoch in range(500):
        for i, (x, y) in enumerate(train_loader):
            # If use cuda please get x and y to device.
            # x = x.to(device)
            # y = y.to(device)
            h = bob.init_hidden(batch_size)
            pre_y, _ = bob(x, h)

            loss = criterion(pre_y, y)
            # Backpropagation
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('loss: ', loss.data.item())
            loss_curve.append(loss.data.item())
            score = r2_score(y, pre_y.detach().numpy())
            print('r2: ', score)


