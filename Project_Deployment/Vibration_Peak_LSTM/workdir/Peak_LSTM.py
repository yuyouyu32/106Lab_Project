import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sys
from utils import _add_error_xml, _add_info_xml, _read_parameters

class LSTMNet(nn.Module):
    """
        The LSTMNet model that is used to formant
    """
 
    def __init__(self, input_size: int, output_size: int, hidden_dim: int,
                 n_layers: int, seq_length: int, bidirectional: bool=True, drop_prob: float=0.1):
        """
            Initialize the model by setting up the layers.
        """
        super(LSTMNet, self).__init__()
 
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
        # assert x.shape==(batch_size,self.seq_length, self.input_size)
        # If not batch_first, switch seq_length and batch_size position.
        x.transpose_(1,0)
        lstm_out, hidden = self.lstm(x, hidden)
        h_n, c_n = hidden
        # assert lstm_out.shape==(self.seq_length, batch_size, self.hidden_dim )
        # assert h_n.shape==c_n.shape==(self.n_layers*self.num_directions, batch_size, self.hidden_dim)
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

def getdata(filename, batch_size, seq_length):
    data = pd.read_csv(filename)
    dian = []
    dian_y = []
    for i in range(len(data)):
        if data.iloc[i, 1] is np.nan:
            list = []
            for j in range(seq_length):
                list.append([0, 0, 0, 0, 0])
            dian.append(list)
            dian_y.append([float(data.iloc[i, 0])])
        else:
            list = []
            for t in range(seq_length-len(data.iloc[i, 1].split(','))):
                list.append([0, 0, 0, 0, 0])
            for j in range(len(data.iloc[i, 1].split(','))):
                list.append([float(data.iloc[i, 1].split(',')[j]), float(data.iloc[i, 2].split(';')[j]),float(data.iloc[i, 3].split(';')[j]),float(data.iloc[i, 4].split(';')[j]),float(data.iloc[i, 5].split(';')[j])])
            dian.append(list)
            dian_y.append([float(data.iloc[i, 0])])
    dian =np.array(dian).astype(np.float32)
    dian_y = np.array(dian_y).astype(np.float32)
    X_train, X_val, y_train, y_val = train_test_split(dian, dian_y, test_size=0.3, random_state=100)
    X_train = torch.from_numpy(X_train)
    X_val = torch.from_numpy(X_val)
    y_train = torch.from_numpy(y_train)
    y_val = torch.from_numpy(y_val)
    dataset_x = Data.TensorDataset(X_train, y_train)
    train_loader = Data.DataLoader(dataset_x, batch_size=batch_size, shuffle=True, num_workers=0)
    return train_loader, X_val, y_val

def main(filename, hidden_dim=5, n_layers=1, seq_length=5, batch_size=50,epoch=10000, lr=0.005):
    train_loader, X_val, y_val = getdata(filename=filename, batch_size=batch_size, seq_length=seq_length)
    lstm = LSTMNet(input_size=5, output_size=1, hidden_dim=hidden_dim,
                n_layers=n_layers, seq_length=seq_length, bidirectional=True, drop_prob=0)

    # Set optimizer and loss function
    optimizer = torch.optim.Adam(lstm.parameters(), lr=lr)
    criterion = nn.SmoothL1Loss() # SmoothL1Loss
    test_r2 = []
    best_score = -1
    flag = 0
    for epoch in range(epoch):
        for i, (x, y) in enumerate(train_loader):
            # If use cuda please get x and y to device.
            # device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
            # x = x.to(device)
            # y = y.to(device)
            lstm.train()
            h = lstm.init_hidden(batch_size)
            pre_y, _ = lstm(x, h)

            loss = criterion(pre_y, y)
            # Backpropagation
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print('train loss: ', loss.data.item())
            score = r2_score(y, pre_y.detach().numpy())
            # print('train r2: ', score)
            # train_loss.append(loss.data.item())
            # train_r2.append(score)
        lstm.eval()
        h_test = lstm.init_hidden(X_val.shape[0])
        pre_y_test, _ = lstm(X_val.clone(), h_test)
        pre_y_test = pre_y_test.detach()
        # loss_test = criterion(pre_y_test, y_val)
        score_test = r2_score(y_val, pre_y_test.numpy())
        # test_loss.append(loss_test.data.item())
        test_r2.append(score_test)
        lstm.train()
        if score_test >= best_score:
            best_score = score_test
            flag = 0
        else:
            flag += 1
        if flag == 50:
            break
        # print('test loss: ', loss_test.data.item())
        # print('test r2: ', score_test)
    torch.save(lstm, 'lstm.pt')
    best_score = format(best_score, ".4f")    
    plt.figure()
    plt.title("test_r2")
    plt.xlabel("epoch")
    plt.ylabel("r2")
    plt.plot(test_r2, label=f"bset r2:{best_score}")
    plt.legend()
    plt.savefig('test_r2.png')
    name = 'test_r2.png'
    return best_score, name

def run():
    # 读取参数
    try:
        parameters = _read_parameters()
        inputCSV = parameters["inputCSV"]
        hidden_dim = parameters['hidden_dim']
        n_layers = parameters['n_layers']
        batch_size = parameters['batch_size']
        epoch = parameters['epoch']
        lr = parameters['lr']
        seq_length = parameters['seq_length']
    except Exception as e:
        _add_error_xml("Parameters Error", str(e))
        with open('log.txt', 'a') as fp:
            fp.write('error\n')
        return
    try:
        best_score, name=main(inputCSV, n_layers=n_layers, hidden_dim=hidden_dim, seq_length=seq_length,
         batch_size=batch_size, epoch=epoch, lr=lr)
    except Exception as e:
        _add_error_xml("RNN Error", str(e))
        with open('log.txt', 'a') as fp:
            fp.write('error\n')
        return 
    
    try:
        _add_info_xml(picture_names=[name], r2={'Final score': best_score})
    except Exception as e:
        _add_error_xml("XML Error", str(e))
        with open('log.txt', 'a') as fp:
            fp.write('error\n')
    
    with open('log.txt', 'a') as fp:
        fp.write('finish\n')
    


if __name__ == "__main__":
    console = sys.stdout
    file = open("task_log.txt", "w")
    sys.stdout = file
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('You are using: ' + str(device))
    run()
    sys.stdout = console
    file.close
    