import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score
from utils import _add_error_xml, _add_info_xml, _read_parameters



def get_scaler(scaler):
    scalers = {
        "minmax": MinMaxScaler,
        "standard": StandardScaler,
        "maxabs": MaxAbsScaler,
        "robust": RobustScaler,
    }
    return scalers.get(scaler.lower())()


def read_data(path, target=[0]):
    dataframe = pd.read_csv(path, usecols=target, engine='python')
    dataset = dataframe.values

    columns = dataframe.columns
    # 将整型变为float
    dataset = dataset.astype('float32')
    return dataset, columns


def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)


def train_val_test_split(X, y, test_ratio):
    val_ratio = test_ratio / (1 - test_ratio)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_ratio, shuffle=False)
    return X_train, X_val, X_test, y_train, y_val, y_test


class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        super(RNNModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # RNN
        self.rnn = nn.RNN(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)

        # Forward propagation by passing in the input and hidden state into the model
        out, h0 = self.rnn(x, h0.detach())

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)
        return out


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        super(LSTMModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # LSTM
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)

        # Initializing cell state for first input with zeros
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        # Forward propagation by passing in the input, hidden state, and cell state into the model
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)

        return out


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        super(GRUModel, self).__init__()

        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim

        # GRU
        self.gru = nn.GRU(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)

        # Forward propagation by passing in the input and hidden state into the model
        out, _ = self.gru(x, h0.detach())

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)

        return out

def get_model(model, model_params):
    models = {
        "rnn": RNNModel,
        "lstm": LSTMModel,
        "gru": GRUModel,
    }
    return models.get(model.lower())(**model_params)


class Optimization:
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []
    
    def train_step(self, x, y):
        self.model.train()
        yhat = self.model(x)
        loss = self.loss_fn(y, yhat)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.item()
        
    def train(self, train_loader, batch_size=64, n_epochs=50, n_features=1):
        model_path = f'./{self.model.__class__.__name__}_{datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")}'

        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.view([batch_size, -1, n_features]).to(device)
                y_batch = y_batch.to(device)
                loss = self.train_step(x_batch, y_batch)
                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)

            with torch.no_grad():
                batch_val_losses = []
                for x_val, y_val in train_loader:
                    x_val = x_val.view([batch_size, -1, n_features]).to(device)
                    y_val = y_val.to(device)
                    self.model.eval()
                    yhat = self.model(x_val)
                    val_loss = self.loss_fn(y_val, yhat).item()
                    batch_val_losses.append(val_loss)
                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)

            if (epoch <= 10) | (epoch % 50 == 0):
                print(
                    f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t Validation loss: {validation_loss:.4f}"
                )

        torch.save(self.model.state_dict(), model_path)
        
    def evaluate(self, test_loader, columns, batch_size=4, n_features=1):
        with torch.no_grad():
            predictions = []
            values = []
            for x_test, y_test in test_loader:
                x_test = x_test.view([batch_size, -1, n_features]).to(device)

                y_test = y_test.to(device)
                self.model.eval()

                yhat = self.model(x_test)
                
                predictions.append(yhat.to(device).detach().cpu().numpy()[0])
                values.append(y_test.to(device).detach().cpu().numpy()[0])

        r2_list = {}

        predictions = np.array(predictions)
        values = np.array(values)

        for i in range(len(columns)):
            r2 = r2_score(predictions[:, i], values[:, i])
            r2_list[columns[i]] = r2

        return r2_list

    def predict(self, dataset, columns, batch_size=4, look_back=4, n_features=1, predict_num = 10): 
        prediction_list = []
        data = dataset[-look_back :]

        for i in range(predict_num):
            data = data.reshape([batch_size, -1, n_features]).to(device)

            self.model.eval()
            prediction = self.model(data)
            prediction_list.append(prediction.detach().cpu().numpy())

            if i < (look_back - 1):
                data = dataset[- look_back + i + 1: ].cpu().numpy()
                data = np.append(data, np.array(prediction_list))
            else :
                data = prediction_list[-look_back : ]

            data = torch.Tensor(data)

        prediction_list = [i[0].tolist() for i in prediction_list]
        predictions = pd.DataFrame(prediction_list, columns=columns)
        self.predictions = predictions
        # csv_predictions_name = f'./predictions/{self.model.__class__.__name__}_predictions_{datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")}.csv'
        csv_predictions_name = f'./result.csv'
        predictions.to_csv(csv_predictions_name, index=False)
        
        return predictions, csv_predictions_name

    def plot_losses(self):
        plt.plot(self.train_losses, label="Training loss")
        plt.plot(self.val_losses, label="Validation loss")
        plt.legend()
        plt.title("Losses")
        plt.show()
        plt.close()

    # 输出图片
    def plot_losses_save(self):
        plt.plot(self.train_losses, label="Training loss")
        plt.plot(self.val_losses, label="Validation loss")
        plt.legend()
        plt.title("Losses")
        pic_name = f'result.png'
        plt.savefig(pic_name)
        plt.close()
        return pic_name

    def plot_predictions(self):
        pic_name_list = {}
        for key,value in self.predictions.iteritems():
            plt.plot(value)
            plt.title(label=key)
            pic_name = f'pic{key}.png'
            plt.savefig(pic_name)
            pic_name_list[key] = pic_name
            plt.close()
        return pic_name_list

def main(inputCSV, start_index, end_index, look_back, predict_num, scaler, model, hidden_dim = 64, layer_dim = 3, batch_size = 4, 
         dropout = 0.2, n_epochs = 100, learning_rate = 1e-3, weight_decay = 1e-6):
    
    '''
    inputCSV : csv文件
    start_index : target起始下标
    end_index : target终止下标
    look_back : 窗口大小
    predict_num : 预测后续数据长度
    scaler : 归一化模型
    model : 循环神经网络名
    hidden_dim : 隐藏层数
    layer_dim : 网络堆叠层数
    batch_size : 一次训练所使用数据数量 batch_size过大会导致测试集为空，计算r2报错
    dropout : 神经元丢失率
    n_epochs : 训练批次
    learning_rate : 优化器学习率
    weight_decay : 优化器权值衰减
    
    '''

    column_indexs = list(range(start_index, end_index + 1, 1))
    dataset, columns = read_data(inputCSV, column_indexs)
    # columns = ['index'] + columns
    scaler = get_scaler('minmax')
    dataset = scaler.fit_transform(dataset)

    data_X, data_y = create_dataset(dataset, look_back)

    input_dim = look_back
    output_dim = len(columns)

    train_features = torch.Tensor(data_X)
    train_targets = torch.Tensor(data_y)

    train = TensorDataset(train_features, train_targets)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=True)

    model_params = {'input_dim': input_dim,
                    'hidden_dim' : hidden_dim,
                    'layer_dim' : layer_dim,
                    'output_dim' : output_dim,
                    'dropout_prob' : dropout}

    model = get_model(model, model_params)
    model.to(device)

    loss_fn = nn.MSELoss(reduction="mean")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    opt = Optimization(model=model, loss_fn=loss_fn, optimizer=optimizer)
    opt.train(train_loader, batch_size=batch_size, n_epochs=n_epochs, n_features=input_dim)
    pic_name = opt.plot_losses_save()

    r2_list = opt.evaluate(train_loader, columns, batch_size=1, n_features=input_dim)
    predictions, csv_predictions_name = opt.predict(torch.Tensor(dataset), columns, batch_size=1, n_features=input_dim, predict_num=predict_num)
    
    pic_predictions_list = opt.plot_predictions()
    pictures = [pic_name] + list(pic_predictions_list.values())
    return r2_list, pic_name, predictions, csv_predictions_name, pic_predictions_list, pictures

if __name__ == "__main__" :

    # 创建存储模型和图片的文件夹
    try:
        os.mkdir('models')
    except:
        pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('You are using: ' + str(device))
    # 读取参数
    parameters = _read_parameters()

    inputCSV = parameters["inputCSV"]
    start_index = parameters["start_index"]
    end_index = parameters["end_index"]
    look_back = parameters["look_back"]
    predict_num = parameters["predict_num"]
    scaler = parameters["scaler"]
    model = parameters["model"]
    hidden_dim = parameters["hidden_dim"]
    layer_dim = parameters["layer_dim"]
    batch_size = parameters["batch_size"]
    dropout = parameters["dropout"]
    n_epochs = parameters["n_epochs"]
    learning_rate = parameters["learning_rate"]
    weight_decay = parameters["weight_decay"]

    r2_list, pic_name, predictions, csv_predictions_name, pic_predictions_list, pictures = main(inputCSV, start_index, end_index, look_back, predict_num, scaler, model, hidden_dim, layer_dim, batch_size, dropout, n_epochs, learning_rate, weight_decay)
    
    result = pd.read_csv('./result.csv')
    try:
        _add_info_xml(picture_names=['result.png']+list(pic_predictions_list.values()), result=result)
    except Exception as e:
        _add_error_xml("XML Error", str(e))
        with open('log.txt', 'a') as fp:
            fp.write('error\n')
    
    with open('log.txt', 'a') as fp:
        fp.write('finish\n')


    print(r2_list)
    print(predictions)
    print(csv_predictions_name)
    print(pictures)
