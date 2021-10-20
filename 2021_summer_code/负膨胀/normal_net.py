import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as rmse_score
from sklearn.model_selection import KFold
from read_data import read_csv
from main import attModel
import numpy as np


class netModel(nn.Module):
    def __init__(self):
        super(netModel, self).__init__()
        self.linear1 = nn.Linear(15, 32)
        self.linear2 = nn.Linear(32, 8)
        self.linear3 = nn.Linear(8, 1)

    def forward(self, x):
        out = nn.functional.relu(self.linear1(x))
        out = nn.functional.relu(self.linear2(out))
        out = nn.functional.relu(self.linear3(out))
        return out


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    train, test = read_csv()
    features = train['feature']
    target = train['target'][:, 3]
    x_pred = torch.from_numpy(test['feature']).to(device)
    y_pred = torch.from_numpy(test['target'][:, 3]).to(device)
    kf_num = 0
    kf = KFold(n_splits=10, shuffle=True, random_state=1)
    for train_index, test_index in kf.split(features):
        kf_num += 1
        if kf_num != 3:
            continue
        print("Train index", train_index)
        print("Test index", test_index)
        x_train = torch.from_numpy(features[train_index, :]).to(device)
        y_train = torch.from_numpy(target[train_index]).to(device)
        x_test = torch.from_numpy(features[test_index, :]).to(device)
        y_test = torch.from_numpy(target[test_index]).to(device)
        model = netModel()
        model.to(device)
        model.double()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        epoch_num = 50000
        for epoch in range(epoch_num):
            def closure():
                optimizer.zero_grad()
                out = model(x_train)
                loss = criterion(torch.squeeze(out), torch.squeeze(y_train))
                loss.backward()
                return loss


            optimizer.step(closure)

            if epoch % 100 == 9:
                print('epoch : ', epoch)
                pred = model(x_test)
                pred_train = model(x_train)
                out = model(x_pred)
                test_loss = criterion(torch.squeeze(pred), torch.squeeze(y_test))
                r2_train = r2_score(torch.squeeze(y_train).cpu().detach().numpy(),
                                    torch.squeeze(pred_train).cpu().detach().numpy())
                # rmse_train = rmse_score(torch.squeeze(y_train).cpu().detach().numpy(),
                #                         torch.squeeze(pred_train).cpu().detach().numpy())
                r2_test = r2_score(torch.squeeze(y_test).cpu().detach().numpy(),
                                   torch.squeeze(pred).cpu().detach().numpy())
                # rmse_test_1 = rmse_score(torch.squeeze(y_test_1).cpu().detach().numpy(),
                #                          torch.squeeze(pred).cpu().detach().numpy())

                # rmse_test_2 = rmse_score(torch.squeeze(y_test_2).cpu().detach().numpy(),
                #                          torch.squeeze(pred).cpu().detach().numpy())
                print("#################")
                print('test loss = ', test_loss.cpu().detach().numpy())
                print('r2_train = ', r2_train)
                print('r2_test = ', r2_test)
                print("test", torch.squeeze(y_pred).cpu().detach().numpy())
                print("pred", torch.squeeze(out).cpu().detach().numpy())
