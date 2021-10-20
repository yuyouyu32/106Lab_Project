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


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    train, test = read_csv()
    features = train['feature']
    target = train['target']
    x_pred_1 = test['feature'][:, 0]
    x_pred_2 = test['feature'][:, 1:8]
    x_pred_3 = test['feature'][:, 8:11]
    y_pred = test['target'][:, 1]
    kf_num = 0
    kf = KFold(n_splits=10, shuffle=True, random_state=1)
    for train_index, test_index in kf.split(features):
        kf_num += 1
        if kf_num != 1:
            continue
        print("Train index", train_index)
        print("Test index", test_index)
        # 90
        # x_train = torch.from_numpy(features[train_index, :]).to(device)
        # y_train = torch.from_numpy(target[train_index]).to(device)
        # x_test = torch.from_numpy(features[test_index, :]).to(device)
        # y_test = torch.from_numpy(target[test_index]).to(device)
        # x_pred = torch.from_numpy(test['feature']).to(device)
        # y_pred = torch.from_numpy(test['target']).to(device)

        # 12
        x_train_1 = torch.from_numpy(features[train_index, 0][:, np.newaxis]).to(device)
        x_train_2 = torch.from_numpy(features[train_index, 1:8]).to(device)
        x_train_3 = torch.from_numpy(features[train_index, 8:11]).to(device)
        y_train = torch.from_numpy(target[train_index, 1]).to(device)    # neg_inflation
        # y_train = torch.from_numpy(train['target'][:, 1]).to(device)  # s_temp
        # y_train = torch.from_numpy(train['target'][:, 2]).to(device)    # e_temp
        # y_train = torch.from_numpy(train['target'][:, 3]).to(device)    # temp_range
        x_test_1 = torch.from_numpy(features[test_index, 0][:, np.newaxis]).to(device)
        x_test_2 = torch.from_numpy(features[test_index, 1:8]).to(device)
        x_test_3 = torch.from_numpy(features[test_index, 8:11]).to(device)
        y_test = torch.from_numpy(target[test_index, 1]).to(device)    # neg_inflation
        # y_test = torch.from_numpy(test['target_1'][:, 1]).to(device)  # s_temp
        # y_test = torch.from_numpy(test['target_1'][:, 2]).to(device)    # e_temp
        # y_test = torch.from_numpy(test['target_1'][:, 3]).to(device)    # temp_range
        x_pred_1 = torch.from_numpy(test['feature'][:, 0][:, np.newaxis]).to(device)
        x_pred_2 = torch.from_numpy(test['feature'][:, 1:8]).to(device)
        x_pred_3 = torch.from_numpy(test['feature'][:, 8:11]).to(device)
        y_pred = torch.from_numpy(test['target']).to(device)
        model = attModel(feature_dim=[1, 7, 3], new_feature_dim=[1, 3, 1])
        model.to(device)
        model.double()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

        epoch_num = 50000
        max_r2_test = -float('inf')
        epoch_index = 0
        r2_train_end = -float('inf')
        rmse_train_end = 0
        rmse_test_end = 0
        for epoch in range(epoch_num):
            def closure():
                optimizer.zero_grad()
                out = model(x_train_1, x_train_2, x_train_3)
                loss = criterion(torch.squeeze(out), torch.squeeze(y_train))
                loss.backward()
                return loss


            optimizer.step(closure)

            if epoch % 100 == 9:
                print('epoch : ', epoch)
                pred = model(x_test_1, x_test_2, x_test_3)
                pred_train = model(x_train_1, x_train_2, x_train_3)
                out = model(x_pred_1, x_pred_2, x_pred_3)
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
                # if max_r2_test <= r2_test:
                #     max_r2_test = r2_test
                #     r2_train_end = r2_train
                #     epoch_index = epoch
                #     rmse_test_end = rmse_test
                #     rmse_train_end = rmse_train
        # Save model.
        # torch_out = torch.onnx.export(model, tuple([x_1_test,x_2_test,x_3_test,x_4_test]),
        #                   "test.onnx",
        #                   export_params=True,
        #                   verbose=True)
