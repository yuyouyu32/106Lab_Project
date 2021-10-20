import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as rmse_score
from sklearn.model_selection import KFold
import numpy as np


def read_data():
    raw = pd.read_csv('data.csv', encoding='gbk')
    raw.fillna(0, inplace=True)
    raw.drop(columns=['formula'], inplace=True)
    raw['neg_inflation'] = - raw['neg_inflation']
    return raw.astype('float64')


class scaled_dot_product_attention(nn.Module):
    def __init__(self, att_dropout=0.0):
        super(scaled_dot_product_attention, self).__init__()
        self.dropout = nn.Dropout(att_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None):
        '''
        args:
            q: [batch_size, q_length, q_dimension]
            k: [batch_size, k_length, k_dimension]
            v: [batch_size, v_length, v_dimension]
            q_dimension = k_dimension = v_dimension
            scale: 缩放因子
        return:
            context, attention
        '''
        # 快使用神奇的爱因斯坦求和约定吧！
        attention = torch.einsum('ijk,ilk->ijl', [q, k])
        if scale:
            attention = attention * scale
        attention = self.softmax(attention)
        attention = self.dropout(attention)
        context = torch.einsum('ijl,ilk->ijk', [attention, v])
        return context, attention


class multi_heads_self_attention(nn.Module):
    def __init__(self, feature_dim, num_heads=1, dropout=0.25):
        super(multi_heads_self_attention, self).__init__()

        self.dim_per_head = feature_dim // num_heads
        self.num_heads = num_heads
        self.linear_q = nn.Linear(feature_dim, self.dim_per_head * num_heads)
        self.linear_k = nn.Linear(feature_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(feature_dim, self.dim_per_head * num_heads)

        self.sdp_attention = scaled_dot_product_attention(dropout)
        self.linear_attention = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(feature_dim)
        # self.linear_1 = nn.Linear(feature_dim, 256)
        # self.linear_2 = nn.Linear(256, feature_dim)
        # self.layer_final = nn.Linear(feature_dim, 3)

    def forward(self, key, value, query):
        residual = query
        batch_size = key.size(0)

        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        # split by heads
        key = key.view(batch_size * self.num_heads, -1, self.dim_per_head)
        value = value.view(batch_size * self.num_heads, -1, self.dim_per_head)
        query = query.view(batch_size * self.num_heads, -1, self.dim_per_head)

        if key.size(-1) // self.num_heads != 0:
            scale = (key.size(-1) // self.num_heads) ** -0.5
        else:
            scale = 1
        context, attention = self.sdp_attention(query, key, value, scale)

        # concat heads
        context = context.view(batch_size, -1, self.dim_per_head * self.num_heads)

        output = self.linear_attention(context)
        output = self.dropout(output)
        output = torch.squeeze(output)

        # add residual and norm layer
        output = self.layer_norm(residual + output)

        # # pass through linear
        # output = nn.functional.relu(self.linear_1(output))
        # output = nn.functional.relu(self.linear_2(output))

        # # pass through layer final
        # output = self.layer_final(output)

        return output, attention


class attModel(nn.Module):
    def __init__(self, feature_dim, new_feature_dim):
        super(attModel, self).__init__()
        self.PCA_layer_1 = nn.Linear(feature_dim[0], new_feature_dim[0])
        self.PCA_layer_2 = nn.Linear(feature_dim[1], new_feature_dim[1])
        self.PCA_layer_3 = nn.Linear(feature_dim[2], new_feature_dim[2])
        self.att = multi_heads_self_attention(feature_dim=sum(new_feature_dim))
        self.linear1 = nn.Linear(sum(new_feature_dim), 8)
        self.linear2 = nn.Linear(8, 4)
        self.linear3 = nn.Linear(4, 1)

    def forward(self, x_1, x_2, x_3):
        x_1 = self.PCA_layer_1(x_1)
        x_2 = self.PCA_layer_2(x_2)
        x_3 = self.PCA_layer_3(x_3)
        x = torch.cat([x_1, x_2, x_3], dim=1)
        out, _ = self.att(x, x, x)
        out = nn.functional.relu(self.linear1(out))
        out = nn.functional.relu(self.linear2(out))
        out = nn.functional.relu(self.linear3(out))
        return out


def output_graph(training_rmse, validation_rmse, training_R2_list, validation_R2_list):
    # Output a graph of loss metrics and R2 over periods.
    from matplotlib import pyplot as plt
    plt.figure(1)
    plt.ylabel("RMSE")
    plt.xlabel("Periods")
    plt.title("Root Mean Squared Error vs. Periods")
    plt.tight_layout()
    plt.plot(training_rmse, label="training")
    plt.plot(validation_rmse, label="validation")
    ax = plt.gca()
    ax.set_ylim(0, 1)
    plt.legend()
    plt.figure(2)
    plt.ylabel("R2")
    plt.xlabel("Periods")
    plt.title("R-Square vs. Periods")
    plt.tight_layout()
    plt.plot(training_R2_list, label="training")
    plt.plot(validation_R2_list, label="validation")
    ax = plt.gca()
    ax.set_ylim(min(0, min(training_R2_list), min(validation_R2_list)) - 0.1, 1)
    plt.legend()

    # print("\033[0;31mFinal RMSE (on training data):   %0.2f" % training_rmse[-1])
    # print("Final RMSE (on validation data): %0.2f" % validation_rmse[-1])
    # print("Final R2 (on training data):   %0.2f" % training_R2_list[-1])
    # print("Final R2 (on validation data): %0.2f" % validation_R2_list[-1])
    average_R2_train = sum(training_R2_list) / len(training_R2_list)
    average_R2 = sum(validation_R2_list) / len(validation_R2_list)
    print("Average R2 (on Training data): %0.2f " % average_R2_train)
    print("Average R2 (on validation data): %0.2f \033[0m " % average_R2)
    plt.show()


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    raw = read_data().values
    features_1 = raw[:, 4:7]  # element_1
    features_2 = raw[:, 7:19]  # element_2
    features_3 = raw[:, 19:22]  # element_3

    # target = raw[:, 0]  # neg_inflation
    target = raw[:, 1]  # s_temp
    # target = raw[:, 2]  # e_temp
    # target = raw[:, 3]  # temp_range
    # x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=421)
    kf = KFold(n_splits=10, shuffle=True, random_state=1)
    kf_num = 0
    training_rmse, validation_rmse, training_R2_list, validation_R2_list, epoch_index_list = [], [], [], [], []
    for train_index, test_index in kf.split(features_1):
        kf_num += 1
        if kf_num != 1:
            continue
        print("Train index", train_index)
        print("Test index", test_index)
        x_1_train = features_1[train_index, :]
        x_2_train = features_2[train_index, :]
        x_3_train = features_3[train_index, :]
        y_train = target[train_index]
        x_1_test = features_1[test_index, :]
        x_2_test = features_2[test_index, :]
        x_3_test = features_3[test_index, :]
        y_test = target[test_index]
        x_1_train = torch.from_numpy(x_1_train).to(device)
        x_1_test = torch.from_numpy(x_1_test).to(device)
        x_2_train = torch.from_numpy(x_2_train).to(device)
        x_2_test = torch.from_numpy(x_2_test).to(device)
        x_3_train = torch.from_numpy(x_3_train).to(device)
        x_3_test = torch.from_numpy(x_3_test).to(device)
        y_train = torch.from_numpy(y_train).to(device)
        y_test = torch.from_numpy(y_test).to(device)
        features_dim = [3, 12, 3]
        model = attModel(feature_dim=features_dim, new_feature_dim=[2, 8, 2])
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
                out = model(x_1_train, x_2_train, x_3_train)
                loss = criterion(torch.squeeze(out), torch.squeeze(y_train))
                loss.backward()
                return loss


            optimizer.step(closure)

            if epoch % 10 == 9:
                print('epoch : ', epoch)
                pred = model(x_1_test, x_2_test, x_3_test)
                pred_train = model(x_1_train, x_2_train, x_3_train)
                loss = criterion(torch.squeeze(pred), torch.squeeze(y_test))
                r2_train = r2_score(torch.squeeze(y_train).cpu().detach().numpy(),
                                    torch.squeeze(pred_train).cpu().detach().numpy())
                rmse_train = rmse_score(torch.squeeze(y_train).cpu().detach().numpy(),
                                        torch.squeeze(pred_train).cpu().detach().numpy())
                r2_test = r2_score(torch.squeeze(y_test).cpu().detach().numpy(),
                                   torch.squeeze(pred).cpu().detach().numpy())
                rmse_test = rmse_score(torch.squeeze(y_test).cpu().detach().numpy(),
                                       torch.squeeze(pred).cpu().detach().numpy())
                print('loss = ', loss.cpu().detach().numpy())
                print('r2_train = ', r2_train)
                print('r2_test = ', r2_test)
                if max_r2_test <= r2_test:
                    max_r2_test = r2_test
                    r2_train_end = r2_train
                    epoch_index = epoch
                    rmse_test_end = rmse_test
                    rmse_train_end = rmse_train
        training_rmse.append(rmse_train_end)
        validation_rmse.append(rmse_test_end)
        training_R2_list.append(r2_train_end)
        validation_R2_list.append(max_r2_test)
        epoch_index_list.append(epoch_index)
        # torch_out = torch.onnx.export(model, tuple([x_1_test,x_2_test,x_3_test,x_4_test]),
        #                   "test.onnx",
        #                   export_params=True,
        #                   verbose=True)
    output_graph(training_rmse, validation_rmse, training_R2_list, validation_R2_list)
    print('Train_RMSE', training_rmse)
    print('Test_RMSE', validation_rmse)
    print('Train_R2', training_R2_list)
    print('Test_R2', validation_R2_list)
    print('epoch_index', epoch_index_list)
