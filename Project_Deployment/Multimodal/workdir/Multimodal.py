from utils import _add_error_xml, _add_info_xml, _read_parameters
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use('Agg')


def read_data(pic_path, original_path, start, end, target_name):
    picture_data = pd.read_csv(pic_path).astype('double').values
    original_data = pd.read_csv(original_path).astype('double')
    targets = original_data[target_name].values
    original_data = original_data.values[:, start:end]
    combine_data = np.concatenate((picture_data, original_data), axis=1)
    x_train, x_test, y_train, y_test = train_test_split(
        combine_data, targets, test_size=0.33, random_state=42)
    x_train_pics = torch.from_numpy(x_train[:, :-16]).reshape(-1, 1, 24, 21)
    x_test_pics = torch.from_numpy(x_test[:, :-16]).reshape(-1, 1, 24, 21)
    x_train_other = torch.from_numpy(x_train[:, -16:])
    x_test_other = torch.from_numpy(x_test[:, -16:])
    y_train = torch.from_numpy(y_train)
    y_test = torch.from_numpy(y_test)
    dataloader = [x_train_pics, x_test_pics,
                  x_train_other, x_test_other, y_train, y_test]
    return dataloader


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
    def __init__(self, feature_dim, num_heads=1, dropout=0.0):
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
        context = context.view(
            batch_size, -1, self.dim_per_head * self.num_heads)

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


class multimodel(nn.Module):
    def __init__(self, fearture_dim):
        super(multimodel, self).__init__()
        # x: (1, 1, 24, 21)
        self.pic_block = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=3, stride=3),  # (1, 1, 8, 7)
            # nn.BatchNorm2d(3),
            # nn.ReLU(inplace=True),
            nn.Conv2d(3, 8, kernel_size=3, stride=1),  # (1, 1, 6, 5)
            # nn.BatchNorm2d(3),
            # nn.ReLU(inplace=True),
            nn.Conv2d(8, 4, kernel_size=3, stride=1),  # (1, 1, 4, 3)
            # nn.BatchNorm2d(3),
            # nn.ReLU(inplace=True),
            nn.Conv2d(4, 1, kernel_size=2, stride=1),  # (1, 1, 3, 2)
            # nn.BatchNorm2d(1),
            # nn.ReLU(inplace=True),
        )
        if fearture_dim % 2 == 0:
            num_heads = 2
        else:
            num_heads = 1
        self.att1 = multi_heads_self_attention(
            feature_dim=fearture_dim, num_heads=num_heads)
        self.linear_1 = nn.Sequential(
            nn.Linear(fearture_dim, 8),
            nn.ReLU(inplace=True),
        )
        self.linear_2 = nn.Sequential(
            nn.Linear(6, 4),
            nn.ReLU(inplace=True),
        )
        self.att2 = multi_heads_self_attention(feature_dim=12, num_heads=2)
        self.output = nn.Sequential(
            nn.BatchNorm1d(12),
            nn.Linear(12, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 4),
            nn.ReLU(inplace=True),
            nn.Linear(4, 1),
        )

    def forward(self, pics, other):
        out_1, _ = self.att1(other, other, other)
        out_1 = self.linear_1(out_1)
        out_pic = self.pic_block(pics)
        out_2 = self.linear_2(torch.flatten(out_pic, start_dim=1))

        out = torch.cat([out_1, out_2], dim=1)
        out_3, _ = self.att2(out, out, out)
        out = self.output(out_3)
        return out


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    # 也可以判断是否为conv2d，使用相应的初始化方式
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    # 是否为批归一化层
    elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def exp(multimodel, data, fearture_dim, epoch_num=50000, learning_rate=0.0001):
    # Data loader
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device("cpu")
    # to cuda or cpu
    for index, item in enumerate(data):
        data[index] = item.to(device)
    x_train_pics, x_test_pics, x_train_other, x_test_other, y_train, y_test = data

    model = multimodel(fearture_dim)
    model.to(device)
    model.double()
    model.apply(weight_init)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.9)
    for epoch in range(epoch_num):
        # Traing
        model.train()
        optimizer.zero_grad()
        out = model(x_train_pics, x_train_other)
        loss = criterion(torch.squeeze(out), torch.squeeze(y_train))
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Loss Record
        model.eval()

        if epoch % 10 == 9:
            pred = model(x_test_pics, x_test_other)
            pred_train = model(x_train_pics, x_train_other)
            train_loss = criterion(torch.squeeze(
                pred_train.cpu()), torch.squeeze(y_train.cpu())).data.item()
            loss = criterion(torch.squeeze(pred.cpu()),
                             torch.squeeze(y_test.cpu())).data.item()
            r2 = r2_score(torch.squeeze(y_test.cpu()).detach(
            ).numpy(), torch.squeeze(pred.cpu()).detach().numpy())

            with open(f'./trainloss.txt', 'a+') as f:
                f.write(f'{train_loss}\n')
            with open(f'./testloss.txt', 'a+') as f:
                f.write(f'{loss}\n')
            with open(f'./r2.txt', 'a+') as f:
                f.write(f'{r2}\n')
            if epoch % 5000 == 9:
                print('epoch : ', epoch)
                print('train loss:', train_loss)
                print('test loss:', loss)
                print('r2:', r2)

    import time
    model_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    torch.save(model.state_dict(), f'./{model_time}.pth')
    model.eval()
    pred = model(x_test_pics, x_test_other)

    return r2_score(torch.squeeze(y_test.cpu()).detach().numpy(), torch.squeeze(pred.cpu()).detach().numpy())


def draw_log(target_name):
    Attention_BN_trainloss = np.loadtxt(f'./trainloss.txt')
    Attention_BN_testloss = np.loadtxt(f'./testloss.txt')
    length = len(Attention_BN_trainloss)
    fig, ax = plt.subplots(figsize=(14, 7))  # 创建图实例
    x = np.arange(0, length//50)  # 创建x的取值范围
    markers_on = np.arange(0, len(x), 5)
    ax.plot(x, Attention_BN_trainloss[0:length:50], label='Train Loss',
            color='b', linestyle='-', marker='o', markevery=markers_on)
    ax.plot(x, Attention_BN_testloss[0:length:50], label='Tess Loss',
            color='r', linestyle='--', marker='o', markevery=markers_on)

    plt.yscale('symlog')
    plt.grid(alpha=0.5)
    plt.xticks(rotation=45)  # 坐标轴旋转45度
    plt.style.use('seaborn-dark')  # 给图片换不同的风格
    ax.set_xlabel('epochs', fontdict={'weight': 'normal', 'size': 15})
    ax.set_ylabel('Loss', fontdict={'weight': 'normal', 'size': 15})
    ax.set_title(f'Loss curves on {target_name} target values in the NIMS steel dataset', fontdict={
                 'weight': 'normal', 'size': 20})
    ax.legend(loc='upper right', prop={'weight': 'normal', 'size': 15})
    plt.savefig(f'./Loss.png', dpi=750, bbox_inches='tight')


def main():
    try:
        parameters = _read_parameters()
        pic_path = parameters["pictureCSV"]
        original_path = parameters["originalCSV"]
        start = parameters["start_index"]
        end = parameters["end_index"]
        target_name = parameters["target_name"]
        epochs = parameters["epochs"]
        learning_rate = parameters["learning_rate"]
    except Exception as e:
        _add_error_xml("Parameters Error", str(e))
        with open('log.txt', 'a') as fp:
            fp.write('error\n')
        return

    try:
        data = read_data(pic_path, original_path, start-1, end, target_name)
        r2 = exp(multimodel=multimodel, data=data, fearture_dim=end - start + 1, epoch_num=epochs, learning_rate=learning_rate)
        draw_log(target_name)
    except Exception as e:
        _add_error_xml("Multimodal Error", str(e))
        with open('log.txt', 'a') as fp:
            fp.write('error\n')
        return

    try:
        _add_info_xml(picture_names=['Loss.png'], score=r2)
    except Exception as e:
        _add_error_xml("XML Error", str(e))
        with open('log.txt', 'a') as fp:
            fp.write('error\n')

    with open('log.txt', 'a') as fp:
        fp.write('finish\n')


if __name__ == "__main__":
    import sys
    console = sys.stdout
    file = open("task_log.txt", "w")
    sys.stdout = file
    main()
    sys.stdout = console
    file.close
