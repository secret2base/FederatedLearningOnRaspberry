import torch.nn as nn
import torch.nn.functional as F
import importlib
import torch
import pickle  # for pkl file reading
import os
import sys
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
import time
import scipy.io
import torchvision.transforms as transforms
from config import SERVER_ADDR, SERVER_PORT
from utils import recv_msg, send_msg
import socket
import struct
from torchvision import transforms
import math


# read data set from pkl files
class TwoConvOneFc(nn.Module):
    def __init__(self, input_shape, out_dim):
        super(TwoConvOneFc, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, out_dim)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out


class CifarCnn(nn.Module):
    def __init__(self, input_shape, out_dim):
        super(CifarCnn, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(64 * 5 * 5, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, out_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                stdv = 1. / math.sqrt(m.weight.size(1))
                m.weight.data.uniform_(-stdv, stdv)
                if m.bias is not None:
                    m.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


def read_data(data_dir):
    """Parses data in given train and test data directories

    Assumes:
        1. the data in the input directories are .json files with keys 'users' and 'user_data'
        2. the set of train set users is the same as the set of test set users

    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data (ndarray)
        test_data: dictionary of test data (ndarray)
    """

    # clients = []
    # groups = []
    data = {}
    print('>>> Read data from:', data_dir)

    # open training dataset pkl files
    with open(data_dir, 'rb') as inf:
        cdata = pickle.load(inf)

    data.update(cdata)

    data = MiniDataset(data['x'], data['y'])

    return data


class MiniDataset(Dataset):
    def __init__(self, data, labels):
        super(MiniDataset, self).__init__()
        self.data = np.array(data)
        self.labels = np.array(labels).astype("int64")

        if self.data.ndim == 4 and self.data.shape[3] == 3:
            self.data = self.data.reshape(-1, 16, 16, 3).astype("uint8")
            self.transform = transforms.Compose(
                [transforms.RandomHorizontalFlip(),
                 transforms.RandomCrop(32, 4),
                 transforms.ToTensor(),
                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                 ]
            )
        elif self.data.ndim == 4 and self.data.shape[3] == 1:
            self.transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.1307,), (0.3081,))
                 ]
            )
        elif self.data.ndim == 3:
            self.data = self.data.reshape(-1, 28, 28, 1).astype("uint8")
            self.transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.1307,), (0.3081,))
                 ]
            )
        else:
            self.data = self.data.astype("float32")
            self.transform = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        data, target = self.data[index], self.labels[index]

        if self.data.ndim == 4 and self.data.shape[3] == 3:
            data = Image.fromarray(data)

        if self.transform is not None:
            data = self.transform(data)

        return data, target


# Model for MQTT_IOT_IDS dataset
class Logistic(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Logistic, self).__init__()
        self.layer = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        logit = self.layer(x)
        return logit


def local_test(model, test_dataloader):
    model.eval()
    print("test")
    test_loss = test_acc = test_total = 0.
    with torch.no_grad():
        for x, y in test_dataloader:
            pred = model(x)
            loss = criterion(pred, y)
            _, predicted = torch.max(pred, 1)
            correct = predicted.eq(y).sum()
            test_acc += correct.item()
            test_loss += loss.item() * y.size(0)
            test_total += y.size(0)
    return test_acc / test_total, test_loss / test_total


# socket
sock = socket.socket()
sock.connect((SERVER_ADDR, SERVER_PORT))
print('---------------------------------------------------------------------------')
try:
    while True:
        msg = recv_msg(sock, 'MSG_INIT_SERVER_TO_CLIENT')
        options = msg[1]
        cid = msg[2]

        # Training parameters
        lr_rate = options['lr']  # Initial learning rate
        weight_decay = 0.99  # Learning rate decay
        num_epoch = options['num_epoch']  # Local epoches
        batch_size = options['batch_size']  # Data sample for training per comm. round
        # num_round = 200  # Communication rounds
        num_round = options['num_round']

        model = Logistic(784, 10)
        # model = Logistic(2500, 6)
        # model = Logistic(1024, 10)
        # model = TwoConvOneFc((3,16,16), 10)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, weight_decay, last_epoch=-1)
        criterion = torch.nn.CrossEntropyLoss()

        # Import the data set
        file_name = './mnist/niid/mnist' + str(cid) + '.pkl'
        train_data = read_data(file_name)
        # train_data, test_data = read_data('./usps/usps1.pkl', './usps/usps_test.pkl')

        print('data read successfully')

        # make the data loader

        train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                                   batch_size=batch_size,
                                                   shuffle=True)

        print('Make dataloader successfully')

        local_train_time = []
        local_wait_time = []
        msg_send_time = []
        while True:
            print('---------------------------------------------------------------------------')
            init = time.time()
            msg = recv_msg(sock, 'MSG_WEIGHT_TAU_SERVER_TO_CLIENT')
            is_last_round = msg[1]
            global_model_weights = msg[2]
            round_i = msg[3]

            model.load_state_dict(global_model_weights)

            # model.train()
            #
            # for iter in range(num_epoch):
            #     for batch_idx, (x,y) in enumerate(train_loader):
            #         optimizer.zero_grad()
            #         pred = model(x)
            #         loss = criterion(pred,y)
            #         loss.backward()
            #         optimizer.step()
            #     print('Global Round:'+ round_i+'\t'+'Local Epoch:')

            start = time.time()
            model.train()

            x, y = next(iter(train_loader))
            # print('Round:', round)
            for i in range(num_epoch):
                # x = Variable(x)
                # y = Variable(y)
                optimizer.zero_grad()
                pred = model(x)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()
            end = time.time()
            # acc, loss = local_test(model=model, test_dataloader=test_loader)
            msg = ['MSG_WEIGHT_TIME_SIZE_CLIENT_TO_SERVER', model.state_dict()]
            send_msg(sock, msg)
            initd = time.time()
            #  print("loss:", loss, "    acc:", acc)
            local_train_time.append(end - start)
            local_wait_time.append(start - init)
            msg_send_time.append(initd - end)
            if is_last_round:
                saveTitle = 'local_' + 'T' + str(options['num_round']) + 'E' + str(options['num_epoch']) + 'B' + str(
                    options['batch_size'])
                saveTitle1 = 'local_wait' + 'T' + str(options['num_round']) + 'E' + str(
                    options['num_epoch']) + 'B' + str(options['batch_size'])
                saveTitle2 = 'local_send' + 'T' + str(options['num_round']) + 'E' + str(
                    options['num_epoch']) + 'B' + str(options['batch_size'])
                scipy.io.savemat(saveTitle2 + '_time' + '.mat', mdict={saveTitle2 + '_time': msg_send_time})
                scipy.io.savemat(saveTitle + '_time' + '.mat', mdict={saveTitle + '_time': local_train_time})
                scipy.io.savemat(saveTitle1 + '_time' + '.mat', mdict={saveTitle1 + '_time': local_wait_time})
                break

except (struct.error, socket.error):
    print('Server has stopped')
    pass
