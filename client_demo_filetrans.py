import torch.nn as nn
import torch.nn.functional as F
import importlib
import torch
import pickle #for pkl file reading
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
from utils import recv_msg, send_msg, file_send
import socket
import struct
from torchvision import transforms
import math

#read data set from pkl files
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
        self.fc1 = nn.Linear(64*5*5, 512)
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

    #clients = []
    #groups = []
    data = {}
    print('>>> Read data from:',data_dir)

    #open training dataset pkl files
    with open(data_dir, 'rb') as inf:
        cdata = pickle.load(inf)
        
    data.update(cdata)

    data= MiniDataset(data['x'], data['y'])

    return data

class MiniDataset(Dataset):
    def __init__(self, data, labels):
        super(MiniDataset, self).__init__()
        self.data = np.array(data)
        self.labels = np.array(labels).astype("int64")

        if self.data.ndim == 4 and self.data.shape[3] == 3:
            self.data = self.data.reshape(-1,16,16,3).astype("uint8")
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
    def __init__(self):
        super(Logistic, self).__init__()
        self.relu = nn.ReLU()
        self.hidden_layer1 = nn.Linear(784,200)
        self.hidden_layer2 = nn.Linear(200,10)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.hidden_layer1(x)
        x =self.relu(x)
        x = self.hidden_layer2(x)
        return self.softmax(x)

class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.max_pool1 = nn.MaxPool2d(2)
        self.max_pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.max_pool1(x)
        x = F.relu(self.conv2(x))
        x = self.max_pool2(x)
        x = x.view(-1, 4*4*64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def local_test(model,test_dataloader):
    model.eval()
    print("test")
    test_loss = test_acc = test_total = 0.
    with torch.no_grad():
        for x,y in test_dataloader:
            pred = model(x)
            loss = criterion(pred, y)
            _, predicted = torch.max(pred, 1)
            correct = predicted.eq(y).sum()
            test_acc += correct.item()
            test_loss += loss.item() * y.size(0)
            test_total += y.size(0)
    return test_acc/test_total, test_loss/test_total


sock = socket.socket()
sock.connect((SERVER_ADDR, SERVER_PORT))
print('---------------------------------------------------------------------------')
try:
    while True:
        msg, temp = recv_msg(sock, 'MSG_INIT_SERVER_TO_CLIENT')
        options = msg[1]
        cid = msg[2]
        print("cid:", cid)

        
        # Training parameters
        lr_rate = options['lr']  # Initial learning rate
        weight_decay = 0.99  # Learning rate decay
        num_epoch = options['num_epoch']  # Local epoches
        batch_size = options['batch_size']  # Data sample for training per comm. round
        model = options['model']
        num_round = options['num_round']

        #model = Logistic(784, 10)
        if model == 'cnn':
            model = MNIST_CNN()
        else:
            model = Logistic()
        #model = Logistic(2500, 6)
        # model = Logistic(1024, 10)
        # model = TwoConvOneFc((3,16,16), 10)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr_rate,momentum=0.9)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,weight_decay,last_epoch=-1)
        criterion = torch.nn.CrossEntropyLoss()

        # Import the data set
        file_name = 'iid/MNIST_iid' + str(cid) + '.pkl'
        train_data = read_data(file_name)

        print('data read successfully')

        # make the data loader
        
        train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                                   batch_size=batch_size,
                                                   shuffle=True)

        print('Make dataloader successfully')

        msg_recv_time = []
        msg_recv_process_time = []
        local_train_time = []
        local_wait_time = []
        msg_send_time = []
        msg_send_process_time = []
        msg_wait_send_time = []
        init_point = []
        start_point = []

        while True:
            print('---------------------------------------------------------------------------')
            init = time.time()
            process_recv = time.time()
            msg, recv_time = recv_msg(sock, 'MSG_WEIGHT_TAU_SERVER_TO_CLIENT')

            is_last_round = msg[1]
            global_model_weights = msg[2]
            round_i = msg[3]

            model.load_state_dict(global_model_weights)

            start = time.time()

            print('last_round:', is_last_round)
            # 文件传输部分
            # if is_last_round:
            #     Titlelist = []
            #     addr = str(sock.getsockname())
            #     addr = addr.replace('.', '_')
            #     addr = addr.replace('(', '')
            #     addr = addr.replace(')', '')
            #     addr = addr.replace('\'', '')
            #     addr = addr.replace(',', '_')
            #     addr = addr.replace(' ', '')
            #     print('addr:', addr)
            #
            #     saveTitle = 'local_train' + 'T' + str(options['num_round']) + 'E' + str(
            #         options['num_epoch']) + 'B' + str(options['batch_size']) + 'id' + str(addr)
            #     saveTitle1 = 'local_wait' + 'T' + str(options['num_round']) + 'E' + str(
            #         options['num_epoch']) + 'B' + str(options['batch_size']) + 'id' + str(addr)
            #     saveTitle2 = 'local_send' + 'T' + str(options['num_round']) + 'E' + str(
            #         options['num_epoch']) + 'B' + str(options['batch_size']) + 'id' + str(addr)
            #     saveTitle3 = 'local_recv' + 'T' + str(options['num_round']) + 'E' + str(
            #         options['num_epoch']) + 'B' + str(options['batch_size']) + 'id' + str(addr)
            #     saveTitle4 = 'local_recv_process' + 'T' + str(options['num_round']) + 'E' + str(
            #         options['num_epoch']) + 'B' + str(options['batch_size']) + 'id' + str(addr)
            #     saveTitle5 = 'local_send_process' + 'T' + str(options['num_round']) + 'E' + str(
            #         options['num_epoch']) + 'B' + str(options['batch_size']) + 'id' + str(addr)
            #     saveTitle6 = 'local_send_wait' + 'T' + str(options['num_round']) + 'E' + str(
            #         options['num_epoch']) + 'B' + str(options['batch_size']) + 'id' + str(addr)
            #     saveTitle7 = 'local_init_point' + 'T' + str(options['num_round']) + 'E' + str(
            #         options['num_epoch']) + 'B' + str(
            #         options['batch_size']) + 'id' + str(addr)
            #     saveTitle8 = 'local_start_point' + 'T' + str(options['num_round']) + 'E' + str(
            #         options['num_epoch']) + 'B' + str(
            #         options['batch_size']) + 'id' + str(addr)
            #
            #     Titlelist.append(saveTitle + '_time' + '.mat')
            #     Titlelist.append(saveTitle1 + '_time' + '.mat')
            #     Titlelist.append(saveTitle2 + '_time' + '.mat')
            #     Titlelist.append(saveTitle3 + '_time' + '.mat')
            #     Titlelist.append(saveTitle4 + '_time' + '.mat')
            #     Titlelist.append(saveTitle5 + '_time' + '.mat')
            #     Titlelist.append(saveTitle6 + '_time' + '.mat')
            #     Titlelist.append(saveTitle7 + '_time' + '.mat')
            #     Titlelist.append(saveTitle8 + '_time' + '.mat')
            #
            #     scipy.io.savemat(saveTitle + '_time' + '.mat', mdict={saveTitle + '_time': local_train_time})
            #     scipy.io.savemat(saveTitle2 + '_time' + '.mat', mdict={saveTitle2 + '_time': msg_send_time})
            #     scipy.io.savemat(saveTitle1 + '_time' + '.mat', mdict={saveTitle1 + '_time': local_wait_time})
            #     scipy.io.savemat(saveTitle3 + '_time' + '.mat', mdict={saveTitle3 + '_time': msg_recv_time})
            #     scipy.io.savemat(saveTitle4 + '_time' + '.mat', mdict={saveTitle4 + '_time': msg_recv_process_time})
            #     scipy.io.savemat(saveTitle5 + '_time' + '.mat', mdict={saveTitle5 + '_time': msg_send_process_time})
            #     scipy.io.savemat(saveTitle6 + '_time' + '.mat', mdict={saveTitle6 + '_time': msg_wait_send_time})
            #     scipy.io.savemat(saveTitle7 + '_time' + '.mat', mdict={saveTitle7 + '_time': init_point})
            #     scipy.io.savemat(saveTitle8 + '_time' + '.mat', mdict={saveTitle8 + '_time': start_point})
            #
            #     # Titlelist[0] = Titlelist[0] + '_time.mat'
            #
            #     for i in range(9):
            #         file_send(sock, Titlelist[i])
            #
            #     for i in range(9):
            #         os.remove(os.path.join(Titlelist[i]))
            #
            #     break

            model.train()

            for iteration in range(num_epoch):
                #for batch_idx, (x,y) in enumerate(train_loader):
                x, y = next(iter(train_loader))
                if options['model'] == 'logistic':
                    x = torch.reshape(x, (x.shape[0], 784))

                optimizer.zero_grad()
                pred = model(x)
                loss = criterion(pred,y)
                loss.backward()
                optimizer.step()
            #     print('Global Round:'+ round_i+'\t'+'Local Epoch:')

            end = time.time()
            # acc, loss = local_test(model=model, test_dataloader=test_loader)
            msg = ['MSG_WEIGHT_TIME_SIZE_CLIENT_TO_SERVER', model.state_dict()]
            send_pkl_time = send_msg(sock, msg)
            initd = time.time()

            msg_recv_time.append(recv_time)
            msg_recv_process_time.append(start - process_recv)
            local_train_time.append(end - start)
            local_wait_time.append(start - init - recv_time)
            msg_wait_send_time.append(initd - end - send_pkl_time)
            msg_send_time.append(send_pkl_time)
            finished = time.time()
            msg_send_process_time.append(finished - initd)
            start_point.append(
                [time.localtime(process_recv).tm_min, time.localtime(process_recv).tm_sec, math.modf(process_recv)[0]])
            init_point.append([time.localtime(init).tm_min, time.localtime(init).tm_sec, math.modf(init)[0]])
            # if is_last_round:



except (struct.error, socket.error):
    print('Server has stopped')
    pass

