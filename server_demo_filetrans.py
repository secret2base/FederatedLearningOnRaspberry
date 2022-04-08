import argparse
import os
import copy
import time
import pickle
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import importlib
from torch.utils.data import DataLoader, Dataset
import scipy.io
# from config import OPTIMIZERS, DATASETS, MODEL_PARAMS, TRAINERS, BATCH_LIST, SERVER_ADDR,SERVER_PORT
from config import SERVER_ADDR, SERVER_PORT
import importlib
import socket
from utils import recv_msg, send_msg, file_recv
from torchvision import transforms
import math
from PIL import Image

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

def exp_details(options):
    print('\nExperimental details:')
    print(f'    Model     : {options.model}')
    print(f'    Optimizer : {options.optimizer}')
    print(f'    Learning  : {options.lr}')
    print(f'    Global Rounds   : {options.epochs}\n')

    print('    Federated parameters:')
    if options.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {options.frac}')
    print(f'    Local Batch size   : {options.local_bs}')
    print(f'    Local Epochs       : {options.local_ep}\n')
    return

def test_inference(args, model, testloader):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda' if args['gpu'] else 'cpu'
    criterion = torch.nn.CrossEntropyLoss()
    # testloader = DataLoader(test_dataset, batch_size=128,
    #                         shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        if options['model'] == 'logistic':
            images = torch.reshape(images, (images.shape[0], 784))

        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    loss = loss/total
    return accuracy, loss

def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

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


def read_options():
    parser = argparse.ArgumentParser()

    '''parser.add_argument('--algo',
                        help='name of trainer;',
                        type=str,
                        choices=OPTIMIZERS,
                        default='fedavg')'''
    parser.add_argument('--dataset',
                        help='name of dataset;',
                        type=str,
                        default='mnist_all_data_0_equal_niid')
    parser.add_argument('--model',
                        help='name of model;',
                        type=str,
                        default='cnn')
    parser.add_argument('--wd',
                        help='weight decay parameter;',
                        type=float,
                        default=0.00001)
    parser.add_argument('--gpu',
                        action='store_true',
                        default=False,
                        help='use gpu (default: False)')
    parser.add_argument('--noprint',
                        action='store_true',
                        default=False,
                        help='whether to print inner result (default: False)')
    parser.add_argument('--noaverage',
                        action='store_true',
                        default=False,
                        help='whether to only average local solutions (default: True)')
    parser.add_argument('--device',
                        help='selected CUDA device',
                        default=0,
                        type=int)
    parser.add_argument('--num_round',
                        help='number of rounds to simulate;',
                        type=int,
                        default=3)
    parser.add_argument('--eval_every',
                        help='evaluate every ____ rounds;',
                        type=int,
                        default=5)
    parser.add_argument('--clients_per_round',
                        help='number of clients trained per round;',
                        type=int,
                        default=1)
    parser.add_argument('--batch_size',
                        help='batch size when clients train on data;',
                        type=int,
                        default=10)
    parser.add_argument('--num_epoch',
                        help='number of epochs when clients train on data;',
                        type=int,
                        default=5)
    parser.add_argument('--lr',
                        help='learning rate for inner solver;',
                        type=float,
                        default=0.01)
    parser.add_argument('--seed',
                        help='seed for randomness;',
                        type=int,
                        default=0)
    parser.add_argument('--dis',
                        help='add more information;',
                        type=str,
                        default='')
    parsed = parser.parse_args()
    options = parsed.__dict__
    options['gpu'] = options['gpu'] and torch.cuda.is_available()


    return options

def select_clients():
    num_clients = min(options['clients_per_round'], n_nodes)
    #np.random.seed(seed)
    return np.random.choice(range(0,len(client_sock_all)), num_clients, replace=False).tolist()


if __name__== '__main__':
    
    listening_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listening_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listening_sock.bind((SERVER_ADDR, SERVER_PORT))
    client_sock_all = []

   
    options = read_options()

    
    n_nodes = 2
    aggregation_count = 0
    # Establish connections to each client, up to n_nodes clients, setup for clients
    while len(client_sock_all) < n_nodes:
        listening_sock.listen(5)
        print("Waiting for incoming connections...")
        (client_sock, (ip, port)) = listening_sock.accept()
        print('Got connection from ', (ip, port))
        print(client_sock)

        client_sock_all.append([ip, port, client_sock])

    for n in range(0, n_nodes):
        msg = ['MSG_INIT_SERVER_TO_CLIENT', options, n]
        send_msg(client_sock_all[n][2], msg)

    print('All clients connected')

   
    # exp_details(options)
    if options['gpu']:
        torch.cuda.set_device(options['gpu'])
    device = 'cuda' if options['gpu'] else 'cpu'

   
    # test_dataset = []
    # train_data, test_data = read_data('./cifar/cifar0.pkl', './cifar/cifar_test.pkl')
    # train_data, test_data = read_data('./fmnist/fmnist0.pkl', './fmnist/FMNIST_test.pkl')
    #train_data, test_data = read_data('./mnist/iid/mnist0.pkl', './mnist/MNIST_test.pkl')
    test_data = read_data('MNIST_test2.pkl')
    #test_data = read_data('./pcap/traffic_test.pkl')
    # train_data, test_data = read_data('./usps/usps0.pkl', './usps/usps_test.pkl')

    test_loader = DataLoader(dataset=test_data,
                             batch_size=64,
                             shuffle=True)

    if options['model'] == 'cnn':
        global_model = MNIST_CNN()
    else:
        global_model = Logistic()

    
    global_model.to(device)
    global_model.train()

    global_weights = global_model.state_dict()

    
    train_accuracy, train_loss = [], []
    cv_loss, cv_acc = [], []
    print_every = 2

    global_train_time = []
    start1 =time.time()
    for i in range(options['num_round'] + 1 ):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {i+1} |\n')

        global_weights = global_model.state_dict()

        
        selected_clients = select_clients()

        
        is_last_round = False
        print('---------------------------------------------------------------------------')
        aggregation_count += 1
        if aggregation_count == options['num_round'] + 1:
            is_last_round = True
            for n in range(n_nodes):
                msg = ['MSG_WEIGHT_TAU_SERVER_TO_CLIENT', is_last_round, global_weights, aggregation_count]
                send_msg(client_sock_all[n][2], msg)
            for n in range(n_nodes):
                for i in range(9):
                    file_recv(client_sock_all[n][2])
            break

        start = time.time()
        for n in selected_clients:
            msg = ['MSG_WEIGHT_TAU_SERVER_TO_CLIENT', is_last_round, global_weights, aggregation_count]
            send_msg(client_sock_all[n][2], msg)

        print('Waiting for local iteration at client')

        for n in selected_clients:
            msg, temp = recv_msg(client_sock_all[n][2], 'MSG_WEIGHT_TIME_SIZE_CLIENT_TO_SERVER')
            
            w = msg[1]
            local_weights.append(copy.deepcopy(w))
            # local_losses.append(copy.deepcopy(loss))


        global_weights = average_weights(local_weights)

       
        global_model.load_state_dict(global_weights)

        # loss_avg = sum(local_losses) / len(local_losses)
        # train_loss.append(loss_avg)

        end = time.time()
        test_acc, test_loss = test_inference(options, global_model, test_loader)
        print(test_acc,test_loss)
        cv_acc.append(test_acc)
        cv_loss.append(test_loss)
        global_train_time.append(end-start)
        end1 = time.time()
        #average_acc=sum(cv_acc)/len(cv_acc)
        #print(average_acc)
        # if is_last_round == True:


        if  test_acc>=1.0:
            print(end1-start1)
            break

    saveTitle ='K' + str(options['clients_per_round']) +  'T' + str(options['num_round']) + 'E' + str(options['num_epoch']) + 'B' + str(options['batch_size'])
    scipy.io.savemat(saveTitle + '_time' + '.mat', mdict={saveTitle + '_time': global_train_time})
    scipy.io.savemat(saveTitle + '_acc' + '.mat', mdict={saveTitle + '_acc': cv_acc})
    scipy.io.savemat(saveTitle + '_loss' + '.mat', mdict={saveTitle + '_loss': cv_loss})
    # Save tracked information



