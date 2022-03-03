# GLOBAL PARAMETERS
DATASETS = ['sent140', 'nist', 'shakespeare',
            'mnist', 'synthetic', 'cifar10','mqtt']
TRAINERS = {'fedavg': 'FedAvgTrainer',
            'fedavg4': 'FedAvg4Trainer',
            'fedavg5': 'FedAvg5Trainer',
            'fedavg9': 'FedAvg9Trainer',}
OPTIMIZERS = TRAINERS.keys()
BATCH_LIST = [32, 64, 128, 256, 32, 64, 128, 256,32, 64, 128, 256,32, 64, 128, 256,32, 64, 128, 256]
# SERVER_ADDR= 'localhost'   # When running in a real distributed setting, change to the server's IP address
SERVER_ADDR= '127.0.0.1'   # When running in a real distributed setting, change to the server's IP address
SERVER_PORT = 51000

