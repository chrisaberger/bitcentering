import torch
import numpy as np
import math 

class OptimizerData:
    def __init__(self,
                 args,
                 x_train, 
                 x_test, 
                 y_train, 
                 y_test):
        self.num_epochs = args.n_epochs
        self.batch_size = args.batch_size
        self.num_batches = math.ceil(x_train.shape[0]/args.batch_size)
        self.T = args.T
        self.n_classes = args.n_classes
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def get_data(self, batch_index):
        start, end = batch_index * self.batch_size, \
                     (batch_index + 1) * self.batch_size
        x = self.x_train[start:end]
        y = self.y_train[start:end]
        return x, y

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

def print_info(epoch, cost, acc):
    print("Epoch: " + str(epoch+1) 
        + ", cost: " + str(round(cost, 6)) 
        + ", acc: " + str(round(acc, 2)) + "%") 