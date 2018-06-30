import numpy as np
import math

import utils
import model
import copy

from interpolator import *
from solvers import *
from data_util import load_mnist

batch_size = 100
num_epochs = 10
T = 5
lr = 0.01*2
n_classes = 10
n_bits = 4
fwd_scale_factor = 10e-1
bck_scale_factor = 10e-3

utils.set_seed(42)
x_train, x_test, y_train, y_test = load_mnist(onehot=False)

model = model.LogisticRegression(n_samples=x_train.shape[0], 
                                 batch_size=batch_size, 
                                 n_bits=n_bits,
                                 fwd_scale_factor=fwd_scale_factor,
                                 bck_scale_factor=bck_scale_factor,
                                 in_features=x_train.shape[1], 
                                 out_features=n_classes,
                                 lr=lr)

num_batches = math.ceil(x_train.shape[0]/batch_size)
in_data = utils.OptimizerData( num_epochs, 
                               num_batches, 
                               batch_size, 
                               x_train, 
                               x_test, 
                               y_train, 
                               y_test, 
                               T )


print("SGD BASELINE")
sgd_baseline(in_data, copy.deepcopy(model))
print()

print("SVRG BASELINE")
svrg_baseline(in_data, copy.deepcopy(model))
print()

print("LP SGD BASELINE")
lp_sgd_baseline(in_data, copy.deepcopy(model))
print()

print("SGD BIT CENTERING")
sgd_bitcentering(in_data, copy.deepcopy(model))
print()

print("SVRG BIT CENTERING")
sgd_bitcentering(in_data, copy.deepcopy(model))
