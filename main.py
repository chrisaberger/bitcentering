import utils
import model
import copy
import argparse

from solvers import *
from data_util import load_mnist

parser = argparse.ArgumentParser()
parser.add_argument("-T", action="store", default=2, type=int,
                    help="T parameter for SVRG type algorithms.")
parser.add_argument("-e", "--n-epochs", action="store", default=10, type=int,
                    help="Number of epochs to run for")
parser.add_argument("-bs", "--batch-size", action="store", default=100, type=int,
                    help="Batch size.")
parser.add_argument("-a", "--alpha", action="store", default=0.01, type=float,
                    help="Learning Rate")
parser.add_argument("-b", "--n-bits", action="store", default=8, type=int,
                    help="Number of bits of precision")
parser.add_argument("--lin-fwd-sf", action="store", default=1, type=float,
                    help="Linear layer forward scale factor.")
parser.add_argument("--lin-bck-sf", action="store", default=1e-2, type=float,
                    help="Linear layer backwards scale factor.")
parser.add_argument("--loss-sf", action="store", default=1e-3, type=float,
                    help="Loss scale factor.")
parser.add_argument("-s", "--seed", action="store", default=42, type=int,
                    help="Random seed.")
parser.add_argument("-c", "--n-classes", action="store", default=10, type=int,
                    help="Number of classes for classification.")
parser.add_argument("--solver", action="store", default="all", type=str,
                    choices=["sgd", "svrg", 
                             "lp-sgd", "lp-svrg", 
                             "bc-sgd", "bc-svrg", 
                             "all"],
                    help="Solver/optimization algorithm.")
args = parser.parse_args()
print(args)

utils.set_seed(args.seed)
x_train, x_test, y_train, y_test = load_mnist(onehot=False)

model = model.LogisticRegression(n_samples=x_train.shape[0], 
                                 batch_size=args.batch_size, 
                                 n_bits=args.n_bits,
                                 fwd_scale_factor=args.lin_fwd_sf,
                                 bck_scale_factor=args.lin_bck_sf,
                                 loss_scale_factor=args.loss_sf,
                                 in_features=x_train.shape[1], 
                                 out_features=args.n_classes,
                                 lr=args.alpha)

in_data = utils.OptimizerData( args, 
                               x_train, 
                               x_test, 
                               y_train, 
                               y_test )

if args.solver == "sgd" or args.solver == "all":
    print("\nRunning SGD...")
    sgd_baseline(in_data, copy.deepcopy(model))
if args.solver == "svrg" or args.solver == "all":
    print("\nRunning SVRG...")
    svrg_baseline(in_data, copy.deepcopy(model))
if args.solver == "bc-sgd" or args.solver == "all":
    print("\nRunning Bit Centering SGD")
    sgd_bitcentering(in_data, copy.deepcopy(model))
if args.solver == "bc-svrg" or args.solver == "all":
    print("\nRunning Bit Centering SVRG")
    svrg_bitcentering(in_data, copy.deepcopy(model))
if args.solver == "lp-sgd" or args.solver == "all":
    print("\nRunning Low Precision SGD")
    lp_sgd_baseline(in_data, copy.deepcopy(model))
if args.solver == "lp-svrg" or args.solver == "all":
    print("\nRunning Low Precision SVRG")
    lp_svrg_baseline(in_data, copy.deepcopy(model))