import os
import datetime
import pickle
import yaml
import shutil
import argparse
import itertools
import torch

import MFVI as mm
from model.MLP_MFVI import MLP_MFVI
from data.data_sets import RegressionDataloader
from experiment.model_evaluation import evaluate_regression


parser = argparse.ArgumentParser(description='Script for dispatching train runs of BNNs over larger search spaces')
parser.add_argument('-c',  '--config', default='./config/kin8nm.yaml')
parser.add_argument('-ds', '--dataset', default='kin8nm')
parser.add_argument('-ld', '--logdir', default='./results')
parser.add_argument('-dd', '--datadir', default='./data_dir')
parser.add_argument('-cm', '--commonname', default=None)
args = parser.parse_args()

experiment_config = yaml.load(open(args.config, 'rb'))

# Script parameters
log_dir = args.logdir
common_name = args.commonname
# data_set = experiment_config['data_set']
data_set = args.dataset
use_cuda = torch.cuda.is_available()

# Set up logging directory and grab the config file
date_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

if common_name is not None:
    results_dir = f'{log_dir}/{data_set}/{common_name}-{date_time}'
else:
    results_dir = f'{log_dir}/{data_set}/{date_time}'

latest_dir = f'{log_dir}/{data_set}/latest'

#####

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

if os.path.islink(latest_dir):
    os.unlink(latest_dir)

os.symlink(os.path.abspath(results_dir), latest_dir)

# Copy config across for reference
shutil.copy2(args.config, results_dir)

#####

# Parse configuration
hidden_sizes = experiment_config['hidden_sizes']
learning_rates = experiment_config['learning_rates']
prior_vars = experiment_config['prior_vars']

hidden_layers = experiment_config['hidden_layers']
batch_size = experiment_config['batch_size']
epochs = experiment_config['epochs']

print(f'Running experiment on {data_set} with parameters:\n'
      f'{experiment_config}\n'
      f'Saving results in {results_dir}\n')


# Load in dataset and related info

train_loader = RegressionDataloader(data_set, batch_size, data_dir=os.path.abspath('./data'), train=True, shuffle=True)
test_loader  = RegressionDataloader(data_set, batch_size, data_dir=os.path.abspath('./data'), train=False, shuffle=False)
input_size, train_length, output_size = train_loader.get_dims()

# Design search space for paramters
param_space = list(itertools.product(hidden_sizes, learning_rates, prior_vars))

# Loop over parameter space
for idx, (hidden_size, lr, prior_var) in enumerate(param_space):

    h = [hidden_size] * hidden_layers

    model = MLP_MFVI(input_size, h, output_size, p_mean=0, p_var = prior_var)
    loss = mm.LogHomoskedasticGaussianLoss(log_var_init=-6)
    if use_cuda:
        model.cuda()
        loss.cuda()

    optimiser = torch.optim.Adam([{'params':model.parameters()}, {'params': loss.parameters()}], lr=lr)

    inference = mm.MeanFieldVariationalInference(model, loss, optimiser, train_loader, test_loader)

    print(f'running model {inference}, parameter set {idx+1} of {len(param_space)}')

    log_dir = f'{results_dir}/logs/{inference}'

    # Run a standard test on the model, logging training info etc
    result = evaluate_regression(inference, epochs, log_freq=100, log_dir=log_dir)

    # Dump results to a sensibly named file
    experiment_config = inference.get_config()
    train_config = {'batch_size': batch_size, 'epochs': epochs, 'results': result}
    result = {**experiment_config, **train_config, 'results': result}
    result_file = f'{results_dir}/{inference}.pkl'

    with open(result_file, 'wb') as h:
        pickle.dump(result, h)
