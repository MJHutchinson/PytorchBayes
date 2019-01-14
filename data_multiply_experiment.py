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
from model.MLP import MLP, MLPTrainer
from data.data_sets import RegressionDataloader
from experiment.model_evaluation import evaluate_bayes_regression, evaluate_point_regression

''' Expereiment to investigate how the size of data affects the KL/LL balance in traingin VB BNNs'''

parser = argparse.ArgumentParser(description='Script for dispatching train runs of BNNs over larger search spaces')
parser.add_argument('-c',  '--config', default='./config/kin8nm.yaml')
parser.add_argument('-ds', '--dataset', default='kin8nm')
parser.add_argument('-ld', '--logdir', default='./results')
parser.add_argument('-dd', '--datadir', default='./data_dir')
parser.add_argument('-cm', '--commonname', default=None)
parser.add_argument('-nd', '--nodatetime', action='store_true')
args = parser.parse_args()

experiment_config = yaml.load(open(args.config, 'rb'))

use_cuda = torch.cuda.is_available()

# Set up logging directory and grab the config file
date_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

if args.commonname is not None :
    if args.nodatetime:
        folder_name = args.commonname
    else:
        folder_name = f'{args.commonname}-{date_time}'
else:
    if not args.nodatetime:
        folder_name = f'{date_time}'
    else:
        raise ValueError('Must supply a common name, or set ude datetime to True')

results_dir = f'{args.logdir}/{args.dataset}/{folder_name}'

latest_dir = f'{args.logdir}/{args.dataset}/latest'

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
hidden_size = experiment_config['hidden_size']
learning_rate = experiment_config['learning_rate']
prior_var = experiment_config['prior_var']

hidden_layer = experiment_config['hidden_layers']
batch_size = experiment_config['batch_size']
epochs = experiment_config['epochs']

print(f'Running experiment on {args.dataset} with parameters:\n'
      f'{experiment_config}\n'
      f'Saving results in {results_dir}\n')


# Load in dataset and related info

train_loader = RegressionDataloader(args.dataset, batch_size, data_dir=args.datadir, train=True, shuffle=True)
test_loader  = RegressionDataloader(args.dataset, batch_size, data_dir=args.datadir, train=False, shuffle=False)
input_size, train_length, output_size = train_loader.get_dims()

# Design search space for paramters
data_multiply = experiment_config['data_multiply']

# Loop over parameter space
for idx, dm in enumerate(data_multiply):

    train_loader.data_size = float(dm) * train_loader.real_data_size # re weight the data

    hidden_size_ = [hidden_size] * hidden_layer

    model = MLP_MFVI(input_size, hidden_size_, output_size, p_mean=0, p_var = prior_var)
    loss = mm.LogHomoskedasticGaussianLoss(log_var_init=-6, reduction='none')
    auxiliary = torch.nn.MSELoss(reduction='none')
    if use_cuda:
        model.cuda()
        loss.cuda()
    optimiser = torch.optim.Adam([{'params':model.parameters()}, {'params': loss.parameters()}], lr=learning_rate)
    inference = mm.MeanFieldVariationalInference(model, loss, auxiliary, optimiser, train_loader, test_loader)

    name = f'data_multiply_{dm}_{inference}'

    print(f'{args.dataset} - running model {inference}, parameter set {idx+1} of {len(data_multiply)}')
    log_dir = f'{results_dir}/logs/{name}'
    result = evaluate_bayes_regression(inference, 100, log_freq=100, log_dir=log_dir, verbose=True)
    experiment_config = inference.get_config()
    train_config = {'data_multiply': dm, 'batch_size': batch_size, 'epochs': epochs, 'results': result}
    result = {**experiment_config, **train_config, 'results': result}

    result_file = f'{results_dir}/{name}.pkl'
    with open(result_file, 'wb') as h:
        pickle.dump(result, h)
