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

'''
Experiment to run a series of different parameter settings on the specified data set using the configuration specified. 
'''

parser = argparse.ArgumentParser(description='Script for dispatching train runs of BNNs over larger search spaces')
parser.add_argument('-c',  '--config', default='./config/kin8nm.yaml')
parser.add_argument('-ds', '--dataset', default='kin8nm')
parser.add_argument('-ld', '--logdir', default='./results')
parser.add_argument('-dd', '--datadir', default='./data_dir')
parser.add_argument('-cm', '--commonname', default=None)
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
hidden_sizes = experiment_config['hidden_sizes']
learning_rates = experiment_config['learning_rates']
prior_vars = experiment_config['prior_vars']

hidden_layers = experiment_config['hidden_layers']
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
param_space = list(itertools.product(hidden_layers, hidden_sizes, learning_rates, prior_vars))

# Loop over parameter space
for idx, (hidden_layer, hidden_size, lr, prior_var) in enumerate(param_space):

    hidden_size_ = [hidden_size] * hidden_layer

    model = MLP_MFVI(input_size, hidden_size_, output_size, p_mean=0, p_var = prior_var)
    loss = mm.LogHomoskedasticGaussianLoss(log_var_init=-6, reduction='none')
    auxiliary = torch.nn.MSELoss(reduction='none')
    if use_cuda:
        model.cuda()
        loss.cuda()
    optimiser = torch.optim.Adam([{'params':model.parameters()}, {'params': loss.parameters()}], lr=lr)
    inference = mm.MeanFieldVariationalInference(model, loss, auxiliary, optimiser, train_loader, test_loader)

    name = f'{inference}'

    print(f'{args.dataset} - running model {inference}, parameter set {idx+1} of {len(param_space)}')
    log_dir = f'{results_dir}/logs/{name}'
    result = evaluate_bayes_regression(inference, epochs, log_freq=100, log_dir=log_dir, verbose=True)
    experiment_config = inference.get_config()
    train_config = {'batch_size': batch_size, 'epochs': epochs, 'results': result}
    result = {**experiment_config, **train_config, 'results': result}

    result_file = f'{results_dir}/{name}.pkl'
    with open(result_file, 'wb') as h:
        pickle.dump(result, h)


    # model = MLP(input_size, hidden_size_, output_size)
    # loss = auxiliary
    # if use_cuda:
    #     model.cuda()
    # optimiser = torch.optim.Adam([{'params':model.parameters()}], lr=lr)
    # model_trainer = MLPTrainer(model, loss, optimiser, train_loader, test_loader)
    #
    # print(f'{args.dataset} - running model {model_trainer}, parameter set {idx+1} of {len(param_space)}')
    # log_dir = f'{results_dir}/logs/{model_trainer}'
    # result = evaluate_point_regression(model_trainer, epochs, log_dir=log_dir, verbose=True)
    # experiment_config = inference.get_config()
    # train_config = {'batch_size': batch_size, 'epochs': epochs, 'results': result}
    # result = {**experiment_config, **train_config, 'results': result}
    #
    # result_file = f'{results_dir}/{inference}.pkl'
    # with open(result_file, 'wb') as hidden_size_:
    #     pickle.dump(result, hidden_size_)
