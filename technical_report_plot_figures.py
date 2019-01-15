import pickle
from collections import defaultdict
from utils.plot_utils import *
from utils.file_utils import get_immediate_files

save_dir = ''

metric_keys = ['elbo', 'test_ll', 'test_auxiliary', 'noise_sigma', 'train_kl', 'train_ll']

bostonHousing_results_dir = './remote_logs/bostonHousing/sweep-2-2019-01-12 19:38:07'
concrete_results_dir = './remote_logs/concrete/sweep-2-2019-01-12 19:38:07'
kin8nm_results_dir = './remote_logs/kin8nm/sweep-2-2019-01-12 19:38:07'
# naval_results_dir = './remote_logs/naval-propulsion-plant/sweep-1-2019-01-11 23:57:27'
power_results_dir = './remote_logs/power-plant/sweep-2-2019-01-12 19:38:07'
protein_dir = './remote_logs/protein-tertiary-structure/sweep-2-2019-01-12 19:38:07'
wine_results_dir = './remote_logs/wine-quality-red/sweep-2-2019-01-12 19:38:07'
yacht_results_dir = './remote_logs/yacht/sweep-2-2019-01-12 19:38:07'

data = {
    'boston':   {'dir':bostonHousing_results_dir, 'dim':13, 'data_size':430},
    'concrete': {'dir':concrete_results_dir, 'dim':8, 'data_size':875},
    'kin8nm':   {'dir':kin8nm_results_dir, 'dim':8, 'data_size':652},
    # 'naval':    {'dir':naval_results_dir, 'dim':8, 'data_size':6963},
    'power':    {'dir':power_results_dir, 'dim':16, 'data_size':10143},
    'protein':  {'dir':protein_dir, 'dim':9, 'data_size':38870},
    'wine':     {'dir':wine_results_dir, 'dim':11, 'data_size':1359},
    'yacht':    {'dir':yacht_results_dir, 'dim':6, 'data_size':261}
}

for key in data.keys():
    data_set = data[key]
    dir = data_set['dir']

    files = get_immediate_files(dir)
    files = [f for f in files if f.split('.')[-1] == 'pkl']

    results = []

    for file in files:
        r = pickle.load(open(f'{dir}/{file}', 'rb'))
        results.append(r)

    data_set['results'] = results

    data[key] = data_set


def plot_results(data_set):
    results = data_set['results']
    dim = data_set['dim']

    for key in metric_keys:
        plot_training_curves(results, val=key)

    num_weights = defaultdict(list)
    layer_size = defaultdict(list)
    final_ll = defaultdict(list)
    final_rmse = defaultdict(list)
    final_cost = defaultdict(list)

    for result in results:
        h = result['hidden_size']
        h = [dim] + h + [1]

        prior_var = result['prior_var']

        weights = 0.
        for idx in range(len(h)-1):
            weights += h[idx]*h[idx+1]

        num_weights[prior_var].append(weights)
        layer_size[prior_var].append(h[1])
        final_ll[prior_var].append(result['results']['test_ll'][-1])
        final_rmse[prior_var].append(result['results']['test_auxiliary'][-1])
        final_cost[prior_var].append(result['results']['elbo'][-1])

    plot_dict(num_weights, final_ll, 'num weights', 'final ll', log_scale=True)
    plot_dict(num_weights, final_rmse, 'num weights', 'final rmse', log_scale=True)
    # plot_dict(num_weights, final_cost, 'num weights', 'final cost', log_scale=True)

    rank_final_value(results, value='test_ll', minimum=False)
    rank_final_value(results, value='test_auxiliary', minimum=True)

    plt.show()


# plot_results(data['boston'])
# plot_results(data['concrete'])
# plot_results(data['kin8nm'])
# plot_results(data['naval'])
# plot_results(data['power'])
# plot_results(data['protein'])
# plot_results(data['wine'])
# plot_results(data['yacht'])
