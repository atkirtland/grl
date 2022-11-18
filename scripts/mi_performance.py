import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from jax.nn import softmax
from pathlib import Path
from collections import namedtuple
np.set_printoptions(precision=4)
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams.update({'font.size': 18})

from grl.utils import load_info
from definitions import ROOT_DIR


# %%
results_dir = Path(ROOT_DIR, 'results', 'pomdps_mi_pi')
vi_results_dir = Path(ROOT_DIR, 'results', 'pomdps_vi')

split_by = ['spec', 'algo']
Args = namedtuple('args', split_by)

# %%

all_results = {}

for results_path in results_dir.iterdir():
    if results_path.suffix != '.npy':
        continue
    info = load_info(results_path)

    args = info['args']
    agent = info['agent']
    init_policy_info = info['logs']['initial_policy_stats']
    init_improvement_info = info['logs']['initial_improvement_stats']
    final_mem_info = info['logs']['final_mem_stats']

    def get_perf(info: dict):
        return (info['state_vals_v'] * info['p0']).sum()

    single_res = {
        'init_policy_perf': get_perf(init_policy_info),
        'init_improvement_perf': get_perf(init_improvement_info),
        'final_mem_perf': get_perf(final_mem_info),
        'init_policy': info['logs']['initial_policy'],
        'init_improvement_policy': info['logs']['initial_improvement_policy'],
        'final_mem': np.array(agent.memory),
        'final_policy': np.array(agent.policy)
    }

    hparams = Args(*tuple(args[s] for s in split_by))

    if hparams not in all_results:
        all_results[hparams] = {}

    for k, v in single_res.items():
        if k not in all_results[hparams]:
            all_results[hparams][k] = []
        all_results[hparams][k].append(v)
    all_results[hparams]['args'] = args

for hparams, res_dict in all_results.items():
    for k, v in res_dict.items():
        if k != 'args':
            all_results[hparams][k] = np.stack(v)

# %%
# Get vi performance
for hparams, res_dict in all_results.items():
    for vi_path in vi_results_dir.iterdir():
        if hparams.spec in vi_path.name:
            vi_info = load_info(vi_path)
            all_results[hparams]['vi_perf'] = np.array([(vi_info['optimal_vs'] * vi_info['p0']).sum()])

# %%
# all_normalized_perf_results = {}
for hparams, res in all_results.items():
    max_v, min_v = -float('inf'), float('inf')
    for k, v in res.items():
        if '_perf' in k:
            max_v = max(v.max(), max_v)
            min_v = min(v.min(), min_v)

    for k, v in res.items():
        if '_perf' in k:
            all_results[hparams][k] = (v - min_v) / (max_v - min_v)



# %%
spec_plot_order = ['example_7', 'slippery_tmaze_5_two_thirds_up',
                   'tiger', 'paint.95', 'cheese.95',
                   'network', 'shuttle.95', '4x3.95']

all_table_results = {}
all_plot_results = {'x': [], 'xlabels': []}

for i, spec in enumerate(spec_plot_order):
    hparam = next(k for k in all_results.keys() if k.spec == spec)
    res = all_results[hparam]
    all_table_results[hparam] = {}
    all_plot_results['x'].append(i)
    all_plot_results['xlabels'].append(hparam.spec)

    for k, v in res.items():
        if 'perf' in k:
            mean = v.mean(axis=0)
            std_err = v.std(axis=0) / np.sqrt(v.shape[0])
            all_table_results[hparam][f'{k}_mean'] = mean
            all_table_results[hparam][f'{k}_std_err'] = std_err

            stripped_str = k.replace('_perf', '')
            if stripped_str not in all_plot_results:
                all_plot_results[stripped_str] = {'mean': [], 'std_err': []}
            all_plot_results[stripped_str]['mean'].append(mean)
            all_plot_results[stripped_str]['std_err'].append(std_err)

        else:
            all_table_results[k] = v

ordered_plot = []
ordered_plot.append(('init_policy', all_plot_results['init_policy']))
ordered_plot.append(('init_improvement', all_plot_results['init_improvement']))
ordered_plot.append(('final_mem', all_plot_results['final_mem']))
# ordered_plot.append(('state_optimal', all_plot_results['vi']))



# %%
ordered_plot

# %%
def maybe_spec_map(id: str):
    spec_map = {
        '4x3.95': '4x3',
        'cheese.95': 'cheese',
        'paint.95': 'paint',
        'shuttle.95': 'shuttle',
        'example_7': 'ex. 7',
        'slippery_tmaze_5_two_thirds_up': 'tmaze',
    }
    if id not in spec_map:
        return id
    return spec_map[id]

group_width = 1
bar_width = group_width / (len(ordered_plot) + 2)
fig, ax = plt.subplots(figsize=(12, 6))

x = np.array(all_plot_results['x'])
xlabels = [maybe_spec_map(l) for l in all_plot_results['xlabels']]

for i, (label, plot_dict) in enumerate(ordered_plot):
    ax.bar(x + (i + 1) * bar_width, plot_dict['mean'], bar_width,
           yerr=plot_dict['std_err'], label=label)
ax.set_ylabel('Performance\n (w.r.t. optimal MDP policy)')
ax.set_xticks(x + group_width / 2)
ax.set_xticklabels(xlabels)
ax.legend(bbox_to_anchor=(0.7, 0.7), framealpha=0.95)
ax.set_title("Performance of ε-greedy (0.1) policies in POMDPs")
