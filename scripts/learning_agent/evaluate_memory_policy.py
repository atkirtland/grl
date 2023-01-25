from functools import partial
import glob
import os

from jax import jit
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from grl import environment
from grl.agents.actorcritic import ActorCritic
from grl.mdp import AbstractMDP, MDP
from grl.memory import memory_cross_product
from grl.environment.memory_lib import get_memory
from grl.utils.math import greedify
from grl.utils.mdp import amdp_get_occupancy
from grl.utils.policy_eval import functional_solve_mdp
from scripts.learning_agent.optuna_to_pandas import load_study

#%%
experiment_name = 'exp15-tmaze5-mi'
env_name = 'tmaze_5_two_thirds_up'
seed = '3'

@partial(jit, static_argnames=['gamma'])
def get_perf(pi_obs: jnp.ndarray,
                   T: jnp.ndarray,
                   R: jnp.ndarray,
                   p0: jnp.ndarray,
                   phi: jnp.ndarray, gamma: float):
    pi_state = phi @ pi_obs
    state_v, state_q = functional_solve_mdp(pi_state, T, R, gamma)
    return jnp.dot(p0, state_v)

results = {}
for results_dir in tqdm(glob.glob(f'results/sample_based/{experiment_name}/{env_name}/*/')):
    seed = int(os.path.basename(results_dir.rstrip("/")))
    # results_dir = f'results/sample_based/{experiment_name}/{env_name}/{seed}/'
    try:
        memory = np.load(results_dir + 'memory.npy')
        policy = np.load(results_dir + 'policy.npy')
        td = np.load(results_dir + 'q_td.npy')
        mc = np.load(results_dir + 'q_mc.npy')
        mc
    except:
        print(f'File not found for seed {seed}')
        continue

    study = load_study(experiment_name, env_name, seed)
    params = [
        study.best_trial.params[key] for key in sorted(study.best_trial.params.keys(), key=int)
    ]
    spec = environment.load_spec(env_name, memory_id=None)
    mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
    env = AbstractMDP(mdp, spec['phi'])
    mem_logits = jnp.log(memory+1e-20)
    amdp_mem = memory_cross_product(env, mem_logits)

    def expected_lambda_discrep(amdp_mem, mem_logits, policy, td, mc):
        c_s = amdp_get_occupancy(greedify(policy), amdp_mem)
        c_o = (c_s @ amdp_mem.phi)
        p_o = c_o / c_o.sum()
        p_oa = (policy * p_o[:,None]).T
        return (abs(td - mc) * p_oa).sum()
    expected_lambda_discrep(amdp_mem, mem_logits, policy, td, mc)
    study.best_value

    performance = get_perf(greedify(policy), amdp_mem.T, amdp_mem.R, amdp_mem.p0, amdp_mem.phi, amdp_mem.gamma)
    results[seed] = performance

for seed, performance in sorted(results.items(), key=lambda x: x[-1]):
    print(f'seed {seed}:', performance)
