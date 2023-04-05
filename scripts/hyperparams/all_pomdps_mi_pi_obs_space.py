from pathlib import Path

exp_name = Path(__file__).stem

hparams = {
    'file_name':
        f'runs_{exp_name}.txt',
    'args': [{
        'spec': [
            'tiger-alt-start', 'network'
            'tmaze_5_two_thirds_up', 'example_7', '4x3.95', 'cheese.95', 'network',
            'shuttle.95',
            'paint.95'
            # 'bridge-repair',
            # 'hallway'
        ],
        'policy_optim_alg': 'policy_iter',
        'value_type': 'q',
        'error_type': 'l2',
        'alpha': 1.,
        'objective': 'obs_space',
        'mi_steps': 20000,
        'pi_steps': 10000,
        'init_pi': 0,
        'optimizer': 'adam',
        'lr': 0.01,
        'use_memory': 0,
        'n_mem_states': 2,
        'mi_iterations': 1,
        'seed': [2020 + i for i in range(10)],
    }]
}
