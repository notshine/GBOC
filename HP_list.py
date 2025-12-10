Multi_algo_HP_dict = {
    'GBOC': {
        'win_size': [2, 5, 10, 50],
        'lr': [1e-3, 1e-4],
        'hidden_dim': [32, 64, 128],
        'num_layers': [1,2,3,4],
        'alpha': [0.1, 0.5, 0.9]
    }
}


Optimal_Multi_algo_HP_dict = {
    'GBOC': {'win_size': 10, 'lr': 0.0001, 'hidden_dim': 32, 'num_layers': 3, 'alpha': 0.9, 'epochs': 30},
}


Uni_algo_HP_dict = {
    'GBOC': {
        'win_size': [2, 5, 10, 50],
        'lr': [1e-3, 1e-4],
        'hidden_dim': [32, 64, 128],
        'num_layers': [1,2,3,4],
        'alpha': [0.1, 0.5, 0.9]
    }
}

Optimal_Uni_algo_HP_dict = {
    'GBOC': {'win_size':5, 'lr': 0.0001, 'hidden_dim': 32, 'num_layers': 3, 'alpha': 0.9, 'epochs': 30},
}