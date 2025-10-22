import wandb
from src.train import run_experiment


if __name__ == "__main__":
    sweep_config = {
        'method': 'grid',  # Use 'grid' for testing all combinations of discrete variables. Use 'random' for a larger search space.
        'metric': {
            'name': 'best_test_accuracy',
            'goal': 'maximize'
        },
        'parameters': {
            # (a) Vary the activation function
            'activation': {
                'values': ['silu']
            },
            # (b) Vary the optimizer
            'optimizer': {
                'values': ['adam']
            },
            # (c) Vary the batch size, epochs, and learning rate
            'batch_size': {
                'values': [128]
            },
            'learning_rate': {
                'values': [0.001]
            },
            'epochs': {
                'values': [100] # Using a small number for the template, use 50-100 for a real experiment
            }
        }
    }

    wandb.login()
    sweep_id = wandb.sweep(sweep_config, project="Enter_project_name")
    wandb.agent(sweep_id, run_experiment, count=1)