from src.train import train_model


if __name__ == "__main__":

    train_config = {
        'activation': 'silu',
        'optimizer': 'adam',
        'batch_size': 128,
        'learning_rate': 0.001,
        'epochs': 1
    }

    train_model(train_config)