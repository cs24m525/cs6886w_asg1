import torch.optim as optim
import torch.nn as nn
import wandb
import torch
import os
from src.utils import GetCifar10
from src.model import vgg, cfg_vgg6 

def get_activation(name):
    """Maps an activation function name string to its PyTorch module."""
    if name.lower() == 'relu':
        return nn.ReLU(inplace=True)
    elif name.lower() == 'sigmoid':
        return nn.Sigmoid()
    elif name.lower() == 'tanh':
        return nn.Tanh()
    elif name.lower() == 'silu':
        # SiLU (Sigmoid Linear Unit) is also known as Swish
        return nn.SiLU()
    elif name.lower() == 'gelu':
        return nn.GELU()
    # Add other activations as needed
    else:
        raise ValueError(f"Unsupported activation function: {name}")
    

def get_optimizer(model, name, lr):
    """Maps an optimizer name string to a PyTorch optimizer instance."""
    params = model.parameters()
    name = name.lower()

    if name == 'sgd':
        # Simple SGD
        return optim.SGD(params, lr=lr)
    elif name == 'nesterov-sgd':
        # SGD with Momentum and Nesterov enabled
        return optim.SGD(params, lr=lr, momentum=0.9, nesterov=True)
    elif name == 'adam':
        return optim.Adam(params, lr=lr)
    elif name == 'adagrad':
        return optim.Adagrad(params, lr=lr)
    elif name == 'rmsprop':
        return optim.RMSprop(params, lr=lr)
    elif name == 'nadam':
        return optim.Adam(params, lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer: {name}")    
    

def run_experiment(config=None):
    # Initialize W&B run
    with wandb.init(config=config) as run:
        # Fetch hyperparameters from W&B config
        config = run.config

        # Data Loading
        # The batch size is now a hyperparameter
        train_loader, test_loader = GetCifar10(config.batch_size)

        # Device setup
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Model instantiation - dynamic activation
        model = vgg(cfg_vgg6,
                      num_classes=10,
                      batch_norm=True,
                      activation_name=config.activation).to(device)

        # Log model architecture (optional but recommended)
        wandb.watch(model, log='all')

        # Criterion and dynamic optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = get_optimizer(model, config.optimizer, config.learning_rate)

        # Training loop
        model.train()
        best_test_acc = 0.0

        # Variable to store the path to the best model checkpoint
        trained_model_path = ""

        for epoch in range(config.epochs):
            running_loss = 0.0

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            train_acc = eval(model, train_loader)
            test_acc = eval(model, test_loader)

            # Log metrics to W&B
            wandb.log({
                "epoch": epoch,
                "train_loss": running_loss / len(train_loader),
                "train_accuracy": train_acc,
                "test_accuracy": test_acc
            })

            print(f"Epoch {epoch} - Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}, Test Acc: {test_acc:.2f}")

            # Save the model if it's the best one so far
            if test_acc > best_test_acc:
                best_test_acc = test_acc

                # Define the local path for the model file within the W&B run directory
                best_model_path = os.path.join(wandb.run.dir, "trained_model.pth")

                # Save the model state dict (PyTorch standard)
                torch.save(model.state_dict(), trained_model_path)
                print(f"New best model saved at Epoch {epoch} with Test Acc: {best_test_acc:.4f}")


        # Log the final best accuracy
        wandb.log({"best_test_accuracy": best_test_acc})

        if os.path.exists(best_model_path):
            artifact = wandb.Artifact(
                name="cifar10-vgg-model",
                type="model",
                description="Best trained VGG model on CIFAR-10"
            )
            # Add the saved file to the artifact
            artifact.add_file(best_model_path)

            # Log the artifact to W&B
            run.log_artifact(artifact)
            print("Model artifact successfully logged to W&B.")