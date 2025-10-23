import torch.optim as optim
import torch.nn as nn
import torch
import os
from src.utils import GetCifar10
from src.model import vgg, cfg_vgg6 
from src.eval import eval

def get_activation(name):
    """Maps an activation function name string to its PyTorch module."""
    if name.lower() == 'relu':
        return nn.ReLU(inplace=True)
    elif name.lower() == 'sigmoid':
        return nn.Sigmoid()
    elif name.lower() == 'tanh':
        return nn.Tanh()
    elif name.lower() == 'silu':
        return nn.SiLU()
    elif name.lower() == 'gelu':
        return nn.GELU()
    else:
        raise ValueError(f"Unsupported activation function: {name}")
    

def get_optimizer(model, name, lr):
    """Maps an optimizer name string to a PyTorch optimizer instance."""
    params = model.parameters()
    name = name.lower()

    if name == 'sgd':
        return optim.SGD(params, lr=lr)
    elif name == 'nesterov-sgd':
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
    

def train_model(trnconfig):
        
        output_directory = 'trainmodel/'
        os.makedirs(output_directory, exist_ok=True)

        trn_activation = trnconfig['activation']
        trn_optimizer = trnconfig['optimizer']
        trn_batchsize = trnconfig['batch_size']
        trn_learnrate = trnconfig['learning_rate']
        trn_epochs = trnconfig['epochs']
        print(trn_activation,trn_optimizer,trn_batchsize,trn_learnrate,trn_epochs)
        
        # Data Loading
        train_loader, test_loader = GetCifar10(trn_batchsize)

        # Device setup
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Model instantiation 
        model = vgg(cfg_vgg6,
                      num_classes=10,
                      batch_norm=True,
                      activation_name=trn_activation).to(device)

        # Criterion and dynamic optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = get_optimizer(model, trn_optimizer, trn_learnrate)

        # Training loop
        model.train()
        best_test_acc = 0.0

        # Variable to store the path to the best model 
        best_model_path = ""

        for epoch in range(trn_epochs):
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

            print(f"Epoch {epoch} - Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}, Test Acc: {test_acc:.2f}")

            if test_acc > best_test_acc:
                best_test_acc = test_acc

                best_model_path = os.path.join(output_directory, "trained_best_model.pth")

                torch.save(model.state_dict(), best_model_path)
                print(f"New best model saved at Epoch {epoch} with Test Acc: {best_test_acc:.4f}")


        print({"best_test_accuracy": best_test_acc})
