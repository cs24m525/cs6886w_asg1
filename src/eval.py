from src.model import vgg, cfg_vgg6 
from src.utils import GetCifar10

import torch

def eval(model,data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in data:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    acc = 100. * correct / total
    return acc

def load_and_test_model(model_path, config_for_architecture, test_loader, eval_fn):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = vgg(config_for_architecture['cfg'],
                num_classes=config_for_architecture['num_classes'],
                batch_norm=config_for_architecture['batch_norm'],
                activation_name=config_for_architecture['activation_name']).to(device)

    print(f"Loading weights from: {model_path}")

    state_dict = torch.load(model_path, map_location=device)

    model.load_state_dict(state_dict)

    model.eval()
    print("Model loaded and set to evaluation mode.")

    with torch.no_grad():
        test_accuracy = eval_fn(model, test_loader)

    print(f"\nFinal Test Accuracy on loaded model: {test_accuracy:.4f}")
    return test_accuracy    
