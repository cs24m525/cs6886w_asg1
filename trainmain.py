from src.train import train_model
from src.model import cfg_vgg6 
from src.utils import GetCifar10,set_seed
from src.eval import eval, load_and_test_model
import os

if __name__ == "__main__":

    train_config = {
        'activation': 'silu',
        'optimizer': 'adam',
        'batch_size': 128,
        'learning_rate': 0.001,
        'epochs': 100
    }

    # model training and saving the best model in directory trainmodel
    train_model(train_config)

    # saved model is evaluated by loading the saved model
    model_path = os.path.join('trainmodel', 'trained_best_model.pth')
    print(model_path)
    set_seed(42)
    arc_config = {
        'cfg': cfg_vgg6,
        'num_classes': 10,
        'batch_norm': True,
        'activation_name': 'SiLU' 
    }

    batch_size = 128
    _, test_loader = GetCifar10(batch_size) 

    final_accuracy = load_and_test_model(
        model_path, 
        arc_config, 
        test_loader, 
        eval
    )    