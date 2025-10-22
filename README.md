CS6886W - System Engineering for Deep Learning - Assignment 1

Trained model is loaded under model folder as "best_model.pth"
    Model Test Accuracy is : 87.1000
    configuration : activation function : SiLU
                    optimizer           : Adam
                    batch_size          : 128
                    learning_rate       : 0.001
                    epochs              : 100

Execute main.py to run the evaluation of model on test data.

Execute trainmain.py to run the model training simulation. 
    Enter the project name at line 35 "Enter_project_name" . 
    This run will require wandb login api key. Once run it will store the model in wandb run artifacts along with logs.

model defenitions is available in src/model.py
data loading and transformation code is available in src/utils.py
model weights loading and evaluation is in src/eval.py
model training code with wandb integration is in src/train.py





