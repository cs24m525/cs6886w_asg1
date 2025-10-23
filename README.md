# CS6886W - System Engineering for Deep Learning
### Assignment 1

Trained model is loaded under model folder as "best_model.pth"
    Model Test Accuracy is : 87.1000
    configuration : 
        activation function : SiLU
        optimizer           : Adam
        batch_size          : 128
        learning_rate       : 0.001
        epochs              : 100

Execute main.py to run the evaluation of pretrained model on test data.

Execute trainmain.py to run the model training simulation. 
    After completion of run it will store the model in "trainmodel/trained_best_model.pth".

model defenitions is available in src/model.py
data loading and transformation code is available in src/utils.py
model weights loading and evaluation is in src/eval.py
model training code is in src/train.py

Clone the repository using
```
git clone https://github.com/cs24m525/cs6886w_asg1.git
```

Install all the necessary dependencies for project using 
pip install -r requirements.txt





