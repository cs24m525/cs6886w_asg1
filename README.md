# CS6886W - System Engineering for Deep Learning
### Assignment 1
###  Manish Kumar - CS24M525

***
### Model result and configuration
Trained model is loaded under model folder as "best_model.pth"  

__Model Test Accuracy is : 87.1000__

configuration : 

- activation function : SiLU
- optimizer           : Adam
- batch_size          : 128
- learning_rate       : 0.001
- epochs              : 100

***
#### Evaluation of pretrained model
Execute main.py to run the evaluation of pretrained model on test data.

#### Model Training simulation and evaluation of newly trained model
Execute trainmain.py to run the model training simulation.  

After completion of run it will store the model in "trainmodel/trained_best_model.pth". 
Once model is saved, evaluation of saved model will run with the test data.

***
#### Source Codes

        .
        ├── src/
        │   ├── model.py                        model defenition
        │   └── utils.py                        data loading and transformation
        │   └── eval.py                         model weights loading and evaluation 
        │   └── train.py                        model training                  
        ├── model/
        │   └── best_model.pth                  pretrained model
        ├── trainmodel/
        │   └── trained_best_model.pth          after training new trained model will be saved    
        ├── main.py                             evaluation of pretrained model                             
        ├── trainmain.py                        model training simulation and evaluation on new trained model
        └── README.md

***

#### Clone and dependency
Clone the repository using  
```python
git clone https://github.com/cs24m525/cs6886w_asg1.git
```

Install all the necessary dependencies for project using  
```python
pip install -r requirements.txt
```




