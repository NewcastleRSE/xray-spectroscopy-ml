
# X-ray Spectroscopy & Machine Learning

## About

TODO

### Project Team
Tom Penfold, Chemistry, Newcastle University  ([tom.penfold@newcastle.ac.uk](mailto:tom.penfold@newcastle.ac.uk))  
Nik Khadijah Nik Aznan, RSE Team, Newcastle University  ([nik.nik-aznan@ncl.ac.uk](mailto:nik.nik-aznan@ncl.ac.uk))  
Kathryn Garside, RSE Team, Newcastle University ([kathryn.garside@newcastle.ac.uk](mailto:kathryn.garside@newcastle.ac.uk))


## Getting Started

### Prerequisites
The code has been designed to support python 3. The project has the following dependencies and version requirements:

- ase
- matplotlib
- numpy
- scikit_learn
- seaborn
- tensorflow>=2.1.0
- torch
- torchinfo
- tqdm


### Installation

```
pip install -r requirements.txt
```
```
pip install mlflow
```

## Training, Inference & Evaluation

To either train a model or perform inference the following command is used:  
```python src/cli.py MODE --model_mode MODEL_MODE <inputs/in.json>```

Select MODE from the following:  
`train_xanes`, `train_xyz`, `train_aegan`, `predict_xyz`, `predict_xanes`, `predict_aegan`, `predict_aegan_xanes`, `predict_aegan_xyz` 

Select MODEL_MODE from the following:  
`mlp`, `cnn`, `ae_mlp`, `ae_cnn`, `aegan_mlp`, `aegan_cnn`

Input for training and prediction should be given in JSON format. The prediction input file gives the path to the input data. Example input files for training and hyper parameter options can be found in the [inputs](https://github.com/NewcastleRSE/xray-spectroscopy-ml/tree/main/inputs) folder.

#### Example of training and inference. 
```python src/cli.py train_xanes --model_mode mlp inputs/in.json```  
```python src/cli.py predict_xanes --model_mode mlp model/model_dir inputs/in_predict.json```


### Tensorboard

[Tensorboard](https://www.tensorflow.org/tensorboard/get_started) is a tool for visualisation and measurement tracking through the machine learning workflow. During model training the following values are currently logged and accessible through the tensor board,
- Training & validation loss

To run tensorboard, run ```tensorboard --logdir=/tmp/tensorboard/ --host 0.0.0.0``` , click on the hyperlink and choose Custom Scalar.

### MLFlow

[MLFlow](https://mlflow.org) is used to track the hyperparameters and the loss for every run.

To run the ui, run ```mlflow ui``` on the terminal (make sure the env is correct and the directory is where the local mlruns folder is), click on the hyperlink.

### Evaluation 

A trained model can be evaluated by performing tests. ```core_eval.py``` currently implements a few basic tests but it can be extended to include any test of a models predictive or reconstructive ability. To suggest more tests please [raise an issue](https://github.com/NewcastleRSE/xray-spectroscopy-ml/issues).

#### Invariance Testing

Model predictions should be closer to ground truth than to some artifically constructed target.  
Four simple methods are implemented to create artificial target data and compare model prediction-target (or reconstruction-target) loss with prediction-modified (reconstruction-modified) target loss.

- ```shuffle-output``` Shuffled test set as modified output
- ```mean_train_output``` Mean of training data as modified output
- ```random_train_output``` Random training data as modified output
- ```guass_train_output``` Simulated from Normal distribution with mean and stdev of training data as modified output

We similarly modify the input data and compare predictions based on this input with true target.

- ```shuffle_input``` 
- ```mean_train_input``` 
- ```random_train_input``` 
- ```guass_train_input``` 

A T-Test is used to return ```True``` or ```False``` if the model has passed the test positively.


#### Example of evaluation of original MLP model used to predict XANES:

```python src/cli.py eval_pred_xanes --model_mode mlp model/model_001 inputs/in_eval.json```


## Available Models

### Original XANES PyTorch Implementation

<!---
To run the mlp version call ```model, score = train_mlp(... )``` in the ```core_learn.py``` and run ```python cli.py learn in.json```.

To run the cnn version call ```model, score = train_cnn(... )``` in the ```core_learn.py``` and run ```python cli.py learn in_cnn.json```.
--->

### AutoEncoder

<!---
To run the basic AutoEncoder to train xanes :
```python src/cli_ae.py train_xanes inputs/in_cnn.json```
and run ```python src/cli_ae.py predict_xyz model/model_0xx inputs/in_predict.json``` to run the test.

To run the basic AutoEncoder to train xyz :
```python src/cli_ae.py train_xyz inputs/in_cnn.json```
and run ```python src/cli_ae.py predict_xanes model/model_0xx inputs/in_predict.json``` to run the test.

--->

### Autoencoder Generative Adversarial Network
Trains model via a coupled Autoencoder Generative adverserial network. It consists of two generative networks, each built from an encoder and decoder block. A shared layer connects the two networks and enables cross-domain paths through the network, allowing reconstruction and prediction for both of the inputs. A discriminator network compares real inputs with those that have either been reconstructed or predicted, forcing the generative network to perform better.

The trained model can then be used for prediction and reconstruction of both structure and spectrum. 

- Predict and reconstruct all  
```python src/cli.py predict_aegan --model_mode aegan_mlp model_dir inputs/in_predict.json```  
- Predict spectrum, reconstruct input structure  
```python src/cli.py predict_aegan_xanes --model_mode aegan_mlp model_dir inputs/in_predict.json```  
- Predict structure, reconstruct input spectrum  
```python src/cli.py predict_aegan_xyz --model_mode aegan_mlp model_dir inputs/in_predict.json```  

<!---
A general layer in the model is MLP consisting of a linear layer, batch norm layer, activation. 

Example model parameters can be found in `in_aegan.json`. The user can specify hidden size of linear layers (*hidden_size*), dropout (*dropout*), the number of hidden layers in the encoder-decoder (*n_hl_gen*), shared (*n_hl_shared*) and discriminator (*n_nl_dis*) networks, activation function (*activation*), loss function for the generative (*loss_gen*) and discriminator (*loss_dis*) networks, learning rates (*lr_gen* and *lr_dis*).
--->

### Uncertainties

The flag ```"True" or "False"``` for bootstrap is in ```inputs/in.json```, ```inputs/in_cnn.json```, and ```inputs/in_aegan.json```. By default, the flag is set to "False".

To run the bootstrap for prediction, run ```python src/cli.py predict_xyz --model xxx bootstrap/bootstrap_0xx inputs/in_predict.json```.

To run ensemble during prediction, change the flag for ensemble in ```inputs/in_predict.json``` to "True". Choose how to combine the model by either combining the prediction ```"combine": "prediction"``` or combining the weight ```"combine": "weight"```

The flag ```"True" or "False"``` for monte-carlo dropout is in ```inputs/in_predict.json```. By default, the flag is set to "False".



## Roadmap

- [x] PyTorch Implementation of Original Model 
- [x] AE
	- [x] MLP
	- [x] CNN
- [ ] AEGAN
	- [x] MLP
	- [ ] CNN
- [ ] Testing framework  

## Contributing


### RSE Contact
Nik Khadijah Nik Aznan, RSE Team, Newcastle University  ([nik.nik-aznan@ncl.ac.uk](mailto:nik.nik-aznan@ncl.ac.uk))  
Kathryn Garside, RSE Team, Newcastle University ([kathryn.garside@newcastle.ac.uk](mailto:kathryn.garside@newcastle.ac.uk))


<!---
### Main Branch
Protected and can only be pushed to via pull requests. Should be considered stable and a representation of production code.

### Dev Branch
Should be considered fragile, code should compile and run but features may be prone to errors.

### Feature Branches
A branch per feature being worked on.

https://nvie.com/posts/a-successful-git-branching-model/




## License

## Citiation

Please cite the associated papers for this work if you use this code:

```
@article{xxx2021paper,
  title={Title},
  author={Author},
  journal={arXiv},
  year={2021}
}
```
## Acknowledgements
This work was funded by a grant from the UK Research Councils, EPSRC grant ref. EP/L012345/1, “Example project title, please update”.

--->
