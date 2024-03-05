<table align="center">
<tr><td align="center" width="10000">

<img src = "./resources/xanesnet_graphic.png" width = "380">

# <strong> X A N E S N E T </strong>

<p>
    <a href="http://penfoldgroup.co.uk">Penfold Group </a> @ <a href="https://ncl.ac.uk">Newcastle University </a>
</p>

<p>
    <a href="https://xanesnet.readthedocs.io">User Manual</a> • <a href="#setup">Setup</a> • <a href="#getting">Quickstart</a> • <a href="#publications">Publications</a>
</p>

</td></tr></table>

#

We think that the theoretical simulation of X-ray spectroscopy (XS) should be fast, affordable, and accessible to all researchers. 

The popularity of XS is on a steep upward trajectory globally, driven by advances at, and widening access to, high-brilliance light sources such as synchrotrons and X-ray free-electron lasers (XFELs). However, the high resolution of modern X-ray spectra, coupled with ever-increasing data acquisition rates, brings into focus the challenge of accurately and cost-effectively analyzing these data. Decoding the dense information content of modern X-ray spectra demands detailed theoretical calculations that are capable of capturing satisfactorily the complexity of the underlying physics but that are - at the same time - fast, affordable, and accessible enough to appeal to researchers. 

This is a tall order - but we're using deep neural networks to make this a reality. 

Our XANESNET code address two fundamental challenges: the so-called forward (property/structure-to-spectrum) and reverse (spectrum-to-property/structure) mapping problems. The forward mapping appraoch is similar to the appraoch used by computational researchers in the sense that an input structure is used to generate a spectral observable. In this area the objective of XANESNET is to supplement and support analysis provided by first principles quantum mechnanical simulations. The reverse mapping problem is perhaps the more natural of the two, as it has a clear connection to the problem that X-ray spectroscopists face day-to-day in their work: how can a measurement/observable be interpreted? Here we are seeking to provide methodologies in allow the direct extraction of properties from a recorded spectrum. 

XANESNET is under continuous development, so feel free to flag up any issues/make pull requests - we appreciate your input!

The original version of XANESNET, which was implemented using Keras, can be obtained from <a href="https://gitlab.com/team-xnet/xanesnet_keras">here.</a>. The <a href="https://xanesnet.readthedocs.io">XANESNET User Manual</a> has more information about the code and its uses.

## Features

* GPLv3 licensed open-source distribution
* Automated data processing: data augmentation, Fourier transform
* Feature extraction: wACSF, SOAP, MBTR, LMBTR, RDC, LMBTR, pDOS, MSR
* Neural network architecture: MLP, CNN, LSTM, Autoencoder, Autoencoder-GAN
* Learning scheme: K-fold, ensemble, bootstrap
* Hyperparameter optimisation
* Learning rate scheduler
* Layer freezing
* Run from an input file
* Easy to extend with new model and descriptor
* Web interface


## Setup

The quickest way to get started with XANESNET is to clone this repository:


<!---
```
git clone https://gitlab.com/team-xnet/xanesnet.git 
```
--->

```
git clone https://gitlab.com/team-xnet/xanesnet.git
```

This contains all the source files as well as example input files.

Training sets for X-ray absorption and emission of molecules constaining first row transition metals can be obtained using:

```
git clone https://gitlab.com/team-xnet/training-sets.git
```

Now you're good to go!

## Getting Started

The code has been designed to support python 3. The dependencies and version requirements are installed using:

```
python -m pip install .
```

### Training 

To train a model, the following command is used:  

```python3 -m xanesnet.cli --mode MODE --in_file <path/to/file.yaml>```

The implemented training MODE include:  
- `train_xanes`: The xanes spectra are used the input data and the featurised structures are the target of the model   
- `train_xyz`: The featurised structures are used the input data and the xanes spectra are the target of the model   
- `train_aegan`: This mode trains featurised structures and xanes spectra simultaneously   

Replace <path/to/file.yaml> by the actual path to a YAML input file.
Examples of commented input files for training and hyperparameter 
options can be found in the 'inputs/' folder

Below is an example command for training a model using MLP architecture, and featurised structures as input data:  

```python3 -m xanesnet.cli --mode train_xyz --in_file inputs/in_mlp.yaml```

The resulting trained model and its metadata will be automatically saved in the 'models/' directory. 

### Prediction

To use a previously developed model for predictions, the following command is used:  

```python3 -m xanesnet.cli --mode MODE --in_model <path/to/model> --in_file <path/to/file.yaml>```

The implemented prediction modes include:  
- `predict_xyz`: The featurised structure is predicted from an input xanes spectrum   
- `predict_xanes`: The xanes spectrum is predicted from a featurised structural input  
- `predict_all`: Simultaneous prediction of a featurised structure and xanes spectrum from corresponding input as well as reconstruction of inputs. Only for AEGAN model type. 

The [-\-in_model] option specifies a directory containing pre-trained model and its metadata.
The [-\-in_file] specifies a path to input file for prediction. As an example, execute the following command to make spectrum predictions using the previously trained MLP model:   

```python3 -m xanesnet.cli --mode predict_xanes --in_model models/model_mlp_001 --in_file inputs/in_predict.yaml```


## Contact

### Project Team

<a href="https://ncl.ac.uk/nes/people/profile/tompenfold.html">Prof. Thomas Penfold </a>, Newcastle University, (tom.penfold@newcastle.ac.uk)\
<a href="https://pure.york.ac.uk/portal/en/persons/conor-rankine">Dr. Conor Rankine </a>, York University (conor.rankine@york.ac.uk)

### RSE Contact
<a href="https://rse.ncldata.dev/team/bowen-li">Dr. Bowen Li </a>, Newcastle University (bowen.li2@newcastle.ac.uk)\
<a href="https://rse.ncldata.dev/team/alex-surtees">Alex Surtees </a>,  Newcastle University (alex.surtees@newcastle.ac.uk)

<!--
### EVALUATION

A trained model can be evaluated by performing some simple invariance tests. To allow model evaluation during training a user can set `model_eval: True` in the input yaml file. Input data is split into three; training (75%), validation (15%) and testing (10%). 

For a model to be useful its predictions should be closer to ground truth than to some artifically constructed target. During model evaluation four methods are implemented to create artificial target data and the distribution of losses is compared to the true test data losses. The artificial and true losses are compared using a one-sided T-Test at the 5% level. A `True` or `False` value is returned if the model performs better than a model which does not learn from the data and returns artifical output. Results should be treat as informative but not binding. 

The four methods of creating artificial data currently include:  
- Shuffling labels on the test data  
- Using the mean of training data  
- Shuffling the labels of the training data  
- Simulating as the mean of training data plus some Normally distributed noise with standard deviation matching the training data   

Results are logged in MLflow. 


### TUNING HYPERPARAMETERS

Model hyperparameters can be automatically tuned by setting `tune: True` within the `optuna_params` in the user input yaml file. 

Tuning uses [Optuna](https://optuna.org), an open source hyperparameter optimization framework to automate hyperparameter search. The user can also specify which hyperameters they would like to tune in `optuna_params`, for example exploring the effect of different activation functions whilst holding all other hyperparameters static. Default options for each hyperparameter are specified with the `optuna_defaults()` function within `optuna_learn.py` and can easily be extended or modified to restrict or include other options. 

The user should specify the number of trials they wish to run during tuning. After tuning the optimal values found will be used to train the model.
-->

## License

This project is licensed under the GPL-3.0 License - see the LICENSE.md file for details.

## Publications

### The Program:
*[A Deep Neural Network for the Rapid Prediction of X-ray Absorption Spectra](https://doi.org/10.1021/acs.jpca.0c03723)* - C. D. Rankine, M. M. M. Madkhali, and T. J. Penfold, *J. Phys. Chem. A*, 2020, **124**, 4263-4270.

*[Accurate, affordable, and generalizable machine learning simulations of transition metal x-ray absorption spectra using the XANESNET deep neural network](https://doi.org/10.1063/5.0087255)* - C. D. Rankine, and T. J. Penfold, *J. Chem. Phys.*, 2022, **156**, 164102.
 
#### Extension to X-ray Emission:
*[A deep neural network for valence-to-core X-ray emission spectroscopy](https://doi.org/10.1080/00268976.2022.2123406)* - T. J. Penfold, and C. D. Rankine, *Mol. Phys.*, 2022, e2123406.

#### The Applications:
*[On the Analysis of X-ray Absorption Spectra for Polyoxometallates](https://doi.org/10.1016/j.cplett.2021.138893)* - E. Falbo, C. D. Rankine, and T. J. Penfold, *Chem. Phys. Lett.*, 2021, **780**, 138893.

*[Enhancing the Anaysis of Disorder in X-ray Absorption Spectra: Application of Deep Neural Networks to T-Jump-X-ray Probe Experiments](https://doi.org/10.1039/D0CP06244H)* - M. M. M. Madkhali, C. D. Rankine, and T. J. Penfold, *Phys. Chem. Chem. Phys.*, 2021, **23**, 9259-9269.

#### Miscellaneous:
*[The Role of Structural Representation in the Performance of a Deep Neural Network for X-ray Spectroscopy](https://doi.org/10.3390/molecules25112715)* - M. M. M. Madkhali, C. D. Rankine, and T. J. Penfold, *Molecules*, 2020, **25**, 2715.
