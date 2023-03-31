<table align="center">
<tr><td align="center" width="10000">

<img src = "./resources/xanesnet_graphic.png" width = "380">

# <strong> X A N E S N E T </strong>

<p>
    <a href="https://pure.york.ac.uk/portal/en/persons/conor-rankine">Dr. Conor Rankine </a>
    <br>
    <a href="https://ncl.ac.uk/nes/people/profile/tompenfold.html">Prof. Thomas Penfold </a>
    <br>
    <a href="https://rse.ncldata.dev/nik-nikaznan">Dr. Nik Khadijah Nik Aznan </a>
    <br>
    <a href="https://rse.ncldata.dev/kathryn-garside">Dr. Kathryn Garside </a>
</p>

<p>
    <a href="http://penfoldgroup.co.uk">Penfold Group </a> @ <a href="https://ncl.ac.uk">Newcastle University </a>
</p>

<p>
    <a href="#setup">Setup</a> • <a href="#getting">Quickstart</a> • <a href="#publications">Publications</a>
</p>

</td></tr></table>

#

We think that the theoretical simulation of X-ray spectroscopy (XS) should be fast, affordable, and accessible to all researchers. 

The popularity of XS is on a steep upward trajectory globally, driven by advances at, and widening access to, high-brilliance light sources such as synchrotrons and X-ray free-electron lasers (XFELs). However, the high resolution of modern X-ray spectra, coupled with ever-increasing data acquisition rates, brings into focus the challenge of accurately and cost-effectively analyzing these data. Decoding the dense information content of modern X-ray spectra demands detailed theoretical calculations that are capable of capturing satisfactorily the complexity of the underlying physics but that are - at the same time - fast, affordable, and accessible enough to appeal to researchers. 

This is a tall order - but we're using deep neural networks to make this a reality. 

Our XANESNET code addresses two fundamental challenges: the so-called forward (property/structure-to-spectrum) and reverse (spectrum-to-property/structure) mapping problems. The forward mapping appraoch is similar to the appraoch used by computational researchers in the sense that an input structure is used to generate a spectral observable. In this area the objective of XANESNET is to supplement and support analysis provided by first principles quantum mechnanical simulations. The reverse mapping problem is perhaps the more natural of the two, as it has a clear connection to the problem that X-ray spectroscopists face day-to-day in their work: how can a measurement/observable be interpreted? Here we are seeking to provide methodologies in allow the direct extraction of properties from a recorded spectrum. 

XANESNET is under continuous development, so feel free to flag up any issues/make pull requests - we appreciate your input!

The original version of XANESNET, which was implemented using Keras, can be obtained from <a href="https://gitlab.com/team-xnet/xanesnet_keras">here.</a>

## SETUP

The quickest way to get started with XANESNET is to clone this repository:

```
git clone https://gitlab.com/team-xnet/xanesnet.git 
```

This contains all the source files as well as example input files.

Training sets for X-ray absorption and emission of molecules constaining first row transition metals can be obtained using:

```
git clone https://gitlab.com/team-xnet/training-sets.git
```

Now you're good to go!

## GETTING STARTING 

The code has been designed to support python 3. The dependencies and version requirements are installed using:

```
pip install -r requirements.txt
```
and 

```
pip install mlflow
```

### TRAINING 

To train a model, the following command is used:
```python3 ${path}/src/cli.py --mode MODE --model_mode MODEL_MODE --inp_f <inputs/in.yaml>```

The implemented training modes include:
`train_xanes`: The xanes spectra are used the input data and the featurised structures are the target of the model.
`train_xyz`: The featurised structures are used the input data and the xanes spectra are the target of the model.
`train_aegan`: This mode trains featurised structures and xanes spectra simultaneously.

The model modes include:
`mlp`: Feed-foward deep multilayer perceptron model 
`cnn`: Feed-foward deep convolution neural network model 
`ae_mlp`: Autoencoder deep multilayer perceptron model
`ae_cnn`: Autoencoder deep convolution neural network model
`aegan_mlp`: Autoencoder Generative Adversarial Network model using a deep multilayer perceptron network.
`aegan_cnn`: Autoencoder Generative Adversarial Network model using a deep convolution neural network.

Input files for training should be given in yaml format. Example of commented input files for training and hyper parameter options can be found in the inputs folder.

### EVALUATION

A trained model can be evaluated by performing Invariance and T-testing

#### Invariance Testing

Any model predictions should be closer to ground truth than to some artifically constructed target. Four simple methods are implemented to create artificial target data and compare model prediction-target (or reconstruction-target) loss with prediction-modified (reconstruction-modified) target loss.

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

An example of the model evalution can be run using:

```python3 ${path}/src/cli.py eval_pred_xanes --model_mode mlp model/model_001 inputs/in_eval.yaml```

### INFERENCE

To perform inference, the following command is used:
```python3 ${path}/src/cli.py --mode predict_xyz --model_mode mlp --mdl model/model_dir --inp_f inputs/in_predict.yaml```

### PREDICTION

To use a model previously developed model for predictions, the following command is used:
```python3 ${path}/src/cli.py --mode MODE --model_mode MODEL_MODE --mdl model/model_dir --inp_f inputs/in_predict.yaml```

The implemented prediction modes include:
`predict_xyz`: The featurised structure is predicted from an input xanes spectrum. 
`predict_xanes`: The xanes spectrum is predicted from a featurised structural input
`predict_aegan`: Simultaneous prediction of a featurised structure and xanes spectrum from corresponding input.
`predict_aegan_xanes`: The xanes spectrum is predicted from a featurised structural input
`predict_aegan_xyz`: The featurised structure is predicted from an input xanes spectrum.

## PUBLICATIONS

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
