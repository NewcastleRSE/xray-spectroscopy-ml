Input File
===============

The XANESNET input file is a YAML configuration file that specifies all the necessary parameters for the training, evaluation, and prediction.
The input file is divided into several sections, each corresponding to a different aspect of the configuration.
This page details the syntax of each section,
providing guidance on how to customise your model.

=====
path
=====

The path section specifies the file directories containing the input data required for the model training.

* ``xyz_path`` (str): Directory for the XYZ files.
* ``xanes_path`` (str): Directory for the XANES spectra files.

Example:
    .. code-block::

        x_path:  data/datasets/fe/preconv/xyz_train
        y_path:  data/datasets/fe/preconv/xanes_train

===========
descriptor
===========

The descriptor section defines the type of descriptor to be used for feature extraction.
A descriptor aims to transform atomic structures into fixed-size numeric vectors.
These vectors are built in a way that they efficiently summarise the contents of the input structure,
and thus are crucial for machine learning models.

XANESNET provides a range of descriptor functions. Each of which requires different parameters to be specified.
Please see :doc:`descriptor` for a detailed explanation.

* ``type`` (str): Type of descriptor (see :doc:`descriptor`).
* ``params``: Parameters for the chosen descriptor type.

========
model
========

The model section defines the architecture and specific parameters of the neural network model.
This section determines how the model is structured, including the type of neural network and the hyperparameters
that control its training and operation.

XANESNET supports a variety of widely-used deep neural network architecture, including
Multilayer Perceptron (MLP), Convolutional Neural Networks (CNNs), Long Short-Term Memory networks (LSTMs),
Autoencoders (AE), and Generative Adversarial Autoencoder Networks (AE-GAN).
Their architectures and parameters are further explained in the :doc:`model`.

* ``type`` (str): Type of model (see :doc:`model`).
* ``params``: Model-specific parameters.

================
hyperparameters
================

The hyperparameter section defines the settings that
govern the training process of the neural network model.
Hyperparameters are the configuration variables that are
external to the model and whose values are set before the training process begins.

* ``batch_size`` (int): Size of training batches.
* ``lr`` (float): Learning rate for the optimiser.
* ``epochs`` (int): Maximum number of epochs.
* ``optim_fn`` (str): Type of optimisation function. Options: *Adam* or *SGD* or *RMSprop*
* ``kernel_init`` (str): Function for initialising the model weights. Options: *xavier_uniform* or *xavier_normal* or *kaiming_uniform* or *normal* or *uniform*
* ``bias_init`` (str): Function for initialising the model bias. Options: *zeros* or *ones*
* ``model_eval`` (bool):  Flag for model evaluation during training, see :doc:`evaluation`.
* ``seed`` (int): Random seed for data preprocessing
* ``weight_seed`` (int): Random seed for weight initialisation.
* ``loss``:

  * ``loss_fn`` (str): Type of loss function. Options: *mse* or *bce* or *emd* or *cosine* or *l1* or *wcc*
  * ``loss_args`` (float or 'null'): Additional arguments for the loss function if needed.
  * ``loss_reg_type`` (str): regularisation type. Options: *null* or *L1* or *L2*
  * ``loss_reg_param`` (float): strength of regularisation.

Example:
    .. code-block::

        hyperparams:
          batch_size: 64
          lr: 0.00001
          epochs: 100
          optim_fn: Adam
          kernel_init: xavier_uniform
          bias_init: zeros
          model_eval : False
          seed: 2021
          weight_seed: 2023
          loss:
            loss_fn: mse
            loss_args: null
            loss_reg_type: L2
            loss_reg_param: 0.001

==================
fourier_transform
==================

The fourier_transform section enables XANESNET to train or predict using
Fourier transformed spectra.
The transformation converts the XANES spectra data into the frequency domain
that emphasise its frequency components.

* ``fourier_transform`` (bool): Flag for toggling Fourier transformation on or off

Example:
    .. code-block::

        fourier_transform: True

==============
lr_scheduler
==============

The lr_scheduler (Learning Rate Scheduler) section specifies
whether to employ a scheduler to dynamically adjusting the learning rate during the training.
A learning rate scheduler modifies the learning rate over time,
typically reducing it according to a predefined schedule
or based on the model's performance.

* ``lr_scheduler`` (bool): Flag for toggling learning rate scheduler on or off
* ``scheduler_params``:

  * ``type`` (str): Type of scheduler. Options: *step* or *multistep* or *exponential* or *linear* or *constant*
  * ``step_size`` (int):  Number of epochs between each learning rate adjustment.
  * ``gamma`` (float):  Multiplicative factor of learning rate decay.

Example:
    .. code-block::

        lr_scheduler: True
        scheduler_params:
          type: step
          step_size: 100
          gamma: 0.5

==============
data_augment
==============

The data_augment section defines if data augmentation is applied to the training dataset.
The method is used to prevent overfitting by increasing the diversity of the dataset.
XANESNET implements two augmentation methods: the *random_noise* option
augments the data by adding noise from a normal distribution to the dataset, and the
*random_combination* option expands the existing dataset by generating new data points
that represent the pointwise average of two randomly chosen samples from the training dataset.

* ``data_augment`` (bool): Flag for toggling data augmentation on or off
* ``augment_params``:

  * ``type`` (str): Type of augmentation. Options: *random_noise* or *random_combination*
  * ``augment_mult`` (int): Multiplier for how much augmented data is generated.
  * ``normal_mean`` (float): mean of the normal distribution (applicable only for the *random_noise* option).
  * ``normal_sd`` (float): standard deviation of the normal distribution (applicable only for the *random_noise* option).

Example:
    .. code-block::

        data_augment: True
        augment_params:
          type: random_noise
          augment_mult: 5
          normal_mean: 0
          normal_sd: 0.1


=======
kfold
=======

The kfold (K-Fold Cross-Validation) section specifies
whether to use K-Fold Cross-Validation during the model training process.
If specified, the entire dataset is divided into 'k' into k subsets or folds.
The model is then trained 'k' times, each time using a different fold as the validation set (for testing the model)
and the remaining 'k-1' folds as the training set.

* ``kfold`` (bool): Flag for toggling kfold on or off
* ``kfold_params``:

  * ``n_splits`` (int): Number of folds or splits.
  * ``n_repeats`` (int):  Number of times the k-fold cross-validation is repeated.
  * ``seed`` (str): Random seed for splitting the dataset.

Example:
    .. code-block::

        kfold: True
        kfold_params:
          n_splits: 5
          n_repeats: 1
          seed: 2022

==============
bootstrap
==============

The bootstrap section enables XANESNET to train the model using bootstrapping method.
If specified, initial training dataset is randomly resampled to
create a new dataset.
The size of this new dataset is determined by multiplying the
original dataset size by a user-defined factor.
The model is then trained using this bootstrapped data for multiple times,
each time with a new random seed.

* ``bootstrap`` (bool): Flag for toggling bootstrap on or off
* ``bootstrap_params``:

  * ``n_boot`` (int): Number of repeats.
  * ``n_size`` (float): Multiplication factor of the dataset size.
  * ``weight_seed`` (list): List of random seeds for each sample.

Example:
    .. code-block::

        bootstrap: True
        bootstrap_params:
          n_boot: 3
          n_size: 1.0
          weight_seed: [97, 39, 22]


==============
ensemble
==============

The ensemble section enables XANESNET to employ ensemble method for model training.
The approach uses different seeds to initialise the weights and biases of the model.
The model is then trained for multiple times, each time with a distinct set of initialisation parameters.

* ``ensemble`` (bool): Flag for toggling bootstrap on or off
* ``ensemble_params``:

  * ``n_ens`` (int): Number of repeats.
  * ``weight_seed`` (list): List of random seeds for model initialisation.

Example:
    .. code-block::

        ensemble: True
        ensemble_params:
          n_ens: 3
          weight_seed: [97, 39, 22]

=========
optuna
=========

The optuna section allows XANESNET to integrate Optuna, a hyperparameter
optimisation framework, into the model training process.
Optuna automatically searches for the best hyperparameter values by exploring
various combinations and evaluating their performance.

The ``optuna_params`` contains a list of flags (i.e, tune_xxx)
that determine which hyperparameters will be optimised.
The predefined optimisation options for each parameter are detailed in the description.

* ``optuna`` (bool): Flag for toggling optuna on or off
* ``optuna_params``:

  * ``n_trials`` (int): Number of trials.
  * ``tune_optim_fn`` (bool):  *Adam*, *SGD*, *RMSprop*
  * ``tune_batch_size`` (bool): 8, 16, 32, 64
  * ``tune_activation`` (bool):  *relu*, *prelu*, *tanh*, *sigmoid*, *elu*, *leakyrelu*, *selu*
  * ``tune_loss_fn`` (bool):  *mse*, *emd*, *cosine*, *l1*, *wcc (min 5, max 15)*
  * ``tune_lr`` (bool): min 1e-7, max 1e-3
  * ``tune_dropout`` (bool):  min 0.2, max 0.5
  * ``tune_mlp`` (bool): **Model MLP and AE-MLP specific**

    * number of hidden layers: min 2, max 5
    * hidden size: 64, 128, 256, 512
    * shrink rate: min 0.2, max 0.5

  * ``tune_cnn`` (bool): **Model CNN and AE-CNN specific**

    * number of convolutional layers: min 1, max 5
    * hidden size: 64, 128, 256, 512

  * ``tune_lstm`` (bool): **Model LSTM specific**

    * number of hidden layers: min 2, max 5
    * hidden size: 64, 128, 256, 512

  * ``tune_aegan_mlp`` (bool): **Model AE-GAN specific**

    * learning rate: min 1e-7, max 1e-3
    * number of hidden layers: min 2, max 5


Example:
    .. code-block::

        optuna: True
        optuna_params:
          n_trials: 3
          tune_optim_fn: True
          tune_batch_size: True
          tune_activation: True
          tune_loss_fn: True
          tune_lr: True
          tune_dropout: True
          tune_mlp: True


=========
freeze
=========

The freeze section specifies whether to use transfer learning strategies
by freezing certain layers of a pre-trained model during the training process.
The approach is useful when the model trained on a large dataset needs to
be fine-tuned on a smaller dataset.



* ``freeze`` (bool): Flag for toggling freeze on or off
* ``freeze_params``:

  * ``model_path`` (str): Path to the pre-trained model.
  * ``n_dense`` (int): Number of dense layers to be frozen.
  * ``n_conv`` (int): Number of convolutional layers to be frozen. **Model CNN and AE-CNN specific**
  * ``n_lstm`` (int): Number of LSTM layers to be frozen. **Model LSTM specific**
  * ``n_encoder`` (int): Number of encoder layers to be frozen. **Model AE-MLP and AE-CNN specific**
  * ``n_decoder`` (int): Number of decoder layers to be frozen. **Model AE-MLP and AE-CNN specific**
  * ``n_encoder1`` (int): Number of encoder layers in domain A to be frozen. **AE-GAN specific**
  * ``n_encoder2`` (int): Number of encoder layers in domain B to be frozen. **AE-GAN specific**
  * ``n_decoder1`` (int): Number of decode layers in domain A to be frozen. **AE-GAN specific**
  * ``n_decoder2`` (int): Number of decode layers in domain B to be frozen. **AE-GAN specific**
  * ``n_shared_encoder`` (int): Number of shared encoder layers to be frozen. **AE-GAN specific**
  * ``n_shared_decoder`` (int): Number of shared decoder layers to be frozen. **AE-GAN specific**
  * ``n_discrim1`` (int): Number of discriminative layers in domain A to be frozen. **AE-GAN specific**
  * ``n_discrim2`` (int): Number of discriminative layers in domain B to be frozen. **AE-GAN specific**

Example:
    .. code-block::

        freeze: True
        freeze_params:
          model_path: ./models/model_cnn_001
          n_conv: 4
          n_dense: 1

=========
shap
=========

The shap section enables XANESNET to use SHAP (SHapley Additive exPlanations)
for interpreting the model prediction result. The analysis can quantify the
contribution of each feature to the prediction,
and contrasting the prediction with the average prediction across the dataset.
The model output will be sampled for different subsets of
the input features to estimate the impact of each feature on the prediction.

* ``shap`` (bool): Flag for toggling SHAP analysis on or off
* ``shap_params``:

  * ``nsamples`` (int):  Number of samples for calculating SHAP values.

Example:
    .. code-block::

        shap: True
        shap_params:
          nsamples: 5