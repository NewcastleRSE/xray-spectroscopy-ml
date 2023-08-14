================
Running the Code
================





----------------
Training a Model
----------------

To train a model, the following command is used: 

.. code-block::

	python3 src/cli.py --mode MODE --model_mode MODEL_MODE --inp_f <inputs/in.yaml>


* **-\ \-mode**

	* ``train_xanes`` Xanes spectra are used the input data and the featurised structures are the target of the model.   
	* ``train_xyz`` Featurised structures are used the input data and the xanes spectra are the target of the model.
	* ``train_aegan`` Trains featurised structures and xanes spectra simultaneously.

* **-\ \-model_mode**

	* ``mlp`` Feed-foward deep multilayer perceptron model  
	* ``cnn`` Feed-foward deep convolution neural network model  
	* ``ae_mlp`` Autoencoder deep multilayer perceptron model 
	* ``ae_cnn`` Autoencoder deep convolution neural network model  
	* ``aegan_mlp``: Autoencoder Generative Adversarial Network model using a deep multilayer perceptron network   
	* ``lstm`` Long Short Term Memory network model

* **-\ \-inp_f**

	Model architecture and training hyperparameters are specified in the input file. The input file should be given in yaml format. Details can be found in :doc:`input_file_training`.


.. toctree::
	:caption: Input YAML Files
	:hidden:

	input_file_training


------------------------------
Addtional Command Line Options
------------------------------

* **-\ \-no-save** 
	Toggle model directory creation and population to off.

* **-\ \-fourier_transform** 
	Train or predict using Fourier transformed spectra. Options are ``True`` or ``False``. Default is ``False``.

* **-\ \-run_shap** 
	SHAP analysis for prediction. Options are ``True`` or ``False``. Default is ``False``.

* **-\ \-shap_nsamples** 
	Number of background samples for SHAP analysis for prediction. Default is integer value ``50``.


-------------------------------------
Experiment Tracking & Logging Results
-------------------------------------

`MLFlow <https://mlflow.org>`_ is used to track hyperparameters, training and validation losses for each training run. Results are automatically logged and users can compare across model runs and track experiments. To open the user interface run the following on the command line and click on the hyperlink:

.. code-block::

	mlflow ui


.. Tensorboard

.. Tensorboard is a tool for visualisation and measurement tracking through the machine learning workflow. During model training the following values are currently logged and accessible through the tensor board,

.. Training & validation loss
.. To run tensorboard, run tensorboard --logdir=/tmp/tensorboard/ --host 0.0.0.0 , click on the hyperlink and choose Custom Scalar.


------------------------
Inference and Prediction
------------------------

A trained model can be used for inference, in which only the input data is given, or prediction, in which matching input and output data are both given. Inference or prediction will be inferred by the number of data paths provided in the input file. If input and output data paths are provided then the mean squared error between the predicted and actual result will be reported. The following command is used for inference/prediction using a trained model: 

.. code-block::
	
	python3 src/cli.py --mode MODE --model_mode MODEL_MODE --mdl_dir <model/model_dir> --inp_f <inputs/in_predict.yaml>

* **-\ \-mode**

	* ``predict_xyz`` The featurised structure is predicted from an input xanes spectrum   
	* ``predict_xanes`` The xanes spectrum is predicted from a featurised structural input  
	* ``predict_all`` Simultaneous prediction of a featurised structure and xanes spectrum from corresponding input as well as reconstruction of inputs. Only for AEGAN model type.


* **-\ \-model_mode** 

	* ``mlp``
	* ``cnn``  
	* ``ae_mlp`` 
	* ``ae_cnn``  
	* ``aegan_mlp``
	* ``lstm``

* **-\ \-mdl_dir**

	Path to trained model directory.

* **-\ \-inp_f**
	
	Input YAML file that specifies path to input data for inference/prediction as well as additional options. Details can be found in :doc:`input_file_predict`.



.. toctree::
	:caption: Input YAML Files
	:hidden:

	input_file_predict



