=========================================
Autoencoder Multilayer Perceptron Network
=========================================


The ``ae_mlp`` model has three main components; encoder, decoder and dense layers. For input data the model can reconstruct the input data as well as predict the output data. Reconstruction performs a forward pass through the encoder and decoder. Prediction performs a forward pass through the encoder and dense layers. Hyperparameter specification is the same as for the ``mlp`` model type.

The encoder is constructed as ``n_hl`` dense layers with the dimension of each linear layer decreasing multiplicately according to the value ``hl_shrink``. The decoder is similarly constructed with the dimension of each linear layer instead increasing multiplicately according to ``hl_shrink``. The dense layers that perform prediction are comprised of two linear layers. An error will be returned if the last hidden layer size of the encoder is less than 1. 

**Network Architecture:**

	* **Encoder Layers:**

		* Layer 1
			* Linear
			* Activation
		* Layer ...
			* Linear
			* Activation
		* Layer n_hl
			* Linear
			* Activation
	
	* **Decoder Layers:**

		* Layer 1
			* Linear
			* Activation
		* Layer ...
			* Linear
			* Activation
		* Layer n_hl
			* Linear
			* Activation

	* **Dense Layers:**
		* Layer 1
			* Linear
			* Activation
			* Dropout
		* Layer 2
			* Linear




**Example hyperparameters:**

.. code-block::

	hyperparams: 
	  model: ae_mlp
	  batch_size: 64
	  n_hl: 5
	  hl_ini_dim: 512
	  hl_shrink: 0.5
	  activation: prelu
	  loss:
	    loss_fn: mse
	    loss_args: null
	    loss_reg_type: L2
	    loss_reg_param: 0.001
	  lr: 0.0001
	  optim_fn: Adam
	  dropout: 0.2
	  weight_init_seed: 2023
	  kernel_init: xavier_uniform
	  bias_init: zeros