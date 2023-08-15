=============================
Multilayer Perceptron Network
=============================

The ``mlp`` model is constructed as ``n_hl`` dense layers. All layers except the final output layer are comprised of a linear layer, a dropout layer and the activation function. The final layer is a linear layer. The size of each hidden linear layer is controlled via the intial dimension ``hl_ini_dim`` and the value ``hl_shrink`` that reduces the layer dimension multiplicatively. An error will be returned if the last hidden layer size is less than 1. 

**Network Architecture:**

A forward pass through the model passes through these layers in sequence:

	* Dense Layer 1
		* Linear
		* Dropout
		* Activation
	* Dense Layer 2
		* Linear
		* Dropout
		* Activation
	* Dense Layer ...
		* Linear
		* Dropout
		* Activation
	* Dense Layer n_hl
		* Linear

**Example hyperparameters:**

.. code-block::

	hyperparams: 
	  model: mlp
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