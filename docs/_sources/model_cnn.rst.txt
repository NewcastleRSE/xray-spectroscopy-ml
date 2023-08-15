============================
Convolutional Neural Network
============================


The ``cnn`` model is constructed as ``n_cl`` convolutional layers and two dense layers. The convolutional layers are comprised of a 1D convolution, batch normalisation, activation function and dropout. The number of ``out_channels`` in the convolutional layers is increased multiplicatively at each layer by the value ``channel_mul``.

**Network Architecture:**

A forward pass through the model passes through these layers in sequence:

	* Convolutional Layer 1
		* 1D Convolution
		* Batch Normalisation
		* Activation
		* Dropout
	* Convolutional Layer ...
		* 1D Convolution
		* Batch Normalisation
		* Activation
		* Dropout
	* Convolutional Layer n_cl
		* 1D Convolution
		* Batch Normalisation
		* Activation
		* Dropout
	* Dense Layer 1
		* Linear
		* Activation
	* Dense Layer 2
		* Linear



**Example hyperparameters:**


.. code-block::

	hyperparams: 
	  model: cnn
	  batch_size: 16
	  n_cl: 3
	  out_channel: 32
	  channel_mul: 2
	  hidden_layer: 64
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
	  kernel_size: 3
	  stride: 1