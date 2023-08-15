========================================
Autoencoder Convolutional Neural Network
========================================



The ``ae_cnn`` model has three main components; encoder, decoder and dense layers. For input data the model can reconstruct the input data as well as predict the output data. Reconstruction performs a forward pass through the encoder and decoder. Prediction performs a forward pass through the encoder and dense layers. Hyperparameter specification is the same as for the ``cnn`` model type.

The encoder is constructed as ``n_cl`` convolutional layers with the number of ``out_channels`` increasing multiplicately according to the value ``channel_mul``. The decoder is constructed as a sequence of transpose convolutional layers with dimensions matching the corresponding layers in the encoder. The dense layers that perform prediction are comprised of two linear layers. 

**Network Architecture:**

	* **Encoder Layers:**

		* Layer 1
			* 1D Convolution
			* Activation
		* Layer ...
			* 1D Convolution
			* Activation
		* Layer n_cl
			* 1D Convolution
			* Activation
	
	* **Decoder Layers:**

		* Layer 1
			* 1D Transpose Convolution
			* Activation
		* Layer ...
			* 1D Transpose Convolution
			* Activation
		* Layer n_cl
			* 1D Transpose Convolution
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
	  model: ae_cnn
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