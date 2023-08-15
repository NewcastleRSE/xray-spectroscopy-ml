==============================
Long Short Term Memory Network
==============================


The ``lstm`` model is constructed as a `PyTorch LSTM <https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html>`_ class with ``num_layers`` and two dense layers. 


**Network Architecture:**

A forward pass through the model passes through these layers in sequence:

	* LSTM Layer
		* LSTM 
	* Dense Layer 1
		* Linear
		* Activation
		* Dropout
	* Dense Layer 2
		* Linear


**Example hyperparameters:**


.. code-block::

	hyperparams:
	  model: lstm
	  batch_size: 64
	  hidden_size: 100
	  num_layers: 1
	  hl_ini_dim : 50
	  activation: prelu
	  loss:
	    loss_fn: mse
	    loss_args: null
	    loss_reg_type: L2
	    loss_reg_param: 0.001
	  lr: 0.00001
	  optim_fn: Adam
	  dropout: 0.2
	  weight_init_seed: 2023
	  kernel_init: xavier_uniform
	  bias_init: zeros