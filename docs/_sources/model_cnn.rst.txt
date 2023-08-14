============================
Convolutional Neural Network
============================


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