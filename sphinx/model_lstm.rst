==============================
Long Short Term Memory Network
==============================


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