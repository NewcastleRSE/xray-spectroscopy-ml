=============================
Multilayer Perceptron Network
=============================



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