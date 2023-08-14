==========================================
Autoencoder Generative Adversarial Network
==========================================


.. code-block::

	hyperparams: 
	  model: aegan
	  batch_size: 64
	  n_hl_gen: 2
	  n_hl_shared: 2
	  n_hl_dis: 2
	  hidden_size: 256
	  activation: prelu
	  loss_gen:
	    loss_fn: mse
	    loss_args: 
	    loss_reg_type: null
	    loss_reg_param: 0.001
	  loss_dis:
	    loss_fn: bce
	    loss_args: null
	    loss_reg_type: null
	    loss_reg_param: 0.001
	  lr_gen: 0.01
	  lr_dis: 0.0001
	  optim_fn_gen: Adam
	  optim_fn_dis: Adam
	  dropout: 0.0
	  weight_init_seed: 2023
	  kernel_init: xavier_uniform
	  bias_init: zeros