===============================
Model Modes and Hyperparameters
===============================

-----------------------------
General Model Hyperparameters
-----------------------------

Each model type has specific hyperparameters used to specify the architecture of the network. These model specific hyperparameters are described on their respective model type page linked below. However, there are some common hyperparameters across all model modes.

.. code-block::

	hyperparams:
		...
		batch_size: 32
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
		...

* **batch_size** [int]

	Size of training batches.

* **activation** [str]
	The user can choose from the following activation functions specified with the matching string in the input file:

	* prelu = `PReLU <https://pytorch.org/docs/stable/generated/torch.nn.PReLU.html#torch.nn.PReLU>`_ (default)  
	* relu = `ReLU <https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html#torch.nn.ReLU>`_  
	* tanh = `Tanh <https://pytorch.org/docs/stable/generated/torch.nn.Tanh.html#torch.nn.Tanh>`_
	* sigmoid = `Sigmoid <https://pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html#torch.nn.Sigmoid>`_
	* elu = `ELU <https://pytorch.org/docs/stable/generated/torch.nn.ELU.html#torch.nn.ELU>`_
	* leakyrelu = `LeakyReLU <https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html#torch.nn.LeakyReLU>`_
	* selu = `SELU <https://pytorch.org/docs/stable/generated/torch.nn.SELU.html#torch.nn.SELU>`_

* **loss** [dict]

	For model modes `mlp`, `cnn`, `ae_mlp` and `ae_cnn` the user specifies the loss function via the `loss` hyperparameter. For model mode `aegan_mlp` the user specifies the loss for the generative and discriminative parts of the model via the hyperparameters `loss_gen` and `loss_dis` respectively. The loss function is further specified using the sub-hyperparameters. An example:

	.. code-block::
	
		loss:
		    loss_fn: mse
		    loss_args: null
		    loss_reg_type: L2
		    loss_reg_param: 0.001

	* **loss_fn** [str]

		* mse = `MSELoss <https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html#torch.nn.MSELoss>`_ (default)
		* bce = `BCEWithLogitsLoss <https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html#torch.nn.BCEWithLogitsLoss>`_
		* emd = `Earth Mover Distance Loss <https://en.wikipedia.org/wiki/Earth_mover's_distance>`_
		* cosine = `Cosine Similarity Loss <https://en.wikipedia.org/wiki/Cosine_similarity>`_
		* l1 = `L1Loss <https://pytorch.org/docs/stable/generated/torch.nn.L1Loss.html#torch.nn.L1Loss>`_
		* wcc = `Weighted Cross Correlation Loss <https://www.sciencedirect.com/science/article/pii/S0301010419313461>`_

	* **loss_args** [numeric or `null`]

		Some of the available loss functions can be parameterised. For example the `wcc` loss function requires the Gaussian half-width-half-maxiumum parameter to be passed. The default value for this loss function argument is `10` but the user can specify any parameters to be passed to the loss function here. See the linked PyTorch documentation above for loss function details. The value `null` can be specified if no arguments are required.

	* **loss_reg_type** [str or `null`]

		The user can choose to perform and control regularisation using the L1 or L2 norms. This technique can be used to prevent overfitting by adding penalty terms to the loss function during training. L1 regularisation or `Lasso <https://en.wikipedia.org/wiki/Lasso_(statistics)>`_ adds a penalty proportional to the absolute value of the model weights. It encourages sparsity and some weights may become set to zero as a result. L2 regularisation or `Ridge regression <https://en.wikipedia.org/wiki/Ridge_regression>`_ adds a penalty proportional to the square of the model weights. It discourages large weight values and aims to prevent the model becoming dominated by a few weights. This results in model weights closer to zero, although does not typically lead to zeroing of coefficients. User options are:

			* null (No regularisation)
			* L1
			* L2 

	* **loss_reg_params** [numeric]

		This parameter scales the strength of regularisation. A smaller value gives stronger regularisation, for example 0.001, and vice versa.

* **lr** [numeric]
	Learning rate for optimiser. This is the initial value for the optimiser which may change if the user has specified a learning rate scheduler elsewhere in the input YAML file.

* **optim_fn** [str]

	* `Adam <https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam>`_ (Default)
	* `SGD <https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD>`_
	* `RMSprop <https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html#torch.optim.RMSprop>`_

* **dropout** [numeric]

	During training model weights are randomly zeroed with the specified probability.

* **weight_init_seed** [int]

	For reproducibility, sets the initial seed for model weight initialisation.

* **kernel_init, bias_init** [str]

	Function for weight and bias initialisation.

	* `xavier_uniform <https://pytorch.org/docs/stable/nn.init.html?highlight=xavier#torch.nn.init.xavier_uniform_>`_ (default for `kernel_init`)
	* `uniform <https://pytorch.org/docs/stable/nn.init.html?highlight=xavier#torch.nn.init.uniform_>`_
	* `normal <https://pytorch.org/docs/stable/nn.init.html?highlight=xavier#torch.nn.init.normal_>`_
	* `xavier_normal <https://pytorch.org/docs/stable/nn.init.html?highlight=xavier#torch.nn.init.xavier_normal_>`_
	* `kaiming_uniform <https://pytorch.org/docs/stable/nn.init.html?highlight=xavier#torch.nn.init.kaiming_uniform_>`_
	* `zeros <https://pytorch.org/docs/stable/nn.init.html?highlight=xavier#torch.nn.init.zeros_>`_ (default for bias_init)
	* `ones <https://pytorch.org/docs/stable/nn.init.html?highlight=xavier#torch.nn.init.ones_>`_


-------------------------
Specific Model Parameters
-------------------------

The user now has the ability to train structure-to-spectrum and spectrum-to-structure
using 5 different model types:

.. toctree::

	model_mlp
	model_cnn
	model_ae_mlp
	model_ae_cnn
	model_aegan_mlp
	model_lstm
