========================
Training Input YAML File
========================

User options in the training input YAML file specify the paths to training data, model hyperparameters, training hyperparameters and specify which additional training options are to be used.

* **Path to training data**

	.. code-block::
		
		x_path:  data/datasets/fe/preconv/xyz_train
		y_path:  data/datasets/fe/preconv/xanes_train

* **Descriptor function for structures**

	.. code-block::

		descriptor:
		    type: wacsf
		    params:
		      r_min: 1.0
		      r_max: 6.0
		      n_g2: 16
		      n_g4: 32

	There are five implemented descriptor functions. Each descriptor requires different parameters to be specified. Example parameter specification for each of the descriptor types can be found in :doc:`example_descriptor`. Options for descriptor type are:

	* `wacsf <https://pubs.aip.org/aip/jcp/article-abstract/148/24/241709/960046/wACSF-Weighted-atom-centered-symmetry-functions-as?redirectedFrom=fulltext>`_
	* `rdc <https://en.wikipedia.org/wiki/Radial_distribution_function>`_
	* `soap <https://singroup.github.io/dscribe/1.0.x/tutorials/descriptors/soap.html>`_
	* `mbtr <https://singroup.github.io/dscribe/0.3.x/tutorials/mbtr.html>`_
	* `lmbtr <https://singroup.github.io/dscribe/0.3.x/tutorials/lmbtr.html>`_

* **Load Guess**

	.. code-block::

		load_guess: False
		loadguess_params:
		  model_dir: ./model/model_001/model.pt
		  freeze_mlp_params:
		    freeze_dense: True
		    n_freeze_dense: 2


	These parameter specifications allow the user to load a pre-trained model, freeze a specified number of layers in the model, and continue training. A layer is *frozen* in the model by setting ``require_grad = False``. This prevents the layer weights being updated during backpropagation. Since each model type has a different architecture and layer type the user must specify these differently for each model. Examples are given in :doc:`example_load_guess`.

* **Bootstrap**

	Bootstrapping during training randomly samples the initial training data to create a new dataset that is the size of the dataset multiplied by ``n_size``. The model is then trained using this bootstrapped data and returned to the user. This is performed for the number of repeates specified by ``n_boot``. For reproducibility a different seed should be specified for each bootstrap iteration using ``seed_boot``. The bootstrap models are saved in the folder format ``bootstrap/bootstrap_001/model_001``, ``bootstrap/bootstrap_001/model_002`` etc.

	.. code-block::

		bootstrap: True 
		bootstrap_params: 
		  n_boot: 3
		  n_size: 1.0
		  seed_boot: [97, 39, 22]


* **Ensemble**

	Ensemble training uses different seeds provided through ``weight_init_seed`` to initialise the weights and biases of the model. The model is then trained for ``n_ens`` times using these different initialisation conditions and returned to the user. The ensemble models are saved in the folder format ``ensemble/ensemble_001/model_001``, ``ensemble/ensemble_001/model_002`` etc.

	.. code-block::

		ensemble: True
		ensemble_params:
		  n_ens: 3
		  weight_init_seed: [97, 39, 22]

* **Data Augmentation Parameters**

	There are two methods which the user can select from if they wish to augment the training data set. If ``random_noise`` type is selected the dataset is augmented by the multiplier ``augment_mult`` by sampling random data from the training set and adding samples from a Normal distribution in a pointwise manner. The parameters of the Normal distribution are specified using ``augment_params``. If ``random_combination`` type is selected the dataset is augmented by creating new data as the pointwise mean of two randomly sampled data from the training set. This data augmentation type does not require any parameters to be specified. Example inputs for each type are given:


	* augment_type = random_noise

	.. code-block::

		data_params: True
		augment:
		  augment_type: random_noise
		  augment_mult: 5
		  augment_params: 
		    normal_mean: 0
		    normal_sd: 0.1

	* augment_type = random_combination

	.. code-block::

		data_params: True
		augment:
		  augment_type: random_combination
		  augment_mult: 5
		  augment_params : null


* **K-fold Cross Validation**

	K-fold cross validation can be used to assess the performance of the model. The initial training data is split into *k* subsets of *folds*. Each single fold is used as the validation set and the remaining folds as the training set. The number of repeats controls how many times the initial dataset is re-partitioned and this process repeated for all folds. The user also has the option to specify the loss function used to evaluate each fold. For example, a certain loss function could be used to train the model during backpropagation but evaluated during k-fold cross validation using a different loss function. The loss function here is specified as above.

	.. code-block::

		kfold: True
		kfold_params:
		  n_splits: 5
		  n_repeats: 1
		  loss: 
		    loss_fn: mse 
		    loss_args: null
		    loss_reg_type: null
		    loss_reg_param: null

* **Learning Rate Scheduler**

	The initial learning rate can be adjusted based on the number of epochs throughout training according to the specified function type. Each scheduler type requires different parameterisation. Please see the linked PyTorch documentation for details. Options are:

	* `step <https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html#torch.optim.lr_scheduler.StepLR>`_
	* `multistep <https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.MultiStepLR.html#torch.optim.lr_scheduler.MultiStepLR>`_
	* `exponential <https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ExponentialLR.html#torch.optim.lr_scheduler.ExponentialLR>`_
	* `linear <https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LinearLR.html#torch.optim.lr_scheduler.LinearLR>`_
	* `constant <https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ConstantLR.html#torch.optim.lr_scheduler.ConstantLR>`_


	.. code-block::

		lr_scheduler: 
		  scheduler: True
		  scheduler_type: step
		  scheduler_param:
		    step_size: 100
		    gamma: 0.5


* **Model Hyperparameters**

	Full details on specifying model hyperparameters are given in :doc:`models`.

	.. code-block::

		hyperparams: 
			...


* **Training Parameters**

	General training parameters include the ``seed`` used to shuffle the training data and the number of ``epochs`` to train. Also specified is the boolean flag to perform model evaluation after training. Details on model evaluation are given in :doc:`model_evaluation`.

	.. code-block::

		seed: 2021
		epochs: 1
		model_eval : False

* **Hyperparameter Tuning Parameters**

	Users can perform tuning of model hyperparameter during training using `Optuna <https://optuna.readthedocs.io/en/stable/>`_. The user specifies how many tuning trials they wish to run and which hyperparameters they would like to vary during tuning. This will override any initially specified ``hyperparams``. Default options for each hyperparameter are specified with the ``optuna_defaults()`` function within ``src/optuna_learn.py`` and can easily be extended or modified to restrict or include other options. At the end of tuning, the optimal parameters are used to train the returned model.

	.. code-block::

		optuna_params:
		  tune: True
		  n_trials: 3
		  tune_optim_fn: True
		  tune_batch_size: True
		  tune_activation: True
		  tune_loss_fn: True
		  tune_lr: True
		  tune_dropout: True
		  tune_hidden_layers: True


.. toctree::
	:hidden:

	example_descriptor
	example_load_guess
	model_evaluation




