==================================
Load Guess Layer Freezing Examples
==================================



* **mlp**

	.. code-block::

		load_guess: False
		loadguess_params:
		  model_dir: ./model/model_001/model.pt
		  freeze_mlp_params:
		    freeze_dense: True
		    n_freeze_dense: 2


* **cnn**

	.. code-block::

		load_guess: False
		loadguess_params:
		  model_dir: ./model/model_001/model.pt
		  freeze_cnn_params:
		    freeze_conv: True
		    freeze_dense: True
		    n_freeze_conv: 4
		    n_freeze_dense: 1


* **ae_mlp**

	.. code-block::

		load_guess: False
		loadguess_params:
		  model_dir: ./model/model_001/model.pt
		  freeze_ae_mlp_params:
		    freeze_encoder: True
		    freeze_decoder: True
		    freeze_dense: True
		    n_freeze_encoder: 2
		    n_freeze_decoder: 2
		    n_freeze_dense: 1


* **ae_cnn**

	.. code-block::

		load_guess: False
		loadguess_params:
		  model_dir: ./model/model_001/model.pt
		  freeze_ae_cnn_params:
		    freeze_encoder: True
		    freeze_decoder: True
		    freeze_dense: True
		    n_freeze_encoder: 2
		    n_freeze_decoder: 2
		    n_freeze_dense: 1


* **aegan_mlp**

	.. code-block::

		load_guess: False
		loadguess_params:
		  model_dir: ./model/model_001/model.pt
		  freeze_params:
		    freeze_encoder1: True
		    freeze_encoder2: True
		    freeze_decoder1: True
		    freeze_decoder2: True
		    freeze_shared_encoder: True
		    freeze_shared_decoder: True
		    freeze_discrim1: True
		    freeze_discrim2: True
		    n_freeze_encoder1: 1
		    n_freeze_encoder2: 1
		    n_freeze_decoder1: 1
		    n_freeze_decoder2: 1
		    n_freeze_shared_encoder: 1
		    n_freeze_shared_decoder: 1
		    n_freeze_discrim1: 1
		    n_freeze_discrim2: 1


* **lstm**

	.. code-block::

		load_guess: False
		loadguess_params:
		  model_dir: ./model/model_001/model.pt
		  freeze_lstm_params:
		    freeze_lstm: True
		    freeze_dense: False
		    n_freeze_lstm: 1
		    n_freeze_dense: 1