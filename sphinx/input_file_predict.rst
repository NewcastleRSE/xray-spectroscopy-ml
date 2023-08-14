==========================
Prediction Input YAML File
==========================

User options in the prediction input YAML file specify the paths to input data and specify which additional options are to be used.


* **Path to input data**

	The user can specify either a single path or two paths. A single path will perform inference using the input data. Two paths will predict using the input data and return the error from the provided target data. For example,

	.. code-block::
		
		x_path:  data/datasets/fe/preconv/xyz_train
		y_path:  data/datasets/fe/preconv/xanes_train

	or 

	.. code-block::
		
		x_path:  data/datasets/fe/preconv/xyz_train
		y_path:  null

* **Plotting predictions**

	Option to plot the predictions for each input and save in the predictions output folder.

	.. code-block::

		plot_save: True

	or 

	.. code-block::

		plot_save: False

* **Monte Carlo Dropout**

	Typically dropout is turned off during prediction but this Monte Carlo option uses dropout to randomly deactivate some of the network nodes during prediction. Predictions are made for ``mc_iter`` number of iterations. The returned prediction is calculated as the mean of all predictions. The standard deviation across all predictions is also returned.

	.. code-block::

		monte_carlo: True
		mc_iter: 100


* **Predict from Bootstrap model**

	If the model has been trained using the bootstrap approach, this option allows predictions from each bootstrap model to be returned. Results are returned in the same output file for each bootstrap model.

	.. code-block::

		bootstrap: True

* **Predict from Ensemble model**

	.. code-block::

		ensemble: True
		ensemble_combine: weight


	If the model has been trained using the ensemble method this option combines predictions from the each ensemble according to the method specified by ``ensemble_combine``.  Options are,

	* ``weight`` Final prediction as mean of predictions from each of the ensemble models.
	* ``prediction`` Final prediction as mean of predictions from each of the ensemble models. Also returns the standard deviation across ensemble model predictions.

