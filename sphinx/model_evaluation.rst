==========================
Evaluating a Trained Model
==========================

Model performance can be evaluated using some simple invariance testing by setting ``model_eval: True`` in the training input YAML file. The input data is split into three partitions; training (75%), validation (15%) and testing (10%). The training set is used to train the model, the validation set is used to report model performance during training. The test set is held out until the model is trained to perform model evaluation.

For a model to be useful its predictions should be closer to ground truth than to some artifically constructed target. During model evaluation four methods are implemented to create artificial target data and the distribution of losses is compared to the true test data losses. The artificial and true losses are compared using a one-sided T-Test at the 5% level. A True or False value is returned if the model performs better than a naive model which does not learn from the data and returns artifical output. Results should be treat as informative but not binding.

There are four types of tests implemented currently which make comparisons between a naive model and the trained model by manipulating either the input or output data. The four types include:

* Shuffling labels of the test data
* Using the mean of the training data
* Shuffling the labels of the training data
* Simulating as the mean of training data plus some Normally distributed noise with standard deviation matching the training data

The *shuffle input* test, for example, shuffles the input test data, passes it through the trained model and then calculates to the loss with the original unshuffled target data. This artifical loss is compared with the true loss if the input data was not shuffled. The *shuffle output* test, shuffles the target test data and passes the input data through the trained model. The artifical loss is calculated as the loss between the model prediction and the shuffled targets. This is compared with the true loss if the output data was not shuffled.

Results from these tests are reported at the command line interface as well as logged in MLFlow. Example output of a model that passes the evaluation tests successfully is printed to the console in the format,


.. code-block::

	==================== Running Model Evaluation Tests ====================
	>>> Shuffle Input            : True
	>>> Shuffle Output           : True
	>>> Mean Train Input         : True
	>>> Mean Train Output        : True
	>>> Mean Std. Train Input    : True
	>>> Mean Std. Train Output   : True
	>>> Random Valid Input       : True
	>>> Random Valid Output      : True
	=================== MLFlow: Evaluation Results Logged ==================



