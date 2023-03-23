# ###############################################################################
# ############################### LIBRARY IMPORTS ###############################
# ###############################################################################

from scipy.stats import ttest_ind

# from torch.utils.tensorboard import SummaryWriter


# # Tensorboard setup
# # layout = {
# #     "Multi": {
# #         "loss": ["Multiline", ["loss/train", "loss/validation"]],
# #     },
# # }
# # writer = SummaryWriter(f"/tmp/tensorboard/{int(time.time())}")
# # writer.add_custom_scalars(layout)

import torch
import numpy as np

def run_model_eval_tests(model, model_mode, trainloader, validloader, evalloader, n_in, out_dim):


	test_suite = ModelEvalTestSuite(model, model_mode, trainloader, validloader, evalloader, n_in, out_dim)
	test_results = test_suite.run_all()

	return test_results


def functional_mse(x,y):
	loss_fn = torch.nn.MSELoss(reduction="none")
	loss = np.sum(loss_fn(x,y).detach().numpy(), axis = 1)
	return loss

class ModelEvalTestSuite:
	def __init__(self, model, model_mode, trainloader, validloader, evalloader,n_in, out_dim):
		self.model = model
		self.model_mode = model_mode
		self.trainloader = trainloader
		self.validloader = validloader
		self.evalloader = evalloader
		self.n_in = n_in
		self.out_dim = out_dim

		self.model.eval()

		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


		# Get mean, sd for model input and output
		mean_input = torch.tensor([0] * self.n_in).to(self.device).float()
		mean_output = torch.tensor([0] * self.out_dim).to(self.device).float()

		std_input = torch.tensor([0] * self.n_in).to(self.device).float()
		std_output = torch.tensor([0] * self.out_dim).to(self.device).float()

		for x, y in self.trainloader:
			mean_input += x.mean([0])
			mean_output += y.mean([0])

		mean_input = mean_input/len(self.trainloader)
		mean_output = mean_output/len(self.trainloader)

		std_input = torch.sqrt(std_input/len(self.trainloader))
		std_output = torch.sqrt(std_output/len(self.trainloader))


		self.mean_input = mean_input.to(self.device).float()
		self.mean_output = mean_output.to(self.device).float()

		self.std_input = std_input.to(self.device).float()
		self.std_output = std_output.to(self.device).float()


	def run_all(self):
		print(f"{'='*20} Running Model Evaluation Tests {'='*20}")
	
		if self.model_mode == 'mlp' or model_mode == 'cnn':

			l0 = self.get_true_loss()

			li1 = self.get_loss_input_shuffle()
			lo1 = self.get_loss_output_shuffle()

			li2 = self.get_loss_input_mean_train()
			lo2 = self.get_loss_output_mean_train()

			li3 = self.get_loss_input_mean_sd_train()
			lo3 = self.get_loss_output_mean_sd_train()

			li4 = self.get_loss_input_random_valid()
			lo4 = self.get_loss_output_random_valid()

			test_results = {}
			
			test_results['Shuffle Input'] = loss_ttest(l0,li1)
			test_results['Shuffle Output'] = loss_ttest(l0,lo1)

			test_results['Mean Train Input'] = loss_ttest(l0,li2)
			test_results['Mean Train Output'] = loss_ttest(l0,lo2)

			test_results['Mean + Std. Train Input'] = loss_ttest(l0,li3)
			test_results['Mean + Std. Train Output'] = loss_ttest(l0,lo3)

			test_results['Random Valid Input'] = loss_ttest(l0,li4)
			test_results['Random Valid Output'] = loss_ttest(l0,lo4)

			for k, v in test_results.items():
				print(f">>> {k:25}: {v}")

		else:

			rl0, pl0 = self.get_true_loss()
			rli1, pli1 = self.get_loss_input_shuffle()
			rlo1, plo1 = self.get_loss_output_shuffle()

			pred_test_results = {}
			recon_test_results = {}

			pred_test_results['Shuffle Input'] = loss_ttest(pl0,pli1)
			pred_test_results['Shuffle Output'] = loss_ttest(pl0,plo1)

			recon_test_results['Shuffle Input'] = loss_ttest(rl0,rli1)
			recon_test_results['Shuffle Output'] = loss_ttest(rl0,rlo1)


			print(">>> Prediction:")
			for k, v in pred_test_results.items():
				print(f">>> {k:20}: {v}")


			print(">>> Reconstruction:")
			for k, v in recon_test_results.items():
				print(f">>> {k:20}: {v}")
			

		print(f"{'='*19} MLFlow: Evaluation Results Logged {'='*18}")
		return test_results


	def get_true_loss(self):

		if self.model_mode == 'mlp' or self.model_mode == 'cnn':
 
			true_loss = []

			for inputs, labels in self.evalloader:
				inputs, labels = inputs.to(self.device), labels.to(self.device)
				inputs, labels = inputs.float(), labels.float()
				target = self.model(inputs)
				loss = functional_mse(target, labels)
				true_loss.extend(loss)

			return true_loss

		else:

			print('Not implemented yet...')
			return None

	def get_loss_input_shuffle(self):

		if self.model_mode == 'mlp' or self.model_mode == 'cnn':

			other_loss = []

			for inputs, labels in self.evalloader:
				inputs, labels = inputs.to(self.device), labels.to(self.device)
				inputs, labels = inputs.float(), labels.float()

				idx = torch.randperm(inputs.shape[0])
				inputs = inputs[idx]

				target = self.model(inputs)
				
				loss = functional_mse(target, labels)
				other_loss.extend(loss)

			return other_loss

		else:

			print('Not implemented yet...')
			return None

	def get_loss_output_shuffle(self):

		if self.model_mode == 'mlp' or self.model_mode == 'cnn':

			other_loss = []

			for inputs, labels in self.evalloader:
				inputs, labels = inputs.to(self.device), labels.to(self.device)
				inputs, labels = inputs.float(), labels.float()

				idx = torch.randperm(labels.shape[0])
				labels = labels[idx]

				target = self.model(inputs)
				
				loss = functional_mse(target, labels)
				other_loss.extend(loss)

			return other_loss

		else:

			print('Not implemented yet...')
			return None

	def get_loss_input_mean_train(self):

		if self.model_mode == 'mlp' or self.model_mode == 'cnn':

			other_loss = []

			target_output = self.model(self.mean_input)

			for _, labels in self.evalloader:
				labels = labels.to(self.device)
				labels = labels.float()
				target = target_output.repeat(labels.shape[0], 1)
				loss = functional_mse(target, labels)
				other_loss.extend(loss)

			return other_loss

		else:

			print('Not implemented yet...')
			return None

	def get_loss_output_mean_train(self):

		if self.model_mode == 'mlp' or self.model_mode == 'cnn':

			other_loss = []

			target_output = self.mean_output

			for _, labels in self.evalloader:
				labels = labels.to(self.device)
				labels = labels.float()
				target = target_output.repeat(labels.shape[0], 1)
				loss = functional_mse(target, labels)
				other_loss.extend(loss)

			return other_loss

		else:

			print('Not implemented yet...')
			return None

	def get_loss_input_mean_sd_train(self):

		if self.model_mode == 'mlp' or self.model_mode == 'cnn':

			other_loss = []

			for _, labels in self.evalloader:
				labels = labels.to(self.device)
				labels = labels.float()

				mean_sd_input = self.mean_input.repeat(labels.shape[0], 1) + torch.normal(torch.zeros([labels.shape[0],self.n_in]),self.std_input)

				target = self.model(mean_sd_input)

				loss = functional_mse(target, labels)
				other_loss.extend(loss)

			return other_loss

		else:

			print('Not implemented yet...')
			return None

	def get_loss_output_mean_sd_train(self):

		if self.model_mode == 'mlp' or self.model_mode == 'cnn':

			other_loss = []

			for _, labels in self.evalloader:
				labels = labels.to(self.device)
				labels = labels.float()
			
				target = self.mean_output.repeat(labels.shape[0], 1) + torch.normal(torch.zeros([labels.shape[0],self.out_dim]),self.std_output)

				loss = functional_mse(target, labels)
				other_loss.extend(loss)

			return other_loss

		else:

			print('Not implemented yet...')
			return None


	def get_loss_input_random_valid(self):

		if self.model_mode == 'mlp' or self.model_mode == 'cnn':

			other_loss = []

			it = iter(self.validloader)

			for _, labels in self.evalloader:
				labels = labels.to(self.device)
				labels = labels.float()
				alt_inputs,_ = next(it)
				alt_inputs = alt_inputs.to(self.device).float()
				if labels.shape[0] < alt_inputs.shape[0]:
					alt_inputs = alt_inputs[labels.shape[0],:]

				target = self.model(alt_inputs)

				loss = functional_mse(target, labels)
				other_loss.extend(loss)

			return other_loss

		else:

			print('Not implemented yet...')
			return None

	def get_loss_output_random_valid(self):

		if self.model_mode == 'mlp' or self.model_mode == 'cnn':

			other_loss = []

			it = iter(self.validloader)

			for _, labels in self.evalloader:
				labels = labels.to(self.device)
				labels = labels.float()
				_, target = next(it)
				target = target.to(self.device).float()
				if labels.shape[0] < target.shape[0]:
					target = target[labels.shape[0],:]

				loss = functional_mse(target, labels)
				other_loss.extend(loss)

			return other_loss

		else:

			print('Not implemented yet...')
			return None


def loss_ttest(true_loss, other_loss, alpha=0.05):
	tstat, pval = ttest_ind(true_loss, other_loss, alternative="less")
	if pval < alpha:
		return True
	else:
		return False
