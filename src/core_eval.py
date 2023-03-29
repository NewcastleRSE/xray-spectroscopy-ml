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


		self.mean_input = mean_input.to(self.device).float().view(1,self.n_in)
		self.mean_output = mean_output.to(self.device).float().view(1,self.out_dim)

		self.std_input = std_input.to(self.device).float()
		self.std_output = std_output.to(self.device).float()


	def run_all(self):
		print(f"{'='*20} Running Model Evaluation Tests {'='*20}")

		test_results = {}
	
		if self.model_mode == 'mlp' or self.model_mode == 'cnn':

			l0 = self.get_true_loss()

			li1 = self.get_loss_input_shuffle()
			lo1 = self.get_loss_output_shuffle()

			li2 = self.get_loss_input_mean_train()
			lo2 = self.get_loss_output_mean_train()

			li3 = self.get_loss_input_mean_sd_train()
			lo3 = self.get_loss_output_mean_sd_train()

			li4 = self.get_loss_input_random_valid()
			lo4 = self.get_loss_output_random_valid()
			
			test_results['Shuffle Input'] = loss_ttest(l0,li1)
			test_results['Shuffle Output'] = loss_ttest(l0,lo1)

			test_results['Mean Train Input'] = loss_ttest(l0,li2)
			test_results['Mean Train Output'] = loss_ttest(l0,lo2)

			test_results['Mean Std. Train Input'] = loss_ttest(l0,li3)
			test_results['Mean Std. Train Output'] = loss_ttest(l0,lo3)

			test_results['Random Valid Input'] = loss_ttest(l0,li4)
			test_results['Random Valid Output'] = loss_ttest(l0,lo4)

			for k, v in test_results.items():
				print(f">>> {k:25}: {v}")

			test_results = {
				'ModelEvalResults-Prediction' : test_results
				}

		elif self.model_mode == 'ae_mlp' or self.model_mode == 'ae_cnn':

			rl0, pl0 = self.get_true_loss()

			rli1, pli1 = self.get_loss_input_shuffle()
			rlo1, plo1 = self.get_loss_output_shuffle()

			rli2, pli2 = self.get_loss_input_mean_train()
			rlo2, plo2 = self.get_loss_output_mean_train()

			rli3, pli3 = self.get_loss_input_mean_sd_train()
			rlo3, plo3 = self.get_loss_output_mean_sd_train()

			rli4, pli4 = self.get_loss_input_random_valid()
			rlo4, plo4 = self.get_loss_output_random_valid()

			p_results = {}
			r_results = {}

			p_results['Shuffle Input'] = loss_ttest(pl0,pli1)
			p_results['Shuffle Output'] = loss_ttest(pl0,plo1)

			r_results['Shuffle Input'] = loss_ttest(rl0,rli1)
			r_results['Shuffle Output'] = loss_ttest(rl0,rlo1)

			p_results['Mean Train Input'] = loss_ttest(pl0,pli2)
			p_results['Mean Train Output'] = loss_ttest(pl0,plo2)

			r_results['Mean Train Input'] = loss_ttest(rl0,rli2)
			r_results['Mean Train Output'] = loss_ttest(rl0,rlo2)

			p_results['Mean Std. Train Input'] = loss_ttest(pl0,pli3)
			p_results['Mean Std. Train Output'] = loss_ttest(pl0,plo3)

			r_results['Mean Std. Train Input'] = loss_ttest(rl0,rli3)
			r_results['Mean Std. Train Output'] = loss_ttest(rl0,rlo3)

			p_results['Random Valid Input'] = loss_ttest(pl0,pli4)
			p_results['Random Valid Output'] = loss_ttest(pl0,plo4)

			r_results['Random Valid Input'] = loss_ttest(rl0,rli4)
			r_results['Random Valid Output'] = loss_ttest(rl0,rlo4)

			print("    Prediction:")
			for k, v in p_results.items():
				print(f">>> {k:25}: {v}")
			print('\n')


			print("    Reconstruction:")
			for k, v in r_results.items():
				print(f">>> {k:25}: {v}")

			test_results = {
				'ModelEvalResults-Reconstruction' : r_results,
				'ModelEvalResults-Prediction:' : p_results
				}

		elif self.model_mode == 'aegan_mlp':

			rxl0, ryl0, pxl0, pyl0 = self.get_true_loss()

			rxli1, ryli1, pxli1, pyli1 = self.get_loss_input_shuffle()
			rxlo1, rylo1, pxlo1, pylo1 = self.get_loss_output_shuffle()

			rxli2, ryli2, pxli2, pyli2 = self.get_loss_input_mean_train()
			rxlo2, rylo2, pxlo2, pylo2 = self.get_loss_output_mean_train()

			rxli3, ryli3, pxli3, pyli3 = self.get_loss_input_mean_sd_train()
			rxlo3, rylo3, pxlo3, pylo3 = self.get_loss_output_mean_sd_train()

			rxli4, ryli4, pxli4, pyli4 = self.get_loss_input_random_valid()
			rxlo4, rylo4, pxlo4, pylo4 = self.get_loss_output_random_valid()

			p_x_results = {}
			p_y_results = {}
			r_x_results = {}
			r_y_results = {}

			# Prediction XYZ
			p_x_results['Shuffle Input'] = loss_ttest(pxl0,pxli1)
			p_x_results['Shuffle Output'] = loss_ttest(pxl0,pxlo1)

			p_x_results['Mean Train Input'] = loss_ttest(pxl0,pxli2)
			p_x_results['Mean Train Output'] = loss_ttest(pxl0,pxlo2)

			p_x_results['Mean Std. Train Input'] = loss_ttest(pxl0,pxli3)
			p_x_results['Mean Std. Train Output'] = loss_ttest(pxl0,pxlo3)

			p_x_results['Random Valid Input'] = loss_ttest(pxl0,pxli4)
			p_x_results['Random Valid Output'] = loss_ttest(pxl0,pxlo4)

			# Prediction Xanes
			p_y_results['Shuffle Input'] = loss_ttest(pyl0,pyli1)
			p_y_results['Shuffle Output'] = loss_ttest(pyl0,pylo1)

			p_y_results['Mean Train Input'] = loss_ttest(pyl0,pyli2)
			p_y_results['Mean Train Output'] = loss_ttest(pyl0,pylo2)

			p_y_results['Mean Std. Train Input'] = loss_ttest(pyl0,pyli3)
			p_y_results['Mean Std. Train Output'] = loss_ttest(pyl0,pylo3)

			p_y_results['Random Valid Input'] = loss_ttest(pyl0,pyli4)
			p_y_results['Random Valid Output'] = loss_ttest(pyl0,pylo4)

			# Reconstruction XYZ
			r_x_results['Shuffle Input'] = loss_ttest(rxl0,rxli1)
			r_x_results['Shuffle Output'] = loss_ttest(rxl0,rxlo1)

			r_x_results['Mean Train Input'] = loss_ttest(rxl0,rxli2)
			r_x_results['Mean Train Output'] = loss_ttest(rxl0,rxlo2)

			r_x_results['Mean Std. Train Input'] = loss_ttest(rxl0,rxli3)
			r_x_results['Mean Std. Train Output'] = loss_ttest(rxl0,rxlo3)

			r_x_results['Random Valid Input'] = loss_ttest(rxl0,rxli4)
			r_x_results['Random Valid Output'] = loss_ttest(rxl0,rxlo4)

			# Reconstruction Xanes
			r_y_results['Shuffle Input'] = loss_ttest(ryl0,ryli1)
			r_y_results['Shuffle Output'] = loss_ttest(ryl0,rylo1)

			r_y_results['Mean Train Input'] = loss_ttest(ryl0,ryli2)
			r_y_results['Mean Train Output'] = loss_ttest(ryl0,rylo2)

			r_y_results['Mean Std. Train Input'] = loss_ttest(ryl0,ryli3)
			r_y_results['Mean Std. Train Output'] = loss_ttest(ryl0,rylo3)

			r_y_results['Random Valid Input'] = loss_ttest(ryl0,ryli4)
			r_y_results['Random Valid Output'] = loss_ttest(ryl0,rylo4)

			print("    Prediction XYZ:")
			for k, v in p_x_results.items():
				print(f">>> {k:25}: {v}")
			print('\n')

			print("    Prediction Xanes:")
			for k, v in p_y_results.items():
				print(f">>> {k:25}: {v}")
			print('\n')

			print("    Reconstruction XYZ:")
			for k, v in r_x_results.items():
				print(f">>> {k:25}: {v}")
			print('\n')

			print("    Reconstruction Xanes:")
			for k, v in r_y_results.items():
				print(f">>> {k:25}: {v}")
			print('\n')

			test_results = {
				'ModelEvalResults-Reconstruction-XYZ' : r_x_results,
				'ModelEvalResults-Reconstruction-Xanes' : r_y_results,
				'ModelEvalResults-Prediction-XYZ:' : p_x_results,
				'ModelEvalResults-Prediction-Xanes' : p_y_results
				}

		else:

			test_results = None


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

		elif self.model_mode == 'ae_mlp' or self.model_mode == 'ae_cnn':

			true_recon_loss = []
			true_pred_loss = []

			for inputs, labels in self.evalloader:
				inputs, labels = inputs.to(self.device), labels.to(self.device)
				inputs, labels = inputs.float(), labels.float()

				recon_target, pred_target = self.model(inputs)

				recon_loss = functional_mse(recon_target, inputs)
				pred_loss = functional_mse(pred_target, labels)

				true_recon_loss.extend(recon_loss)
				true_pred_loss.extend(pred_loss)

			return true_recon_loss, true_pred_loss

		elif self.model_mode == 'aegan_mlp':

			true_recon_x_loss = []
			true_recon_y_loss = []
			true_pred_x_loss = []
			true_pred_y_loss = []

			for inputs, labels in self.evalloader:
				inputs, labels = inputs.to(self.device), labels.to(self.device)
				inputs, labels = inputs.float(), labels.float()

				recon_x, recon_y, pred_x, pred_y = self.model.reconstruct_all_predict_all(
                    inputs, labels
                )

				recon_x_loss = functional_mse(recon_x, inputs)
				recon_y_loss = functional_mse(recon_y, labels)
				pred_x_loss = functional_mse(pred_x, inputs)
				pred_y_loss = functional_mse(pred_y, labels)

				true_recon_x_loss.extend(recon_x_loss)
				true_recon_y_loss.extend(recon_y_loss)
				true_pred_x_loss.extend(pred_x_loss)
				true_pred_y_loss.extend(pred_y_loss)

			return true_recon_x_loss, true_recon_y_loss, true_pred_x_loss, true_pred_y_loss

		else:

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

		elif self.model_mode == 'ae_mlp' or self.model_mode == 'ae_cnn':
			
			other_recon_loss = []
			other_pred_loss = []

			for inputs, labels in self.evalloader:
				inputs, labels = inputs.to(self.device), labels.to(self.device)
				inputs, labels = inputs.float(), labels.float()

				idx = torch.randperm(inputs.shape[0])
				inputs_shuffle = inputs[idx]

				recon_target, pred_target = self.model(inputs_shuffle)

				recon_loss = functional_mse(recon_target, inputs)
				pred_loss = functional_mse(pred_target, labels)

				other_recon_loss.extend(recon_loss)
				other_pred_loss.extend(pred_loss)

			return other_recon_loss, other_pred_loss

		elif self.model_mode == 'aegan_mlp':

			other_recon_x_loss = []
			other_recon_y_loss = []
			other_pred_x_loss = []
			other_pred_y_loss = []

			for inputs, labels in self.evalloader:
				inputs, labels = inputs.to(self.device), labels.to(self.device)
				inputs, labels = inputs.float(), labels.float()

				idx = torch.randperm(inputs.shape[0])
				inputs_shuffle = inputs[idx]

				jdx = torch.randperm(labels.shape[0])
				labels_shuffle = labels[jdx]

				recon_x, recon_y, pred_x, pred_y = self.model.reconstruct_all_predict_all(
                    inputs_shuffle, labels_shuffle
                )

				recon_x_loss = functional_mse(recon_x, inputs)
				recon_y_loss = functional_mse(recon_y, labels)
				pred_x_loss = functional_mse(pred_x, inputs)
				pred_y_loss = functional_mse(pred_y, labels)

				other_recon_x_loss.extend(recon_x_loss)
				other_recon_y_loss.extend(recon_y_loss)
				other_pred_x_loss.extend(pred_x_loss)
				other_pred_y_loss.extend(pred_y_loss)

			return other_recon_x_loss, other_recon_y_loss, other_pred_x_loss, other_pred_y_loss

		else:

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

		elif self.model_mode == 'ae_mlp' or self.model_mode == 'ae_cnn':

			other_recon_loss = []
			other_pred_loss = []

			for inputs, labels in self.evalloader:
				inputs, labels = inputs.to(self.device), labels.to(self.device)
				inputs, labels = inputs.float(), labels.float()

				idx = torch.randperm(labels.shape[0])
				labels_shuffle = labels[idx]

				recon_target, pred_target = self.model(inputs)

				jdx = torch.randperm(recon_target.shape[0])
				recon_target_shuffle = recon_target[jdx]

				recon_loss = functional_mse(recon_target_shuffle, inputs)
				pred_loss = functional_mse(pred_target, labels_shuffle)

				other_recon_loss.extend(recon_loss)
				other_pred_loss.extend(pred_loss)

			return other_recon_loss, other_pred_loss

		elif self.model_mode == 'aegan_mlp':

			other_recon_x_loss = []
			other_recon_y_loss = []
			other_pred_x_loss = []
			other_pred_y_loss = []

			for inputs, labels in self.evalloader:
				inputs, labels = inputs.to(self.device), labels.to(self.device)
				inputs, labels = inputs.float(), labels.float()

				idx = torch.randperm(inputs.shape[0])
				inputs_shuffle = inputs[idx]

				jdx = torch.randperm(labels.shape[0])
				labels_shuffle = labels[jdx]

				recon_x, recon_y, pred_x, pred_y = self.model.reconstruct_all_predict_all(
                    inputs, labels
                )

				recon_x_loss = functional_mse(recon_x, inputs_shuffle)
				recon_y_loss = functional_mse(recon_y, labels_shuffle)
				pred_x_loss = functional_mse(pred_x, inputs_shuffle)
				pred_y_loss = functional_mse(pred_y, labels_shuffle)

				other_recon_x_loss.extend(recon_x_loss)
				other_recon_y_loss.extend(recon_y_loss)
				other_pred_x_loss.extend(pred_x_loss)
				other_pred_y_loss.extend(pred_y_loss)

			return other_recon_x_loss, other_recon_y_loss, other_pred_x_loss, other_pred_y_loss		

		else:

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

		elif self.model_mode == 'ae_mlp' or self.model_mode == 'ae_cnn':
			
			other_recon_loss = []
			other_pred_loss = []

			recon_target_val, pred_target_val = self.model(self.mean_input)

			for inputs, labels in self.evalloader:
				inputs, labels = inputs.to(self.device), labels.to(self.device)
				inputs, labels = inputs.float(), labels.float()

				recon_target = recon_target_val.repeat(labels.shape[0], 1)
				pred_target = pred_target_val.repeat(labels.shape[0], 1)

				recon_loss = functional_mse(recon_target, inputs)
				pred_loss = functional_mse(pred_target, labels)

				other_recon_loss.extend(recon_loss)
				other_pred_loss.extend(pred_loss)

			return other_recon_loss, other_pred_loss

		elif self.model_mode == 'aegan_mlp':

			other_recon_x_loss = []
			other_recon_y_loss = []
			other_pred_x_loss = []
			other_pred_y_loss = []

			recon_x_val, recon_y_val, pred_x_val, pred_y_val = self.model.reconstruct_all_predict_all(
				self.mean_input, self.mean_output)

			for inputs, labels in self.evalloader:
				inputs, labels = inputs.to(self.device), labels.to(self.device)
				inputs, labels = inputs.float(), labels.float()

				recon_x = recon_x_val.repeat(labels.shape[0], 1)
				recon_y = recon_y_val.repeat(labels.shape[0], 1)
				pred_x = pred_x_val.repeat(labels.shape[0], 1)
				pred_y = pred_y_val.repeat(labels.shape[0], 1)

				recon_x_loss = functional_mse(recon_x, inputs)
				recon_y_loss = functional_mse(recon_y, labels)
				pred_x_loss = functional_mse(pred_x, inputs)
				pred_y_loss = functional_mse(pred_y, labels)

				other_recon_x_loss.extend(recon_x_loss)
				other_recon_y_loss.extend(recon_y_loss)
				other_pred_x_loss.extend(pred_x_loss)
				other_pred_y_loss.extend(pred_y_loss)

			return other_recon_x_loss, other_recon_y_loss, other_pred_x_loss, other_pred_y_loss

		else:

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

		elif self.model_mode == 'ae_mlp' or self.model_mode == 'ae_cnn':
			
			other_recon_loss = []
			other_pred_loss = []

			recon_target_val = self.mean_input
			pred_target_val = self.mean_output

			for inputs, labels in self.evalloader:
				inputs, labels = inputs.to(self.device), labels.to(self.device)
				inputs, labels = inputs.float(), labels.float()

				recon_target = recon_target_val.repeat(labels.shape[0], 1)
				pred_target = pred_target_val.repeat(labels.shape[0], 1)

				recon_loss = functional_mse(recon_target, inputs)
				pred_loss = functional_mse(pred_target, labels)

				other_recon_loss.extend(recon_loss)
				other_pred_loss.extend(pred_loss)

			return other_recon_loss, other_pred_loss

		elif self.model_mode == 'aegan_mlp':

			other_recon_x_loss = []
			other_recon_y_loss = []
			other_pred_x_loss = []
			other_pred_y_loss = []

			recon_x_val = pred_x_val = self.mean_input
			recon_y_val = pred_y_val = self.mean_output

			for inputs, labels in self.evalloader:
				inputs, labels = inputs.to(self.device), labels.to(self.device)
				inputs, labels = inputs.float(), labels.float()

				recon_x = recon_x_val.repeat(labels.shape[0], 1)
				recon_y = recon_y_val.repeat(labels.shape[0], 1)
				pred_x = pred_x_val.repeat(labels.shape[0], 1)
				pred_y = pred_y_val.repeat(labels.shape[0], 1)

				recon_x_loss = functional_mse(recon_x, inputs)
				recon_y_loss = functional_mse(recon_y, labels)
				pred_x_loss = functional_mse(pred_x, inputs)
				pred_y_loss = functional_mse(pred_y, labels)

				other_recon_x_loss.extend(recon_x_loss)
				other_recon_y_loss.extend(recon_y_loss)
				other_pred_x_loss.extend(pred_x_loss)
				other_pred_y_loss.extend(pred_y_loss)

			return other_recon_x_loss, other_recon_y_loss, other_pred_x_loss, other_pred_y_loss	

		else:

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

		elif self.model_mode == 'ae_mlp' or self.model_mode == 'ae_cnn':
			
			other_recon_loss = []
			other_pred_loss = []

			for inputs, labels in self.evalloader:
				inputs, labels = inputs.to(self.device), labels.to(self.device)
				inputs, labels = inputs.float(), labels.float()

				mean_sd_input = self.mean_input.repeat(labels.shape[0], 1) + torch.normal(torch.zeros([labels.shape[0],self.n_in]),self.std_input)

				recon_target, pred_target = self.model(mean_sd_input)

				recon_loss = functional_mse(recon_target, inputs)
				pred_loss = functional_mse(pred_target, labels)

				other_recon_loss.extend(recon_loss)
				other_pred_loss.extend(pred_loss)

			return other_recon_loss, other_pred_loss

		elif self.model_mode == 'aegan_mlp':

			other_recon_x_loss = []
			other_recon_y_loss = []
			other_pred_x_loss = []
			other_pred_y_loss = []

			for inputs, labels in self.evalloader:
				inputs, labels = inputs.to(self.device), labels.to(self.device)
				inputs, labels = inputs.float(), labels.float()

				mean_sd_input = self.mean_input.repeat(labels.shape[0], 1) + torch.normal(torch.zeros([labels.shape[0],self.n_in]),self.std_input)
				mean_sd_output = self.mean_output.repeat(labels.shape[0], 1) + torch.normal(torch.zeros([labels.shape[0],self.out_dim]),self.std_output)

				recon_x, recon_y, pred_x, pred_y = self.model.reconstruct_all_predict_all(
					mean_sd_input, mean_sd_output)

				recon_x_loss = functional_mse(recon_x, inputs)
				recon_y_loss = functional_mse(recon_y, labels)
				pred_x_loss = functional_mse(pred_x, inputs)
				pred_y_loss = functional_mse(pred_y, labels)

				other_recon_x_loss.extend(recon_x_loss)
				other_recon_y_loss.extend(recon_y_loss)
				other_pred_x_loss.extend(pred_x_loss)
				other_pred_y_loss.extend(pred_y_loss)

			return other_recon_x_loss, other_recon_y_loss, other_pred_x_loss, other_pred_y_loss

		else:

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

		elif self.model_mode == 'ae_mlp' or self.model_mode == 'ae_cnn':
			
			other_recon_loss = []
			other_pred_loss = []

			for inputs, labels in self.evalloader:
				inputs, labels = inputs.to(self.device), labels.to(self.device)
				inputs, labels = inputs.float(), labels.float()

				recon_target = self.mean_input.repeat(labels.shape[0], 1) + torch.normal(torch.zeros([labels.shape[0],self.n_in]),self.std_input)
				pred_target = self.mean_output.repeat(labels.shape[0], 1) + torch.normal(torch.zeros([labels.shape[0],self.out_dim]),self.std_output)

				recon_loss = functional_mse(recon_target, inputs)
				pred_loss = functional_mse(pred_target, labels)

				other_recon_loss.extend(recon_loss)
				other_pred_loss.extend(pred_loss)

			return other_recon_loss, other_pred_loss

		elif self.model_mode == 'aegan_mlp':

			other_recon_x_loss = []
			other_recon_y_loss = []
			other_pred_x_loss = []
			other_pred_y_loss = []

			for inputs, labels in self.evalloader:
				inputs, labels = inputs.to(self.device), labels.to(self.device)
				inputs, labels = inputs.float(), labels.float()

				recon_x = self.mean_input.repeat(labels.shape[0], 1) + torch.normal(torch.zeros([labels.shape[0],self.n_in]),self.std_input)
				recon_y = self.mean_output.repeat(labels.shape[0], 1) + torch.normal(torch.zeros([labels.shape[0],self.out_dim]),self.std_output)
				pred_x = self.mean_input.repeat(labels.shape[0], 1) + torch.normal(torch.zeros([labels.shape[0],self.n_in]),self.std_input)
				pred_y = self.mean_output.repeat(labels.shape[0], 1) + torch.normal(torch.zeros([labels.shape[0],self.out_dim]),self.std_output)

				recon_x_loss = functional_mse(recon_x, inputs)
				recon_y_loss = functional_mse(recon_y, labels)
				pred_x_loss = functional_mse(pred_x, inputs)
				pred_y_loss = functional_mse(pred_y, labels)

				other_recon_x_loss.extend(recon_x_loss)
				other_recon_y_loss.extend(recon_y_loss)
				other_pred_x_loss.extend(pred_x_loss)
				other_pred_y_loss.extend(pred_y_loss)

			return other_recon_x_loss, other_recon_y_loss, other_pred_x_loss, other_pred_y_loss	

		else:

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
					alt_inputs = alt_inputs[:labels.shape[0],:]

				target = self.model(alt_inputs)

				loss = functional_mse(target, labels)
				other_loss.extend(loss)

			return other_loss

		elif self.model_mode == 'ae_mlp' or self.model_mode == 'ae_cnn':
			
			other_recon_loss = []
			other_pred_loss = []

			it = iter(self.validloader)

			for inputs, labels in self.evalloader:
				inputs, labels = inputs.to(self.device), labels.to(self.device)
				inputs, labels = inputs.float(), labels.float()

				alt_inputs, alt_labels = next(it)

				alt_inputs = alt_inputs.to(self.device).float()
				alt_labels = alt_labels.to(self.device).float()

				if labels.shape[0] < alt_inputs.shape[0]:
					alt_inputs = alt_inputs[:labels.shape[0],:]
					alt_labels = alt_labels[:labels.shape[0],:]

				recon_target, pred_target = self.model(alt_inputs)

				recon_loss = functional_mse(recon_target, inputs)
				pred_loss = functional_mse(pred_target, labels)

				other_recon_loss.extend(recon_loss)
				other_pred_loss.extend(pred_loss)

			return other_recon_loss, other_pred_loss

		elif self.model_mode == 'aegan_mlp':

			other_recon_x_loss = []
			other_recon_y_loss = []
			other_pred_x_loss = []
			other_pred_y_loss = []

			it = iter(self.validloader)

			for inputs, labels in self.evalloader:
				inputs, labels = inputs.to(self.device), labels.to(self.device)
				inputs, labels = inputs.float(), labels.float()

				alt_inputs, alt_labels = next(it)

				alt_inputs = alt_inputs.to(self.device).float()
				alt_labels = alt_labels.to(self.device).float()

				if labels.shape[0] < alt_inputs.shape[0]:
					alt_inputs = alt_inputs[:labels.shape[0],:]
					alt_labels = alt_labels[:labels.shape[0],:]

				recon_x, recon_y, pred_x, pred_y = self.model.reconstruct_all_predict_all(
					alt_inputs, alt_labels)

				recon_x_loss = functional_mse(recon_x, inputs)
				recon_y_loss = functional_mse(recon_y, labels)
				pred_x_loss = functional_mse(pred_x, inputs)
				pred_y_loss = functional_mse(pred_y, labels)

				other_recon_x_loss.extend(recon_x_loss)
				other_recon_y_loss.extend(recon_y_loss)
				other_pred_x_loss.extend(pred_x_loss)
				other_pred_y_loss.extend(pred_y_loss)

			return other_recon_x_loss, other_recon_y_loss, other_pred_x_loss, other_pred_y_loss

		else:

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
					target = target[:labels.shape[0],:]

				loss = functional_mse(target, labels)
				other_loss.extend(loss)

			return other_loss

		elif self.model_mode == 'ae_mlp' or self.model_mode == 'ae_cnn':
			
			other_recon_loss = []
			other_pred_loss = []

			it = iter(self.validloader)

			for inputs, labels in self.evalloader:
				inputs, labels = inputs.to(self.device), labels.to(self.device)
				inputs, labels = inputs.float(), labels.float()

				alt_inputs, alt_labels = next(it)

				alt_inputs = alt_inputs.to(self.device).float()
				alt_labels = alt_labels.to(self.device).float()

				if labels.shape[0] < alt_inputs.shape[0]:
					alt_inputs = alt_inputs[:labels.shape[0],:]
					alt_labels = alt_labels[:labels.shape[0],:]

				recon_target, pred_target = alt_inputs, alt_labels

				recon_loss = functional_mse(recon_target, inputs)
				pred_loss = functional_mse(pred_target, labels)

				other_recon_loss.extend(recon_loss)
				other_pred_loss.extend(pred_loss)

			return other_recon_loss, other_pred_loss

		elif self.model_mode == 'aegan_mlp':

			other_recon_x_loss = []
			other_recon_y_loss = []
			other_pred_x_loss = []
			other_pred_y_loss = []

			it = iter(self.validloader)

			for inputs, labels in self.evalloader:
				inputs, labels = inputs.to(self.device), labels.to(self.device)
				inputs, labels = inputs.float(), labels.float()

				alt_inputs, alt_labels = next(it)

				alt_inputs = alt_inputs.to(self.device).float()
				alt_labels = alt_labels.to(self.device).float()

				if labels.shape[0] < alt_inputs.shape[0]:
					alt_inputs = alt_inputs[:labels.shape[0],:]
					alt_labels = alt_labels[:labels.shape[0],:]

				recon_x = alt_inputs
				recon_y = alt_labels

				idx = torch.randperm(labels.shape[0])
				jdx = torch.randperm(labels.shape[0])
				pred_x = alt_inputs[idx]
				pred_y = alt_labels[jdx]

				recon_x_loss = functional_mse(recon_x, inputs)
				recon_y_loss = functional_mse(recon_y, labels)
				pred_x_loss = functional_mse(pred_x, inputs)
				pred_y_loss = functional_mse(pred_y, labels)

				other_recon_x_loss.extend(recon_x_loss)
				other_recon_y_loss.extend(recon_y_loss)
				other_pred_x_loss.extend(pred_x_loss)
				other_pred_y_loss.extend(pred_y_loss)

			return other_recon_x_loss, other_recon_y_loss, other_pred_x_loss, other_pred_y_loss

		else:

			return None


def loss_ttest(true_loss, other_loss, alpha=0.05):
	tstat, pval = ttest_ind(true_loss, other_loss, alternative="less")
	if pval < alpha:
		return True
	else:
		return False
