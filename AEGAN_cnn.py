import torch
from torch import nn, optim

from torch import nn
import torch
from torch.autograd import Variable
import numpy as np
import math


class AEGAN_CNN(nn.Module):
	def __init__(self, **params):
		super().__init__()

		self.input_size_a = params["dim_a"]
		self.input_size_b = params["dim_b"]
		self.hidden_size = params["hidden_size"]
		self.dropout = params["dropout"]

		self.n_hl_gen = params["n_hl_gen"]
		self.n_hl_shared = params["n_hl_shared"]
		self.n_hl_dis = params["n_hl_dis"]

		# Select activation function
		activation_switch = ActivationSwitch()
		self.activation = activation_switch.fn(params["activation"])

		# Select loss functions
		loss_switch = LossSwitch()
		self.loss_fn_gen = loss_switch.fn(params["loss_gen"])
		self.loss_fn_dis = loss_switch.fn(params["loss_dis"])

		# Convolutional parameters
		self.in_channel = 1
		self.out_channel = params["out_channel"]
		self.channel_mul = params["channel_mul"]
		self.kernel_size = params["kernel_size"]
		self.stride = params["stride"]

		# Select pooling layer function
		self.include_pooling = params["include_pooling"]
		self.pool_fn = params["pool_fn"]
		pooling_switch = PoolingSwitch(kernel_size=self.kernel_size, stride=self.stride)
		self.pooling_layer = pooling_switch.fn(self.pool_fn)

		# Get max convolutional blocks
		self.num_conv_blocks = params["num_conv_blocks"]
		self.gen_a_max_conv_blocks = self.get_max_conv_blocks(
			in_length=self.input_size_a
		)
		self.gen_b_max_conv_blocks = self.get_max_conv_blocks(
			in_length=self.input_size_b
		)

		self.gen_a_num_conv_blocks = (
			self.num_conv_blocks
			if self.num_conv_blocks <= self.gen_a_max_conv_blocks
			else self.gen_a_max_conv_blocks
		)
		self.gen_b_num_conv_blocks = (
			self.num_conv_blocks
			if self.num_conv_blocks <= self.gen_b_max_conv_blocks
			else self.gen_b_max_conv_blocks
		)

		self.gen_a_encoder_conv_output_size = self.get_conv_blocks_output_size(
			in_length=self.input_size_a, num_conv_blocks=self.gen_a_num_conv_blocks
		)
		self.gen_b_encoder_conv_output_size = self.get_conv_blocks_output_size(
			in_length=self.input_size_b, num_conv_blocks=self.gen_b_num_conv_blocks
		)

		self.gen_a_decoder_conv_output_size = self.get_dec_blocks_output_size(
			num_conv_blocks=self.gen_a_num_conv_blocks
		)
		self.gen_b_decoder_conv_output_size = self.get_dec_blocks_output_size(
			num_conv_blocks=self.gen_b_num_conv_blocks
		)

		# ---------------------------------------- Initialise generative networks ---------------------------------------- #
		# Autoencoder for domain A
		self.gen_a = AEGenCNN(
			input_size=self.input_size_a,
			num_hidden_layer=self.n_hl_gen,
			hidden_size=self.hidden_size,
			dropout=self.dropout,
			activation=self.activation,
			out_channel=self.out_channel,
			channel_mul=self.channel_mul,
			kernel_size=self.kernel_size,
			stride=self.stride,
			pooling_layer=self.pooling_layer,
			num_conv_blocks=self.gen_a_num_conv_blocks,
			enc_conv_block_output_size=self.gen_a_encoder_conv_output_size,
			enc_shared_hidden_size=self.hidden_size,
			dec_conv_block_output_size=self.gen_a_decoder_conv_output_size,
		)
		# Autoencoder for domain B
		self.gen_b = AEGenCNN(
			input_size=self.input_size_b,
			num_hidden_layer=self.n_hl_gen,
			hidden_size=self.hidden_size,
			dropout=self.dropout,
			activation=self.activation,
			out_channel=self.out_channel,
			channel_mul=self.channel_mul,
			kernel_size=self.kernel_size,
			stride=self.stride,
			pooling_layer=self.pooling_layer,
			num_conv_blocks=self.gen_b_num_conv_blocks,
			enc_conv_block_output_size=self.gen_b_encoder_conv_output_size,
			enc_shared_hidden_size=self.hidden_size,
			dec_conv_block_output_size=self.gen_b_decoder_conv_output_size,
		)

		# Shared autoencoder layers
		self.enc_shared = SharedLayer(
			num_hidden_layer=self.n_hl_shared,
			hidden_size=self.hidden_size,
			dropout=self.dropout,
			activation=self.activation,
		)
		self.dec_shared = SharedLayer(
			num_hidden_layer=self.n_hl_shared,
			hidden_size=self.hidden_size,
			dropout=self.dropout,
			activation=self.activation,
		)

		# ---------------------------------------- Initialise discriminator networks ---------------------------------------- #
		self.dis_a = Dis(
			input_size=self.input_size_a,
			num_hidden_layer=self.n_hl_dis,
			hidden_size=self.hidden_size,
			dropout=self.dropout,
			activation=self.activation,
			loss_fn=self.loss_fn_dis,
		)  # discriminator for domain a
		self.dis_b = Dis(
			input_size=self.input_size_b,
			num_hidden_layer=self.n_hl_dis,
			hidden_size=self.hidden_size,
			dropout=self.dropout,
			activation=self.activation,
			loss_fn=self.loss_fn_dis,
		)  # discriminator for domain b

		# Learning rate
		self.lr_gen = params["lr_gen"]
		self.lr_dis = params["lr_dis"]

		params_gen = [
			param for name, param in self.named_parameters() if "dis" not in name
		]
		params_dis = [param for name, param in self.named_parameters() if "dis" in name]

		# Optim for generators and discriminators
		self.gen_opt = torch.optim.Adam(params_gen, lr=self.lr_gen, weight_decay=1e-5)
		self.dis_opt = torch.optim.Adam(params_dis, lr=self.lr_dis, weight_decay=1e-5)

	def get_max_conv_blocks(self, in_length):
		kernel_size = self.kernel_size
		stride = self.stride
		include_pooling = self.include_pooling
		conv_in_channel = self.in_channel
		conv_out_channel = self.out_channel
		channel_mul = self.channel_mul
		max_conv_blocks = 0
		valid_block = True
		while valid_block:
			max_conv_blocks += 1
			conv_out_length = int(np.floor((in_length - kernel_size) / stride + 1))
			conv_out_channel = conv_out_channel * (channel_mul)
			in_length = conv_out_length
			if conv_out_length < kernel_size:
				valid_block = False
		if include_pooling:
			max_conv_blocks = max_conv_blocks - 1
		return max_conv_blocks

	def get_conv_blocks_output_size(self, in_length, num_conv_blocks):
		channel_mul = self.channel_mul
		kernel_size = self.kernel_size
		stride = self.stride
		include_pooling = self.include_pooling
		conv_out_length = int(np.floor((in_length - kernel_size) / stride + 1))
		conv_in_channel = self.in_channel
		conv_out_channel = self.out_channel
		for block in range(num_conv_blocks):
			conv_out_length = int(np.floor((in_length - kernel_size) / stride + 1))
			output_length = conv_out_channel * conv_out_length
			conv_in_channel = int(conv_out_channel)
			conv_out_channel = conv_out_channel * (channel_mul)
			in_length = conv_out_length
		if include_pooling:
			conv_out_length = int(np.floor((in_length - kernel_size) / stride + 1))
		return conv_in_channel * conv_out_length

	def get_dec_blocks_output_size(self, num_conv_blocks):
		in_length = self.hidden_size
		conv_out_length = int(self.stride * (in_length - 1) + self.kernel_size)
		for block in range(num_conv_blocks):
			in_length = conv_out_length
			conv_out_length = int(self.stride * (in_length - 1) + self.kernel_size)
		return conv_out_length

	# Reconstruct structure from structure
	def reconstruct_structure(self, x):
		enc = self.gen_a.encode(x)
		shared_enc = self.enc_shared.forward(enc)
		shared_dec = self.dec_shared.forward(shared_enc)
		recon = self.gen_a.decode(shared_dec)
		return recon

	# Reconstruct spectrum from spectrum
	def reconstruct_spectrum(self, x):
		enc = self.gen_b.encode(x)
		shared_enc = self.enc_shared.forward(enc)
		shared_dec = self.dec_shared.forward(shared_enc)
		recon = self.gen_b.decode(shared_dec)
		return recon

	# Predict spectrum from structure
	def predict_spectrum(self, x):
		enc = self.gen_a.encode(x)
		shared_enc = self.enc_shared.forward(enc)
		shared_dec = self.dec_shared.forward(shared_enc)
		pred = self.gen_b.decode(shared_dec)
		return pred

	# Predict structure from spectrum
	def predict_structure(self, x):
		enc = self.gen_b.encode(x)
		shared_enc = self.enc_shared.forward(enc)
		shared_dec = self.dec_shared.forward(shared_enc)
		pred = self.gen_a.decode(shared_dec)
		return pred

	# Reconstruct and predict spectrum and structure from inputs
	def reconstruct_all_predict_all(self, x_a, x_b):

		enc_a = self.gen_a.encode(x_a)
		enc_b = self.gen_b.encode(x_b)

		shared_enc_a = self.enc_shared.forward(enc_a)
		shared_enc_b = self.enc_shared.forward(enc_b)

		shared_dec_a = self.dec_shared.forward(shared_enc_a)
		shared_dec_b = self.dec_shared.forward(shared_enc_b)

		x_ba = self.gen_a.decode(shared_dec_b)
		x_ab = self.gen_b.decode(shared_dec_a)

		x_a_recon = self.gen_a.decode(shared_dec_a)
		x_b_recon = self.gen_b.decode(shared_dec_b)

		# x_ba = self.gen_a.decode_mlp(shared_dec_b)
		# x_ab = self.gen_b.decode_mlp(shared_dec_a)

		# x_a_recon = self.gen_a.decode_mlp(shared_dec_a)
		# x_b_recon = self.gen_b.decode_mlp(shared_dec_b)

		return x_a_recon, x_b_recon, x_ba, x_ab

	def recon_criterion(self, pred, target):
		loss_fn = self.loss_fn_gen
		loss = loss_fn(pred, target)
		return loss

	def forward(self, x_a, x_b):
		enc_a = self.gen_a.encode(x_a)
		enc_b = self.gen_b.encode(x_b)

		shared_enc_a = self.enc_shared.forward(enc_a)
		shared_enc_b = self.enc_shared.forward(enc_b)

		shared_dec_a = self.dec_shared.forward(shared_enc_a)
		shared_dec_b = self.dec_shared.forward(shared_enc_b)

		x_ba = self.gen_a.decode(shared_dec_b)
		x_ab = self.gen_b.decode(shared_dec_a)

		# x_ba = self.gen_a.decode_mlp(shared_dec_b)
		# x_ab = self.gen_b.decode_mlp(shared_dec_a)

		return x_ab, x_ba

	def gen_update(self, x_a, x_b):
		self.gen_opt.zero_grad()
		# encode
		enc_a = self.gen_a.encode(x_a)
		enc_b = self.gen_b.encode(x_b)

		# encode shared layer
		shared_enc_a = self.enc_shared.forward(enc_a)
		shared_enc_b = self.enc_shared.forward(enc_b)
		# decode shared layer
		shared_dec_a = self.dec_shared.forward(shared_enc_a)
		shared_dec_b = self.dec_shared.forward(shared_enc_b)

		# decode (within domain)
		x_a_recon = self.gen_a.decode(shared_dec_a)
		x_b_recon = self.gen_b.decode(shared_dec_b)

		# decode (cross domain)
		x_ba = self.gen_a.decode(shared_dec_b)
		x_ab = self.gen_b.decode(shared_dec_a)

		# # decode (within domain)
		# x_a_recon = self.gen_a.decode_mlp(shared_dec_a)
		# x_b_recon = self.gen_b.decode_mlp(shared_dec_b)

		# # decode (cross domain)
		# x_ba = self.gen_a.decode_mlp(shared_dec_b)
		# x_ab = self.gen_b.decode_mlp(shared_dec_a)

		# scale loss by mean-maximum value of input
		a_max = torch.max(x_a)
		b_max = torch.max(x_b)

		# reconstruction loss
		loss_recon_a = self.recon_criterion(x_a_recon, x_a) / a_max
		loss_recon_b = self.recon_criterion(x_b_recon, x_b) / b_max
		loss_pred_a = self.recon_criterion(x_ba, x_a) / a_max
		loss_pred_b = self.recon_criterion(x_ab, x_b) / b_max

		# total loss
		loss_total = loss_recon_a + loss_recon_b + loss_pred_a + loss_pred_b

		# loss_gen = loss_gen_total.item()

		loss_total.backward()
		self.gen_opt.step()

	def dis_update(self, x_a, x_b):
		# encode
		self.dis_opt.zero_grad()
		y_a = self.gen_a.encode(x_a)
		y_b = self.gen_b.encode(x_b)

		# encode shared layer
		shared_enc_a = self.enc_shared.forward(y_a)
		shared_enc_b = self.enc_shared.forward(y_b)
		# decode shared layer
		shared_dec_a = self.dec_shared.forward(shared_enc_a)
		shared_dec_b = self.dec_shared.forward(shared_enc_b)

		# decode (cross domain)
		x_ba = self.gen_a.decode(shared_dec_b)
		x_ab = self.gen_b.decode(shared_dec_a)
		# decode (within domain)
		x_a_recon = self.gen_a.decode(shared_dec_a)
		x_b_recon = self.gen_b.decode(shared_dec_b)

		# # decode (cross domain)
		# x_ba = self.gen_a.decode_mlp(shared_dec_b)
		# x_ab = self.gen_b.decode_mlp(shared_dec_a)
		# # decode (within domain)
		# x_a_recon = self.gen_a.decode_mlp(shared_dec_a)
		# x_b_recon = self.gen_b.decode_mlp(shared_dec_b)

		# Discriminator loss for real inputs
		loss_dis_adv_a = self.dis_a.calc_gen_loss(x_a)
		loss_dis_adv_b = self.dis_b.calc_gen_loss(x_b)

		loss_gen_adv_a = self.dis_a.calc_dis_loss(x_ba, x_a)
		loss_gen_adv_b = self.dis_b.calc_dis_loss(x_ab, x_b)

		loss_gen_recon_a = self.dis_a.calc_dis_loss(x_a_recon, x_a)
		loss_gen_recon_b = self.dis_b.calc_dis_loss(x_b_recon, x_b)

		loss_real = 0.5 * (loss_dis_adv_a + loss_dis_adv_b)
		loss_fake = 0.25 * (
			loss_gen_adv_a + loss_gen_recon_a + loss_gen_adv_b + loss_gen_recon_b
		)

		self.loss_dis_total = loss_real + loss_fake

		self.loss_dis_total.backward()
		self.dis_opt.step()


# class SharedLayer(nn.Module):
# 	# Autoencoder architecture
# 	def __init__(self, num_hidden_layer, hidden_size, dropout, activation):
# 		super().__init__()
# 		self.num_hidden_layer = num_hidden_layer
# 		self.hidden_size = hidden_size
# 		self.activation = activation
# 		self.dropout = dropout
# 		linear_layers = []
# 		for layer in range(num_hidden_layer):
# 			linear_layers.append(nn.Linear(self.hidden_size, self.hidden_size))
# 			linear_layers.append(nn.BatchNorm1d(self.hidden_size))
# 			linear_layers.append(self.activation)
# 		self.linear_layers = nn.Sequential(*linear_layers)
# 		self.out_layer = nn.Sequential(
# 			nn.Linear(self.hidden_size, self.hidden_size),
# 			nn.BatchNorm1d(self.hidden_size),
# 		)

# 	def forward(self, x):
# 		x = self.linear_layers(x)
# 		x = self.out_layer(x)
# 		return x

# WITHOUT BATCH NORMALISATION
class SharedLayer(nn.Module):
	# Autoencoder architecture
	def __init__(self, num_hidden_layer, hidden_size, dropout, activation):
		super().__init__()
		self.num_hidden_layer = num_hidden_layer
		self.hidden_size = hidden_size
		self.activation = activation
		self.dropout = dropout
		linear_layers = []
		for layer in range(num_hidden_layer):
			linear_layers.append(nn.Linear(self.hidden_size, self.hidden_size))
			linear_layers.append(self.activation)
		self.linear_layers = nn.Sequential(*linear_layers)
		self.out_layer = nn.Sequential(
			nn.Linear(self.hidden_size, self.hidden_size),
		)

	def forward(self, x):
		x = self.linear_layers(x)
		x = self.out_layer(x)
		return x


class AEGenMLP(nn.Module):
	# Autoencoder architecture
	def __init__(
		self, input_size, num_hidden_layer, hidden_size, dropout, activation, pool
	):
		super().__init__()
		self.input_size = input_size
		self.output_size = input_size
		self.num_hidden_layer = num_hidden_layer
		self.activation = activation
		self.hidden_size = hidden_size
		self.dropout = dropout

		enc_layer = []
		for layer in range(num_hidden_layer - 1):
			enc_layer.append(nn.Linear(self.hidden_size, self.hidden_size))
			enc_layer.append(nn.BatchNorm1d(self.hidden_size))
			enc_layer.append(self.activation)
		self.enc_layer = nn.Sequential(*enc_layer)

		dec_layer = []
		for layer in range(num_hidden_layer):
			dec_layer.append(nn.Linear(self.hidden_size, self.hidden_size))
			dec_layer.append(nn.BatchNorm1d(self.hidden_size))
			dec_layer.append(self.activation)
		self.dec_layer = nn.Sequential(*dec_layer)
		self.enc_input = nn.Sequential(
			nn.Linear(self.input_size, self.hidden_size),
			nn.BatchNorm1d(self.hidden_size),
			self.activation,
		)
		self.enc_output = nn.Sequential(
			nn.Linear(self.hidden_size, self.hidden_size),
			nn.BatchNorm1d(self.hidden_size),
		)
		self.dec_output = nn.Linear(self.hidden_size, self.output_size)
		# self.dec_output = nn.Sequential(
		#     nn.Linear(self.hidden_size, self.output_size),
		#     self.activation)

	def encode(self, x):
		x = self.enc_input(x)
		x = self.enc_layer(x)
		out = self.enc_output(x)
		return x

	def decode(self, x):
		x = self.dec_layer(x)
		out = self.dec_output(x)
		return out



class AEGenCNN(nn.Module):
	# Autoencoder architecture
	def __init__(
		self,
		input_size,
		num_hidden_layer,
		hidden_size,
		dropout,
		activation,
		out_channel,
		channel_mul,
		kernel_size,
		stride,
		pooling_layer,
		num_conv_blocks,
		enc_conv_block_output_size,
		enc_shared_hidden_size,
		dec_conv_block_output_size,
	):
		super().__init__()
		self.input_size = input_size
		self.output_size = input_size
		self.num_hidden_layer = num_hidden_layer
		self.activation = activation
		self.hidden_size = hidden_size
		self.dropout = dropout

		self.out_channel = out_channel
		self.channel_mul = channel_mul
		self.kernel_size = kernel_size
		self.stride = stride
		self.pooling_layer = pooling_layer

		self.num_conv_blocks = num_conv_blocks
		self.enc_conv_block_output_size = enc_conv_block_output_size
		self.enc_shared_hidden_size = enc_shared_hidden_size
		self.dec_conv_block_output_size = dec_conv_block_output_size

		# ---------------------------
		# ENCODER CONVOLUTIONAL BLOCK
		self.enc_conv_block = []

		max_channel = ((num_conv_blocks - 1) * out_channel * channel_mul)
		Lin_enc = self.input_size

		enc_out_channel = self.out_channel
		enc_conv_ll = []

		self.enc_conv_block.append(nn.Conv1d(in_channels=1, out_channels=out_channel, kernel_size=kernel_size, stride=stride))
		self.enc_conv_block.append(self.activation)

		for block in range(self.num_conv_blocks):
		  enc_conv_ll.append(Lin_enc)
		  enc_in_channel = int(enc_out_channel)
		  enc_out_channel = int(enc_out_channel * self.channel_mul)
		  Lout_enc = math.floor((Lin_enc-(self.kernel_size-1)-1)/self.stride + 1)
		  conv_layer = nn.Conv1d(enc_in_channel, enc_out_channel, kernel_size=self.kernel_size, stride=self.stride)
		  Lin_enc = Lout_enc
		  if block < (self.num_conv_blocks-1):
		    self.enc_conv_block.append(conv_layer)
		    self.enc_conv_block.append(self.activation)

		self.enc_conv_block = nn.Sequential(*self.enc_conv_block)
		enc_conv_ll.reverse()
		# ---------------------------
		# DECODER CONVOLUTIONAL BLOCK
		self.max_channel = (
			(self.num_conv_blocks - 1) * self.out_channel * self.channel_mul
		)
		self.dec_conv_block = []

		dec_out_channel = max_channel

		Lin_dec = Lout_enc

		for block in range(self.num_conv_blocks):
		  dec_in_channel = int(dec_out_channel)
		  dec_out_channel = int(dec_out_channel / channel_mul)
		  Lout_dec = (Lin_dec-1)*self.stride + (self.kernel_size-1) + 1
		  output_padding = enc_conv_ll[block] - Lout_dec
		  Lout_dec += output_padding
		  tconv_layer = nn.ConvTranspose1d(dec_in_channel, dec_out_channel, kernel_size=self.kernel_size, stride=self.stride, output_padding = output_padding)
		  Lin_dec = Lout_dec
		  if block < (num_conv_blocks-1):
		    self.dec_conv_block.append(tconv_layer)
		    self.dec_conv_block.append(self.activation)
		  elif block < num_conv_blocks:
		    self.dec_conv_block.append(nn.ConvTranspose1d(in_channels=dec_in_channel, out_channels=1, kernel_size=self.kernel_size, stride=self.stride, output_padding = output_padding))
		    self.dec_conv_block.append(self.activation)

		self.dec_conv_block = nn.Sequential(*self.dec_conv_block)
		# ---------------------------

	def encode(self, x):
		x = x.unsqueeze(0)
		x = x.permute(1, 0, 2)
		out = self.enc_conv_block(x)
		out = out.flatten(1)
		out = nn.Linear(self.enc_conv_block_output_size, self.hidden_size)(out)
		return out

	def decode(self, x):
		x = nn.Linear(self.hidden_size, self.enc_conv_block_output_size)(x)
		x = x.unflatten(1,(self.max_channel, int(self.enc_conv_block_output_size / self.max_channel)))
		x = self.dec_conv_block(x)
		out = x.squeeze(dim=1)
		return out

	# def encode(self, x):
	# 	x = x.unsqueeze(0)
	# 	x = x.permute(1, 0, 2)
	# 	x = self.enc_conv_block(x)
	# 	x = x.view(x.size(0), -1)
	# 	out = nn.Linear(self.enc_conv_block_output_size, self.enc_shared_hidden_size)(x)
	# 	return out

	# def decode(self, x):
	# 	# convolutional decoder
	# 	x = nn.Linear(self.enc_shared_hidden_size, self.enc_conv_block_output_size)(x)
	# 	x = x.unsqueeze(0).permute(1,0,2)
	# 	x = x.view(x.size(0),self.max_channel, int(self.enc_conv_block_output_size / self.max_channel))
	# 	x = self.dec_conv_block(x)
	# 	x = x.view(x.size(0), -1)
	# 	out = x
	# 	# out = nn.Linear(self.dec_conv_block_output_size, self.output_size)(x)
	# 	return out


class Dis(nn.Module):
	# Discriminator architecture
	def __init__(
		self, input_size, num_hidden_layer, hidden_size, dropout, activation, loss_fn
	):
		super().__init__()
		self.input_size = input_size
		self.num_hidden_layer = num_hidden_layer
		self.hidden_size = hidden_size
		self.dropout = dropout
		self.activation = activation
		self.loss_fn = loss_fn
		linear_layers = []
		for layer in range(num_hidden_layer - 1):
			linear_layers.append(nn.Linear(self.hidden_size, self.hidden_size))
			linear_layers.append(nn.BatchNorm1d(self.hidden_size))
			linear_layers.append(self.activation)
		self.linear_layers = nn.Sequential(*linear_layers)
		self.input_layer = nn.Sequential(
			nn.Linear(self.input_size, self.hidden_size),
			nn.BatchNorm1d(self.hidden_size),
			self.activation,
		)
		self.output_layer = nn.Sequential(nn.Linear(self.hidden_size, 1), nn.Sigmoid())

	def forward(self, x):
		x = self.input_layer(x)
		x = self.linear_layers(x)
		out = self.output_layer(x)
		return out

	def calc_dis_loss(self, input_fake, input_real):
		# Calculate the loss to train D
		out0 = self.forward(input_fake)
		out1 = self.forward(input_real)
		loss = self.loss_fn(out0, out1)
		return loss

	def calc_gen_loss(self, input_fake):
		# Calculate the loss to train G
		out0 = self.forward(input_fake)
		ones = torch.ones((input_fake.size(0), 1))
		loss = self.loss_fn(out0, ones)
		return loss


def weight_init(m):
	if isinstance(m, nn.Linear):
		nn.init.zeros_(m.bias)
		nn.init.xavier_uniform_(m.weight)


# Select activation function from hyperparams inputs
class ActivationSwitch:
	def fn(self, activation):
		default = nn.PReLU()
		return getattr(
			self, f"activation_function_{activation.lower()}", lambda: default
		)()

	def activation_function_relu(self):
		return nn.ReLU()

	def activation_function_prelu(self):
		return nn.PReLU()

	def activation_function_tanh(self):
		return nn.Tanh()

	def activation_function_sigmoid(self):
		return nn.Sigmoid()

	def activation_function_elu(self):
		return nn.ELU()

	def activation_function_leakyrelu(self):
		return nn.LeakyReLU()

	def activation_function_selu(self):
		return nn.SELU()


# Select loss function from hyperparams inputs
class LossSwitch:
	def fn(self, loss_fn):
		default = nn.MSELoss()
		return getattr(self, f"loss_function_{loss_fn.lower()}", lambda: default)()

	def loss_function_mse(self):
		return nn.MSELoss()

	def loss_function_bce(self):
		# return nn.BCELoss()
		return nn.BCEWithLogitsLoss()

	def loss_function_emd(self):
		return EMDLoss()


# Earth mover distance as loss function
class EMDLoss(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, y_true, y_pred):
		loss = torch.mean(
			torch.square(torch.cumsum(y_true, dim=-1) - torch.cumsum(y_pred, dim=-1)),
			dim=-1,
		).sum()
		return loss


# Select convolutional pooling layer function from hyperparams inputs
class PoolingSwitch:
	def __init__(self, kernel_size, stride):
		self.kernel_size = kernel_size
		self.stride = stride

	def fn(self, pool_fn):
		default = no_pooling()
		return getattr(self, f"pooling_layer_{pool_fn.lower()}", lambda: default)()

	def pooling_layer_avg(self):
		return nn.AvgPool1d(self.kernel_size, self.stride)

	def pooling_layer_max(self):
		return nn.MaxPool1d(self.kernel_size, self.stride)


# Pooling function that just returns input
class no_pooling(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, x):
		return x


def train_aegan_cnn(x, y, hyperparams, n_epoch):

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	torch.manual_seed(1)

	x = torch.from_numpy(x)
	y = torch.from_numpy(y)

	n_x_features = x.shape[1]
	n_y_features = y.shape[1]

	dataset = torch.utils.data.TensorDataset(x, y)
	trainloader = torch.utils.data.DataLoader(dataset, batch_size=64)

	hyperparams["input_size_a"] = n_x_features
	hyperparams["input_size_b"] = n_y_features

	model = AEGAN_CNN(
		dim_a=hyperparams["input_size_a"],
		dim_b=hyperparams["input_size_b"],
		hidden_size=hyperparams["hidden_size"],
		dropout=hyperparams["dropout"],
		n_hl_gen=hyperparams["n_hl_gen"],
		n_hl_shared=hyperparams["n_hl_shared"],
		n_hl_dis=hyperparams["n_hl_dis"],
		activation=hyperparams["activation"],
		loss_gen=hyperparams["loss_gen"],
		loss_dis=hyperparams["loss_dis"],
		lr_gen=hyperparams["lr_gen"],
		lr_dis=hyperparams["lr_dis"],
		out_channel=hyperparams["out_channel"],
		channel_mul=hyperparams["channel_mul"],
		kernel_size=hyperparams["kernel_size"],
		stride=hyperparams["stride"],
		pool_fn=hyperparams["pool_fn"],
		include_pooling=hyperparams["include_pooling"],
		num_conv_blocks=hyperparams["num_conv_blocks"],
	)

	model.to(device)

	# Initialise weights
	model.apply(weight_init)

	model.train()
	loss_fn = nn.MSELoss()

	train_total_loss = [None] * n_epoch
	train_loss_x_recon = [None] * n_epoch
	train_loss_y_recon = [None] * n_epoch
	train_loss_x_pred = [None] * n_epoch
	train_loss_y_pred = [None] * n_epoch

	for epoch in range(n_epoch):
		running_loss_recon_x = 0
		running_loss_recon_y = 0
		running_loss_pred_x = 0
		running_loss_pred_y = 0
		running_gen_loss = 0
		running_dis_loss = 0
		for inputs_x, inputs_y in trainloader:
			inputs_x, inputs_y = inputs_x.to(device), inputs_y.to(device)
			inputs_x, inputs_y = inputs_x.float(), inputs_y.float()

			model.gen_update(inputs_x, inputs_y)
			model.dis_update(inputs_x, inputs_y)

			recon_x, recon_y, pred_x, pred_y = model.reconstruct_all_predict_all(
				inputs_x, inputs_y
			)

			# Track running losses
			running_loss_recon_x += loss_fn(recon_x, inputs_x)
			running_loss_recon_y += loss_fn(recon_y, inputs_y)
			running_loss_pred_x += loss_fn(pred_x, inputs_x)
			running_loss_pred_y += loss_fn(pred_y, inputs_y)

			loss_gen_total = (
				running_loss_recon_x
				+ running_loss_recon_y
				+ running_loss_pred_x
				+ running_loss_pred_y
			)
			loss_dis = model.loss_dis_total

			running_gen_loss += loss_gen_total.item()
			running_dis_loss += loss_dis.item()

		running_gen_loss = running_gen_loss / len(trainloader)
		running_dis_loss = running_dis_loss / len(trainloader)

		running_loss_recon_x = running_loss_recon_x.item() / len(trainloader)
		running_loss_recon_y = running_loss_recon_y.item() / len(trainloader)
		running_loss_pred_x = running_loss_pred_x.item() / len(trainloader)
		running_loss_pred_y = running_loss_pred_y.item() / len(trainloader)

		train_loss_x_recon[epoch] = running_loss_recon_x
		train_loss_y_recon[epoch] = running_loss_recon_y
		train_loss_x_pred[epoch] = running_loss_pred_x
		train_loss_y_pred[epoch] = running_loss_pred_y

		train_total_loss[epoch] = running_gen_loss

		print(f">>> Epoch {epoch}...")
		print(
			f">>> Running reconstruction loss (structure) = {running_loss_recon_x:.4f}"
		)
		print(
			f">>> Running reconstruction loss (spectrum) =  {running_loss_recon_y:.4f}"
		)
		print(
			f">>> Running prediction loss (structure) =     {running_loss_pred_x:.4f}"
		)
		print(
			f">>> Running prediction loss (spectrum) =      {running_loss_pred_y:.4f}"
		)

	losses = {
		"train_loss": train_total_loss,
		"loss_x_recon": train_loss_x_recon,
		"loss_y_recon": train_loss_y_recon,
		"loss_x_pred": train_loss_x_pred,
		"loss_y_pred": train_loss_y_pred,
	}

	return losses, model
