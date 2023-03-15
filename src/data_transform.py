import numpy as np
import torch

def fourier_transform_data(x):
	""" 
	Transform xanes spectra using Fourier
	"""
	y = np.hstack((x, x[:,::-1]))
	f = np.fft.fft(y)
	z = f.real
	return(z)


def inverse_fourier_transform_data(z):
	"""
	Get inverse of fourier transformed data
	"""
	iz = torch.fft.ifft(z).real[:,:z.shape[1]//2]
	return(iz)
