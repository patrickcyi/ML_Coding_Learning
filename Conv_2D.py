import numpy as np

def simple_conv2d(input_matrix: np.ndarray, kernel: np.ndarray, padding: int, stride: int):
	input_height, input_width = input_matrix.shape
	kernel_height, kernel_width = kernel.shape
	pad_image= np.pad(input_matrix, ((padding, padding), (padding, padding)), mode="constant")
	pad_height, pad_width = pad_image.shape
	
	op_h = (pad_height- kernel_height)//stride +1
	op_w = (pad_width- kernel_width)//stride +1
	output = np.zeros((op_h, op_w))
	
	for i in range(op_h):
		for j in range(op_w):
			start_h, start_w = i * stride , j* stride
			end_h, end_w = start_h + kernel_height, start_w+ kernel_width
			
			region = pad_image[start_h:end_h, start_w:end_w]
			output[i,j] = np.sum(region * kernel)	
	return output
	
input_matrix = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
])

kernel = np.array([
    [1, 0],
    [-1, 1]
])

padding = 1
stride = 2
