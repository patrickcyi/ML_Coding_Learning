import numpy as np

def maxpool_forward(input, pool_size=2, stride=2):
    h, w = input.shape
    output_h = (h - pool_size) // stride + 1
    output_w = (w - pool_size) // stride + 1
    
    output = np.zeros((output_h, output_w))
    
    for i in range(output_h):
        for j in range(output_w):
            start_h, start_w = i * stride, j * stride
            end_h, end_w = start_h + pool_size, start_w + pool_size
            window = input[start_h:end_h, start_w:end_w]
            output[i, j] = np.max(window)
    
    return output
    
input_image = np.array([
    [1, 3, 2, 4],
    [5, 6, 1, 8],
    [2, 4, 5, 7],
    [3, 2, 9, 0]
])

output = maxpool_forward(input_image, pool_size=2, stride=2)
print("Output of Max Pooling Forward Pass:")
print(output)


backpass:

    grad_input = np.zeros_like(input) beofre loop
    
    
            # Find the position of the maximum value in the window
            max_pos = np.where(window == max_value)
            
            # Propagate the gradient to the position of the maximum value
            grad_input[start_h:end_h, start_w:end_w][max_pos] = grad_output[i, j]
