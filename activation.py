def activation_fn(self, x):
        """
        A method of FFL which contains the operation and defination of given activation function.
        """        
        if self.activation == 'relu':
            x[x < 0] = 0
            return x
        if self.activation == None or self.activation == "linear":
            return x        
        if self.activation == 'tanh':
            return np.tanh(x)
        if self.activation == 'sigmoid':    
            return 1 / (1 + np.exp(-x))
        if self.activation == "softmax":
            x = x - np.max(x)
            s = np.exp(x)
            return s / np.sum(s)
