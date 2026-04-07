import numpy as np
from sklearn.datasets import make_moons, make_circles, make_blobs

class Layer:
    def __init__(self, n_input, n_output, activation='relu'):
        self.weights = np.random.randn(n_input, n_output) * 0.1
        self.bias = np.zeros((1, n_output))
        self.activation_type = activation
        
        # State for visualization
        self.last_input = None
        self.last_z = None
        self.last_a = None
        
        # Gradients
        self.dW = None
        self.db = None

    def _activate(self, z):
        if self.activation_type == 'relu':
            return np.maximum(0, z)
        elif self.activation_type == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        elif self.activation_type == 'tanh':
            return np.tanh(z)
        return z

    def _activate_derivative(self, a):
        if self.activation_type == 'relu':
            return (a > 0).astype(float)
        elif self.activation_type == 'sigmoid':
            return a * (1 - a)
        elif self.activation_type == 'tanh':
            return 1 - a**2
        return np.ones_like(a)

    def forward(self, x):
        self.last_input = x
        self.last_z = np.dot(x, self.weights) + self.bias
        self.last_a = self._activate(self.last_z)
        return self.last_a

    def backward(self, da, learning_rate):
        # da is dL/da
        dz = da * self._activate_derivative(self.last_a) # dL/dz
        
        self.dW = np.dot(self.last_input.T, dz)
        self.db = np.sum(dz, axis=0, keepdims=True)
        
        dx = np.dot(dz, self.weights.T) # dL/dx for previous layer
        
        # Update weights (SGD)
        self.weights -= learning_rate * self.dW
        self.bias -= learning_rate * self.db
        
        return dx

class NeuralNetwork:
    def __init__(self, layers_config, learning_rate=0.01):
        self.layers = []
        self.learning_rate = learning_rate
        for i in range(len(layers_config) - 1):
            n_in, n_out, act = layers_config[i][0], layers_config[i+1][0], layers_config[i][1]
            self.layers.append(Layer(n_in, n_out, act))

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dL_dy):
        da = dL_dy
        for layer in reversed(self.layers):
            da = layer.backward(da, self.learning_rate)

    def train_step(self, x, y):
        # Forward
        y_pred = self.forward(x)
        
        # Loss (MSE for simplicity in visualization, or Binary Cross Entropy)
        # Using MSE here for general purpose
        loss = np.mean(0.5 * (y - y_pred)**2)
        
        # Backward
        dL_dy = -(y - y_pred) # dL/dy for MSE: 0.5 * (y-y_pred)^2 -> -(y-y_pred)
        self.backward(dL_dy)
        
        return loss

def generate_data(dataset_type='moons', n_samples=300, noise=0.1):
    if dataset_type == 'moons':
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    elif dataset_type == 'circles':
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=42)
    elif dataset_type == 'blobs':
        X, y = make_blobs(n_samples=n_samples, centers=2, n_features=2, cluster_std=noise*10, random_state=42)
    else:
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    
    return X, y.reshape(-1, 1)
