import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    X = np.array(X)
    y = np.array(y)
    N, d = X.shape
    
    w = np.zeros(d)
    b = 0.0

    for _ in range(steps):
        z = np.dot(X, w) + b
        y_hat = _sigmoid(z)
    
        error = y_hat - y
        w_gradient = 1/N * np.dot(X.T, error)
        b_gradient = np.mean(error)
    
        w -= (lr*w_gradient)
        b -= (lr*b_gradient)

    return w, b
