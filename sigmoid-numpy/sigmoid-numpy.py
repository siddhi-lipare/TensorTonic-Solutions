import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    # Write code here
    x_new = np.array(x)

    return 1/(1+np.exp(-x_new))
    