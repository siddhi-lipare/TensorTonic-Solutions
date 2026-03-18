import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    # Write code here
    x = np.array(x)
    p = np.array(p)
    if p.sum() == 1 and x.shape==p.shape:
        exp_value = np.dot(x,p)
    else:
        raise ValueError("probabilities must sum to 1 or the shapes of x and p don't match")

    return exp_value
