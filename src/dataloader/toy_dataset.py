import numpy as np
from numpy.random import uniform


def generate_1d_regression(
        n_points=2000,
        domain=(-2.0, 2.0),
        noise_std=0.1,
        seed=None):
    """
    Generate data points from periodic function
    Parameters
    ----------
    n_points : Number of data points to generate
    domain : range for data points
    noise_std : noise introduced to the data points
    seed : random seed

    Returns
    -------
    xs, ys : generated  data points
    """

    def f(x):
        x = np.atleast_1d(x)
        y = np.sin(6 * x) + 0.4 * x ** 2 - 0.1 * x ** 3 - x * np.cos(9 * np.sqrt(np.exp(x)))
        return y
    rng = np.random.RandomState(seed)
    xs = rng.uniform(*domain, size=n_points).astype(np.float32).reshape(-1, 1)
    ys = (f(xs) + rng.normal(0.0, noise_std, size=(n_points, 1))).astype(np.float32)
    return xs, ys