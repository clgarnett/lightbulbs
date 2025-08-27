import numpy as np 
from scipy.optimize import brentq
from experiment import experiment

def sim_f(n: int):
    """
    Simulates f(k) analytically as proposed in part (b) of the exercise.
    Namely, finds p in (0,1) analytically such that E[X] = k. This approximates min(0 < p < 1 : E[X] < k).
    Once p is found for the given k, sets f(k) = p. 
    Returns {f(k) : k in 2, ..., 50}
    :param n: Sample size.
    """

    MIN_K_VAL = 2
    MAX_K_VAL = 50

    def find_f_k(k):
        """
        Function inspired by https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brentq.html.
        Given a value of k, finds p such that E[X] - k = 0 with Brent’s method.
        Uses (1e-6, 1 - 1e-6) as an approximation of (0,1) for the range of possible values for p.
        We set f(k) to this value of p.
        """
        try:
            return brentq(lambda p: experiment(k, p).estimate_expectation_B(n) - k, 1e-6, 1 - 1e-6)
        except ValueError:
            return -1

    k_values = np.arange(MIN_K_VAL, MAX_K_VAL+1)            # all possible values of k: 2, 3, ..., 50
    f_k_values = np.array([find_f_k(k) for k in k_values])  # finds f(k) for all k: 2, 3, ..., 50
    return f_k_values