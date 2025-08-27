import numpy as np

class experiment:
    def __init__(self, k:int, p, num_groups:int= None, debug:bool = False):
        """ 
        An experiment object used for testing bulbs.
            
        :param k: A positive integer describing the number of bulbs produced.
        :param p: The probability that any one bulb is defective.
        :param num_groups: A number of groups to further divide the bulbs into, for tests of type C.
        :param debug: Enables print statements, useful for analyzing the performance of our program. 
        """ 
        assert k > 0 and k == int(k), f"Insert a positive integer for k."
        assert 0 <= p <= 1,           f"Insert a real number between 0 and 1 for p."
        assert num_groups is None or (0 < num_groups <= k and num_groups == int(num_groups)), f"Number of groups must be an integer between 1 and {k}"
        assert p < 1 or not num_groups is None, f"Insert a value for p less than one or set a value for num_groups." # Avoids computing ln(0) in D_E_C
        self.k = k                   # Number of bulbs.
        self.p = p                   # Probability that any bulb is defective.
        self.debug = debug           # Turns on print statements useful in development.
        np.random.seed(314159)       # Sets the seed for reproducible results

        def get_E_C(num_groups:int):
            """
            Returns the theoretical estimate of E[X] with test C given the number of groups.
            """
            return k + num_groups - k*(1-p)**(k/num_groups) 

        if num_groups is None:
            x_vals = range(1, k+1)                   # generates 1, 2, ..., k possible group sizes
            g_x_vals = [get_E_C(x) for x in x_vals]  # calculates the theoretical expectation E[X] for all of them
            num_groups = np.argmin(g_x_vals)+1       # returns the point on the domain where E[X] is lowest
        
        self.num_groups = num_groups       
        self.E_B = k + 1 - k*(1 - p)**k        # Theoretical expectation of X in test B.
        self.E_C = get_E_C(num_groups)         # Theoretical expectation of X in test C.
        self.estimation_B = None
        self.estimation_C = None               # We set both of these to none as we haven't simulated any tests yet. 
        self.BULBS = np.random.binomial(n = 1, p = 1-self.p, size = self.k)  # A constant sequence of bulbs. 
        
    def get_bulbs(self):
        """
        Mints a fresh batch of k bulbs and returns it. 
        """
        return np.random.binomial(n = 1, p = 1 - self.p, size = self.k) # n = 1 makes it a Bernoulli trial, as we want it
        
    def A(self):                  # Simulates test A
        return self.k
    
    def B(self, bulbs = None):    # Simulates test B
        """
        Places all the bulbs in series and tests them all at once.
        If this doesn't close the current, performs a test on each individual bulb. 
        """
        if bulbs is None:
            bulbs = self.get_bulbs() # Generates a random batch of k bulbs if none are given. 
        
        if all(bulbs):         # If all given bulbs work, performs only one test. 
            return 1
        return len(bulbs) + 1  # Else, resorts to checking each bulb individually, performing len(bulbs) + 1 tests.         
        
    def estimate_expectation_B(self, n:int) -> float:
        """
        Calculates E[X_1] using the outcome of n experiments of type B,
        which by the strong law of large numbers equals (X_1 + ... + X_n)/n.
        """
        estimation = 0
        for i in range(n):
            bulbs = self.get_bulbs()     # Generates a new random batch of bulbs.
            estimation += self.B(bulbs)  # Simulates test B on the newly simulated batch of bulbs and ads the number of tests to the tally.
        estimation /= n
        self.estimation_B = estimation
        return estimation
    
    def get_confidence_B(self) -> float:
        """
        Returns one minus the relative error between the estimation of B and its theoretical expected value. 
        """
        assert self.estimation_B is not None, "You must run estimate_expectation_B(n) before calling this method."
        return 1 - abs(self.estimation_B - self.E_B)/abs(self.E_B)
        
    def C(self, bulbs = None):
        """
        Performs test B over a number of partitions of the total series of bulbs. 
        """
        if bulbs is None:
            bulbs = self.get_bulbs()  # Generates a random batch of k bulbs if none are given. 
        
        groups = np.array_split(np.array(bulbs), self.num_groups) # Splits the k bulbs into num_groups groups of group_size
        return sum([self.B(groups[i]) for i in range(self.num_groups)])
    
    def estimate_expectation_C(self, n:int) -> float:
        """
        Calculates E[X_1] using the outcome of n experiments of type C,
        which by the strong law of large numbers equals (X_1 + ... + X_n)/n.
        """
        estimation = 0
        for i in range(n):
            bulbs = self.get_bulbs()
            estimation += self.C(bulbs)
        estimation /= n
        self.estimation_C = estimation
        return estimation 
    
    def get_confidence_C(self) -> float:
        """
        Returns one minus the relative error between the estimation of C and its theoretical expected value. 
        """
        assert self.estimation_C is not None, "You must run estimate_expectation_B(n) before calling this method."
        return 1 - abs(self.estimation_C - self.E_C)/abs(self.E_C)