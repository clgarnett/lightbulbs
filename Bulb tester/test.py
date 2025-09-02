import time
from sim_f import *
from experiment import *
import matplotlib.pyplot as plt
from round_to_fit import *
import os

def round_to_fit(n, digits):
    """
    Rounds as little as possible for it to fit in a certain number of digits.
    In the most extreme case, it returns n with all its decimal places cut off. 
    """
    assert int(digits) == digits and digits > 0, "Round up to a positive integer number of digits."
    integer_digits = num_digits = len(str(int(abs(n))))
    
    if integer_digits >= digits: # Returns the nearest integer if we don't have any space for any decimal places
        return int(round(n,0))
    
    return round(n, digits-integer_digits) # Returns the rational number with as many digits as it can to fit


# We got the code relating to the time module from Chat-GPT
start_time = time.time()

num_experiments = 10**3
num_plots       = 8

bulb_vector = [2, 10**1, 10**2, 10**3, 10**4]
prob_vector = [0.001, 0.01, 0.05, 0.5]
DIGITS_THAT_FIT = 5                             # How many digits we can aesthetically display in each bin in the heatmaps

test_matrices = [[] for _ in range(num_plots)]

for i in range(len(bulb_vector)):
    test_vectors = [[] for _ in range(num_plots)]
    
    for j in range(len(prob_vector)):
        test_start_time = time.time()
        e = experiment(k = bulb_vector[i], p = prob_vector[j])
        
        sim_B = e.estimate_expectation_B(num_experiments)
        sim_C = e.estimate_expectation_C(num_experiments)
        
        sim_B_confidence = e.get_confidence_B()
        sim_C_confidence = e.get_confidence_C()
        
        test_vectors[0].append(round_to_fit(sim_B, DIGITS_THAT_FIT))  # Simulated expectation of B
        test_vectors[1].append(round_to_fit(e.E_B, DIGITS_THAT_FIT))  # Theoretical expectation of B
        
        test_vectors[2].append(round_to_fit(sim_C, DIGITS_THAT_FIT))  # Simulated expectation of C
        test_vectors[3].append(round_to_fit(e.E_C, DIGITS_THAT_FIT))  # Theoretical expectation of C
        
        test_vectors[4].append(round_to_fit(sim_B_confidence, DIGITS_THAT_FIT))  # Confidence of simulation B
        test_vectors[5].append(round_to_fit(sim_C_confidence, DIGITS_THAT_FIT))  # Confidence of simulation C
        
        test_vectors[6].append(round_to_fit(sim_B/sim_C, DIGITS_THAT_FIT))   # Advantage of C over B
        test_vectors[7].append(e.num_groups)                                 # Number of groups in C
        
        test_end_time = time.time()
        print(f"Experimented with k = {bulb_vector[i]}, p = {prob_vector[j]}. Elapsed time: {round(test_end_time - test_start_time, 3)} seconds.")
        
    for j in range(num_plots):
        test_matrices[j].append(test_vectors[j])
        
end_time = time.time()
elapsed_time = end_time - start_time
print()

from draw_plot import *
print(f"Total elapsed time: {round(elapsed_time, 3)} seconds")  # Print with 2 decimal places










# The theoretical value that f(k) takes on for k in {2, ..., 50}
THEORETICAL_F = (lambda x: 1 - (1/x)**(1/x))(np.arange(2, 51))
    
start_time = time.time()    # Measures simulation time
f_k = sim_f(num_experiments)    # Simulates f_k
end_time = time.time()      # Measures simulation time

# Displays the results explicitly to the console
print(f"Elapsed time: {round(end_time - start_time, 3)} seconds")
print()
print(f"Simulated f(k) with sample size {num_experiments}:")
print(f_k)
print()
print(f"Theoretical f(k):")
print(THEORETICAL_F)
print()
print(f"Absolute difference:")
print(abs(f_k-THEORETICAL_F))
print()
print(f"Relative error:")
print(abs(f_k-THEORETICAL_F)/abs(THEORETICAL_F))
print()
print(f"Mean absolute difference: {np.mean(abs(f_k-THEORETICAL_F))}")
print()
print(f"Mean absolute relative error: {np.mean(abs(f_k-THEORETICAL_F)/abs(THEORETICAL_F))}")

# Displays the results in a "nice plot"
x = np.arange(2, 50+1, 1)
plt.plot(x, THEORETICAL_F, label="Theoretical value", color="mediumblue", linewidth=3, zorder=1)
plt.scatter(x, f_k, label="Simulated result", color="white", marker="X",
            edgecolors="orangered", linewidth=0.6, s=25, zorder=2)

plt.xlabel("k")
plt.xlim(2-1, 50+1)

# Gets the current ticks, which somehow includes a new k=60, removes this last one with [:-1] and adds the tick k=2
plt.xticks([2] + plt.gca().get_xticks().tolist()[:-1])

plt.ylabel("f(k)")
plt.legend()
plt.title("Simulated and theoretical results of f")
plt.grid(True)
os.makedirs("figs", exist_ok=True)
plt.savefig("figs/numerical_vs_theoretical_f_k.png", dpi = 300, bbox_inches = "tight")
plt.show()