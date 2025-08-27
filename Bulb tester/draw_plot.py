import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from experiment import *
from test import prob_vector, bulb_vector, num_plots, test_matrices, num_experiments

def draw_plot(ax, matrix, title):
    plt.style.use('_mpl-gallery-nogrid')

    ax.set_xticks(np.arange(len(prob_vector)))
    ax.set_xticklabels(prob_vector)

    ax.set_yticks(np.arange(len(bulb_vector)))
    ax.set_yticklabels(bulb_vector)

    min_heatmap_value = min(min(matrix))
    max_heatmap_value = max(max(matrix))
    
    # Define the colormap and normalization for the positive values
    norm = mcolors.LogNorm(vmin = min_heatmap_value, vmax = max_heatmap_value)
    
    # Conditional formatting for colours in the confidence intervals
    if 0 < min_heatmap_value < max_heatmap_value <= 1:
        min_heatmap_value = 0
        max_heatmap_value = 1
        norm = mcolors.Normalize(vmin = min_heatmap_value, vmax = max_heatmap_value)
    elif min_heatmap_value < 1:
        min_heatmap_value = 1
        norm = mcolors.LogNorm(vmin = min_heatmap_value, vmax = max_heatmap_value)
        
    # Plot negative values first
    cmap = "viridis"
    neg_norm = mcolors.BoundaryNorm(boundaries=[-np.inf, 0], ncolors=1)  # Only the boundary for negatives
    ax.imshow(matrix, cmap = cmap, norm = neg_norm)

    # Now plots positive/zero values, masking and avoiding negative values
    data_pos = np.ma.masked_less_equal(matrix, 0)
    ax.imshow(data_pos, cmap = cmap, norm = norm)    

    plt.xlabel("Probability of breaking")
    plt.ylabel("Number of bulbs")

    for (i, j), val in np.ndenumerate(matrix):
        if int(val) == val:                     # Stops the heatmap from displaying 10001 as 10001.0
            val = int(val)
        ax.text(j, i, val, ha='center', va='center', color='white', fontsize=12, fontweight='bold')

    ax.set_title(title, fontsize = 14)
    ax.set_xlabel("Defective blub probability", fontsize = 9)  # Add x-axis label
    ax.set_ylabel("Number of bulbs", fontsize = 9)  # Add y-axis label
    
    fig.colorbar(ax.imshow(data_pos, cmap=cmap, norm=norm, alpha=1.0), ax=ax, shrink=1)
    


# We used Chat-GPT to help us plot the following eight plots all at once:
TITLES_VECTOR = [
    "Simulated number of tries in B", 
    "Theoretical number of tries in B",
    "Simulated number of tries in C", 
    "Theoretical number of tries in C",
    "Confidence in B",
    "Confidence in C",
    "Efficiency ratio of C over B",
    "Number of groups in C"
]

# Create a 4x2 grid of subplots
fig, ax = plt.subplots(nrows = 4, ncols=2, figsize=(10, 20))
ax = ax.flatten()  # Flatten for easy iteration

# Draw each plot
for i in range(num_plots):
    draw_plot(ax[i], test_matrices[i], TITLES_VECTOR[i])

fig.suptitle(f"Results of tests B and C with sample size {num_experiments}", fontsize = 18, y = 1)
plt.tight_layout()
plt.savefig("figs/sim_theo_conf_B_and_C.png", dpi = 300, bbox_inches = "tight")
plt.show()