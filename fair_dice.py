import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare

# Step 1: Simulate rolling a fair die multiple times
def roll_die(n_rolls):
    return np.random.randint(1, 7, size=n_rolls)


def biased_die_rolls(n_rolls):
    # Define custom probabilities for each face (1 to 6)
    probabilities = [0.1, 0.1, 0.2, 0.2, 0.3, 0.1]
    return np.random.choice([1, 2, 3, 4, 5, 6], size=n_rolls, p=probabilities)

# Step 2: Simulate 1000 rolls of a fair die
n_rolls = 1000
rolls = roll_die(n_rolls)
biased_rolls = biased_die_rolls(1000)

# Step 3: Calculate the frequency of each outcome (1-6)
outcomes, counts = np.unique(rolls, return_counts=True)
outcomes, counts = np.unique(biased_die_rolls, return_counts=True)

observed_frequencies, _ = np.histogram(rolls, bins=np.arange(1, 8))
expected_frequencies = np.full(6, n_rolls / 6)
chi2_stat, p_value = chisquare(observed_frequencies, f_exp=expected_frequencies)


# Step 4: Plot the histogram of outcomes
plt.figure()
plt.bar(outcomes, counts, tick_label=[1, 2, 3, 4, 5, 6])

# Step 5: Print the outcomes and their counts
for outcome, count in zip(outcomes, counts):
    print(f"Face {outcome}: {count} times")
