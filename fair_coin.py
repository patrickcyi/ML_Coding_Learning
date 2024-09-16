import numpy as np
# from scipy.stats import binom_test

# Step 1: Simulate coin tosses using numpy
def simulate_coin_tosses(n_tosses, p_head=0.1):
    return np.random.binomial(1, p_head, size=n_tosses)

def hypothesis_test_fair_coin(n_tosses, observed_heads):
    p_value = binom_test(observed_heads, n=n_tosses, p=0.5, alternative='two-sided')
    
    return p_value

def bootstrap_p_value(tosses, n_bootstrap=1000):
    observed_heads = np.sum(tosses)
    n_tosses = len(tosses)
    bootstrap_counts = []
    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.binomial(1, 0.1, size=n_tosses)  # Fair coin simulation
        bootstrap_heads = np.sum(bootstrap_sample)
        bootstrap_counts.append(bootstrap_heads)
    
    # Calculate the p-value: proportion of bootstrap samples that are as extreme as the observed_heads
    extreme_counts = np.sum(np.abs(np.array(bootstrap_counts) - n_tosses * 0.5) >= np.abs(observed_heads - n_tosses * 0.5))
    p_value = extreme_counts / n_bootstrap
    
    return p_value

# Step 3: Simulate and test
n_tosses = 50  # Number of coin tosses
coin_tosses = simulate_coin_tosses(n_tosses)

p_value = bootstrap_p_value(coin_tosses)
print(p_value)

