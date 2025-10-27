import numpy as np
import scipy.stats as stats

# Sample Data (weights in grams)
weights = np.array([48, 50, 52, 47, 49, 51, 53, 50, 48, 52])
     
# Sample Mean and Variance
mean = np.mean(weights)
variance = np.var(weights, ddof=1)  # unbiased estimate
n = len(weights)
     
# Confidence Interval for Mean (95%)
alpha = 0.05
t_critical = stats.t.ppf(1 - alpha/2, df=n-1)
margin_of_error = t_critical * (np.sqrt(variance/n))
ci_lower = mean - margin_of_error
ci_upper = mean + margin_of_error

print("Sample Variance (σ² estimate):", variance)
print("95% Confidence Interval for Mean: (", ci_lower, ",", ci_upper, ")")
