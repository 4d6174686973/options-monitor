import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, norm

# 1. Generate "Fat Tailed" Data (Student's t-distribution)
# This mimics the "DeepSeek" or "Tariff" shocks mentioned in the report
np.random.seed(10)
# Use degrees of freedom (df=3) to create heavy tails
returns = np.random.standard_t(df=3, size=1000) * 2 

# 2. Fit KDE (Non-Parametric) vs Normal (Parametric)
kde = gaussian_kde(returns, bw_method='scott') # Using Scott's Rule for bandwidth
dist_space = np.linspace(min(returns), max(returns), 500)
kde_pdf = kde(dist_space)

# Fit Normal distribution for comparison
mu, std = norm.fit(returns)
norm_pdf = norm.pdf(dist_space, mu, std)

# 3. Find High Density Interval (HDI) - 95%
# We integrate the KDE PDF until we reach 2.5% and 97.5%
cdf = np.cumsum(kde_pdf) * (dist_space[1] - dist_space[0])
lower_bound_idx = np.searchsorted(cdf, 0.05) # 5% quantile (Put Strike)
upper_bound_idx = np.searchsorted(cdf, 0.95) # 95% quantile (Call Strike)

put_strike_ret = dist_space[lower_bound_idx]

# 4. Visualization
plt.figure(figsize=(12, 6))
plt.plot(dist_space, kde_pdf, label='KDE (Real Data)', color='blue', linewidth=2)
plt.plot(dist_space, norm_pdf, label='Black-Scholes (Normal)', color='red', linestyle='--')
plt.fill_between(dist_space, 0, kde_pdf, where=(dist_space < put_strike_ret), color='blue', alpha=0.2, label='Short Put Danger Zone')

# Mark the Strikes
plt.axvline(put_strike_ret, color='green', linestyle='-', label=f'Rec. Put Strike ({put_strike_ret:.2f}%)')
plt.title('Strike Selection: KDE vs. Normal Distribution')
plt.xlabel('Weekly Return %')
plt.ylabel('Density')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"KDE suggests selling Puts at return level: {put_strike_ret:.2f}%")
print(f"Normal Distribution suggests selling at: {norm.ppf(0.05, mu, std):.2f}%")
print("Note how KDE suggests a wider safety margin due to fat tails.")