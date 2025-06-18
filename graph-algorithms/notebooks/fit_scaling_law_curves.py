# %%
import numpy as np
import matplotlib.pyplot as plt

num_of_tokens = np.array([2.27E+04, 4.53E+04, 1.13E+05, 2.27E+05, 4.53E+05, 1.13E+06, 2.27E+06])
losses = np.array([0.54, 0.51, 0.42, 0.24, 0.17, 0.06, 0.05])

num_of_tokens = np.log10(num_of_tokens)
losses = np.log10(losses)

# fit linear coefficients
coefficients = np.polyfit(num_of_tokens, losses, 1)
slope, intercept = coefficients
print(f"Slope: {slope}, Intercept: {intercept}")

# test how the significant the slope is
from scipy.stats import pearsonr
corr = pearsonr(num_of_tokens, losses)
print(f"Pearson correlation coefficient: {corr[0]}, p-value: {corr[1]}")


plt.figure(figsize=(10, 6))
plt.plot(num_of_tokens, losses, marker='o', linestyle='-', color='b', lw=3)
plt.plot(num_of_tokens, slope * num_of_tokens + intercept, color='r', linestyle='--', label='Fitted Line', lw=3)
plt.yticks(np.arange(-1.4, 0.4, 0.4), [ 0.03,  0.1, 0.3, 0.6, 1.6,])
plt.xticks(np.arange(4, 7, 1), ["10E4", "10E5", "10E6"])
plt.xlim(3.8, 6.5)
plt.tick_params(axis='both', which='major', labelsize=28)
plt.xlabel('Number of Tokens', fontsize=28)
plt.ylabel('Test Loss', fontsize=28)
plt.grid(True)
plt.legend(fontsize=28)
plt.savefig('fir_scaling_law_curves_10.png', dpi=300, bbox_inches='tight')
plt.show() 

# %%

import numpy as np
import matplotlib.pyplot as plt

num_of_tokens = np.array([4.63E+04, 9.26E+04, 2.31E+05, 4.63E+05, 9.26E+05, 2.31E+06, 4.63E+06])
losses = np.array([0.6754046082496643, 0.5697358846664429, 0.28, 0.26, 0.18, 0.06, 0.04])

num_of_tokens = np.log10(num_of_tokens)
losses = np.log10(losses)

# fit linear coefficients
coefficients = np.polyfit(num_of_tokens, losses, 1)
slope, intercept = coefficients
print(f"Slope: {slope}, Intercept: {intercept}")

# test how the significant the slope is
from scipy.stats import pearsonr
corr = pearsonr(num_of_tokens, losses)
print(f"Pearson correlation coefficient: {corr[0]}, p-value: {corr[1]}")

plt.figure(figsize=(10, 6))
plt.plot(num_of_tokens, losses, marker='o', linestyle='-', color='b', lw=3)
plt.plot(num_of_tokens, slope * num_of_tokens + intercept, color='r', linestyle='--', label='Fitted Line', lw=3)
plt.yticks(np.arange(-1.4, 0.4, 0.4), [ 0.03,  0.1, 0.3, 0.6, 1.6,])
plt.xticks(np.arange(4, 8, 1), ["10E4", "10E5", "10E6", "10E7"])
plt.xlim(4.5, 7.2)
plt.tick_params(axis='both', which='major', labelsize=28)
plt.xlabel('Number of Tokens', fontsize=28)
plt.ylabel('Test Loss', fontsize=28)
plt.grid(True)
plt.legend(fontsize=28)
plt.savefig('fir_scaling_law_curves_15.png', dpi=300, bbox_inches='tight')
plt.show() 