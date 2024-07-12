# %%
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import seaborn as sns

from matplotlib import rc
rc('font', **{'family':'sans-serif','sans-serif':['Helvetica']})
mpl.rcParams['savefig.dpi'] = 1200
mpl.rcParams['text.usetex'] = True  # not really needed

f, ax2 = plt.subplots(figsize=(6.8, 5))

output_accuracy = np.array([0, 48.44, 50.3889, 52.223, 54.2367, 56.2717, 58.655, 59.655])

insertion_sort = np.array([0, 56.3680, 67.2002, 79.8100, 91.4673, 95.4097, 99.03, 99.71])
x = np.arange(len(output_accuracy))
ax2.scatter(x, output_accuracy, s=150, color="royalblue", marker="o")
ax2.scatter(x, insertion_sort, s=150, color="forestgreen", marker="o")
p1 = ax2.plot(x, insertion_sort, lw = 4, color="forestgreen", linestyle="solid", label=r"$\mathrm{Our~approach}$") # r"$\mathrm{}x\mathrm{~as~input~prompt}$"

p2 = ax2.plot(x, output_accuracy, lw = 4, color="royalblue", linestyle="dashed", label=r"$\mathrm{Single~prompt}$") # "$\mathrm{}P_x\mathrm{~as~input~prompt}$"

# ax2.fill_between(
#     x, 
#     neg+neg_std,
#     neg-neg_std, 
#     color="orange", alpha=0.3
# )

ax2.set_ylabel(r'$\mathrm{Accuracy}~(\%)$', fontsize = 32)
ax2.set_xlabel(r'$\mathrm{\#~training~examples}$', fontsize = 32) # '+r'$\mathrm{~number~of~sampled~sets}
# ax.set_yticks(np.arange(0, 1.1, .2))
plt.xticks(x, [r'$0$', r'$10^3$', "", "", r'$10^4$', "", "", r'$10^5$'], fontsize=32)
ax2.set_yticks(np.arange(0, 101, 20))
# ax2.set_ylim((-3, 10))
# ax.set_xlim((-2.5, 3.5))

ax2.set_title(r'$\mathrm{Insertion~sort~}(m=5)$', fontsize = 32)
ax2.tick_params(labelsize=32)
ax2.grid(ls=':', lw=0.8)
plt.legend(fontsize=28, loc="lower right")

plt.tight_layout()
plt.savefig("./figures/sample_complexity_comparison_inputs.pdf", format="pdf", dpi=1200)
plt.show()