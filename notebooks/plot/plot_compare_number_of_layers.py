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

layer2 = np.array([0,  56.3680, 67.2002, 79.8100, 91.4673, 95.4097, 99.53, 99.71])
layer6 = np.array([0, 52.5520, 64.1750, 74.8895, 87.6128, 93.0090, 98.03, 99.71])
layer12 = np.array([0, 49.5967, 60.8560, 71.1193, 85.7183, 90.7130, 96.03, 98.71])
x = np.arange(len(layer2))
ax2.scatter(x, layer2, s=100, color="forestgreen", marker="o")
ax2.scatter(x, layer6, s=100, color="royalblue", marker="o")
ax2.scatter(x, layer12, s=100, color="crimson", marker="o")
p2 = ax2.plot(x, layer2, lw = 4, color="forestgreen", linestyle="solid", label=r"$\mathrm{Depth~2}$")
p1 = ax2.plot(x, layer6, lw = 4, color="royalblue", linestyle="dashed", label=r"$\mathrm{Depth~6}$")
p3 = ax2.plot(x, layer12, lw = 4, color="crimson", linestyle="dashdot", label=r"$\mathrm{Depth~12}$")

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
# ax2.set_ylim((-3, 103))
# ax.set_xlim((-2.5, 3.5))

ax2.set_title(r'$\mathrm{Insertion~sort~}(m=5)$', fontsize = 32)
ax2.tick_params(labelsize=32)
ax2.grid(ls=':', lw=0.8)
plt.legend(fontsize=28)

plt.tight_layout()
plt.savefig("./figures/sample_complexity_insertion_sort_various_layers.pdf", format="pdf", dpi=1200)
plt.show()
