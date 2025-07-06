# %%
import matplotlib.pyplot as plt
import numpy as np

import matplotlib as mpl

from matplotlib import rc
rc('font', **{'family':'sans-serif','sans-serif':['Helvetica']})
mpl.rcParams['savefig.dpi'] = 1200
mpl.rcParams['text.usetex'] = True  # not really needed

x = np.arange(1, 6)
naive_mtl = np.array([0.6946, 0.7101, 0.7147, 0.7162, 0.7180])

task_grouping = np.array([0.6946, 0.7267, 0.7404, 0.7527, 0.7478])

# mtft = (3863 + (x-2)*772)/1024
# ours = (3863 + np.array([3, 4, 6, 8, 10])*772)/1024

# Create scatter plot
f, ax = plt.subplots(figsize=(7, 5))

scatter1 = ax.scatter([5], [0.7478], color='orange', s=400,  marker='h', label=r'$\mathrm{Single~task~learning}$')

scatter1 = ax.scatter(x[:4], task_grouping[:4], color='royalblue', s=200,  marker='o')
ax.plot(x, task_grouping, color='royalblue', linewidth=4, ls='--', label=r'$\mathrm{MTL~models~on~task~groups}$')


scatter1 = ax.scatter(x, naive_mtl, color='tomato', s=200,  marker='s')
ax.plot(x, naive_mtl, color='tomato', linewidth=4, ls='-.', label=r'$\mathrm{Single~MTL~model}$')

# ax.set_ylim(0.4, 5.6)
# # ax.set_yticks(np.arange(0, 170, 40))
# plt.xticks([4.91998093, 5.71342875, 6.90306546, 8.98719682],
#            [r"$\mathrm{0.1}$", r"$\mathrm{0.3}$", r"$\mathrm{1}$", r"$\mathrm{8}$"])
# plt.yticks([1, 2, 3, 4, 5 ],
#            [r"$\mathrm{3}$", r"$\mathrm{6}$", r"$\mathrm{20}$", r"$\mathrm{60}$", r"$\mathrm{120}$"])

plt.legend(fontsize=22,)
plt.xticks([1, 2, 3, 4, 5], [r"$\mathrm{1}$", r"$\mathrm{2}$", r"$\mathrm{4}$",  r"$\mathrm{8}$", r"$\mathrm{12}$"])
plt.yticks(np.arange(0.66, 0.762, 0.04))
plt.ylim(0.657, 0.763)
ax.set_xlabel(r'$\mathrm{Model~memory}$', fontsize=32)
ax.set_ylabel(r'$\mathrm{Avg.~accuracy}$', fontsize=32)
ax.tick_params(labelsize=32)

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("./figures/plot_performance_memory_tradeoff.pdf", format="pdf", dpi=1200)
plt.show()