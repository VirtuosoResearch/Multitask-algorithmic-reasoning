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

f, ax2 = plt.subplots(figsize=(8, 4))


w_steps = 100-np.array([
35.3500, 
35.5000, 
46.2000, 
66.4000, 
77.5000, 
79.9500, 
])
w_steps_std = np.array([
2.7500,
1.6000,
4.2000,
3.0000,
0.9000,
1.1500,
])
wo_steps = 100-np.array([
39.9375,
52.9375,
62.6875,
68.1250,
71.3125,
72.6250,
])
wo_steps_std = np.array([
0.0625,
0.0625,
0.6875,
1.3750,
0.4375,
0.3750,
])
x = np.arange(len(w_steps))
for i in range(len(x)):
    scatter2 = ax2.scatter(x[i], w_steps[i], s=200, marker="D", edgecolors = "none", facecolors='orange')

for i in range(len(x)):
    scatter1 = ax2.scatter(x[i], wo_steps[i], s=200, marker="o", edgecolors = "none", facecolors='royalblue')


p1 = ax2.plot(x, wo_steps, lw = 4, color="royalblue", linestyle="dashed", label=r"$\mathrm{Trained~on}$"+ r"$\mathrm{~the~last~step}$")
ax2.fill_between(
    x, 
    wo_steps+wo_steps_std,
    wo_steps-wo_steps_std, 
    color="royalblue", alpha=0.3
)


p2 = ax2.plot(x, w_steps, lw = 4, color="orange", linestyle="solid", label=r"$\mathrm{Trained~on}$" + r"$\mathrm{~all~steps}$")
ax2.fill_between(
    x, 
    w_steps+w_steps_std,
    w_steps-w_steps_std, 
    color="orange", alpha=0.3
)

ax2.set_ylabel(r'$\mathrm{Error~rates}~(\%)$', fontsize = 32)
ax2.set_xlabel(r'$\mathrm{Number~of~samples}$', fontsize = 32) # '+r'$\mathrm{~number~of~sampled~sets}
plt.xticks(x, [r"$100$", r"$200$", r"$500$", r"$1000$", r"$2000$", r"$5000$"])
# ax2.set_yticks(np.arange(-20, 81, 4))
ax2.set_ylim((16, 90))
# ax.set_xlim((-2.5, 3.5))

ax2.tick_params(labelsize=32)
ax2.grid(ls=':', lw=0.8)
# plt.title(r"$\mathrm{Evaluated~on~Bellman}$" + "-"  + r"$\mathrm{Ford}$", fontsize=32)
plt.legend(fontsize=24)

plt.tight_layout()
plt.savefig("./figures/plot_compare_w_and_wo_steps_bellman_ford.pdf", format="pdf", dpi=1200)
plt.show()