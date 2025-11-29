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

f, ax2 = plt.subplots(figsize=(6, 5))


w_steps = 100-np.array([
(32.7500 + 31.1500)/2,
(38.7500 + 41.4500)/2,
(65.8000 + 59.7500)/2,
(81.2500 + 74.0500)/2,
(88.6500 + 79.3500)/2,
(94.3000 + 84.9015)/2,
])
w_steps_std = np.array([
(0.2500 + 0.1500)/2,
(0.3500 + 1.2500)/2,
(0.7500 + 0.2000)/2,
(1.8500 + 0.4500)/2,
(2.2500 + 2.5500)/2,
(1.2500 + 0.3000)/2,
])
wo_steps = 100-np.array([
(35.3500 + 32.2500)/2,
(35.5000 + 36.2500)/2,
(46.2000 + 52.0000)/2,
(66.4000 + 77.4500)/2,
(74.5000 + 81.7000)/2,
(76.9500 + 89.1000)/2,
])
wo_steps_std = np.array([
(2.7500 + 1.2500)/2,
(1.6000 + 0.5500)/2,
(4.2000 + 3.5000)/2,
(3.0000 + 0.9500)/2,
(0.9000 + 0.9000)/2,
(1.1500 + 1.7000)/2,
])
x = np.arange(len(w_steps))
for i in range(len(x)):
    scatter2 = ax2.scatter(x[i], w_steps[i], s=200, marker="D", edgecolors = "none", facecolors='orange')

for i in range(len(x)):
    scatter1 = ax2.scatter(x[i], wo_steps[i], s=200, marker="o", edgecolors = "none", facecolors='royalblue')


p1 = ax2.plot(x, wo_steps, lw = 4, color="royalblue", linestyle="dashed", label=r"$\mathrm{Trained~on~each~dataset~alone}$")
ax2.fill_between(
    x, 
    wo_steps+wo_steps_std,
    wo_steps-wo_steps_std, 
    color="royalblue", alpha=0.3
)

p2 = ax2.plot(x, w_steps, lw = 4, color="orange", linestyle="solid", label=r"$\mathrm{Trained~on~the~combined~dataset}$")
ax2.fill_between(
    x, 
    w_steps+w_steps_std,
    w_steps-w_steps_std, 
    color="orange", alpha=0.3
)

ax2.set_ylabel(r'$\mathrm{Error~rates}~(\%)$', fontsize =28)
ax2.set_xlabel(r'$\mathrm{Number~of~samples}$', fontsize =28) # '+r'$\mathrm{~number~of~sampled~sets}
plt.xticks(x, [r"$100$", r"$200$", r"$500$", r"$1000$", r"$2000$", r"$5000$"])
ax2.set_yticks(np.arange(20, 81, 20))
ax2.set_ylim((6, 95))
# ax.set_xlim((-2.5, 3.5))

ax2.tick_params(labelsize=28)
ax2.grid(ls=':', lw=0.8)
# plt.title(r"$\mathrm{Evaluated~on~Bellman}$" + "-"  + r"$\mathrm{Ford~and~BFS}$", fontsize=28)
plt.legend(fontsize=19)

plt.tight_layout()
plt.savefig("./figures/plot_compare_combining_bellman_ford_and_bfs.pdf", format="pdf", dpi=1200)
plt.show()

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

f, ax2 = plt.subplots(figsize=(6, 5))


w_steps = 100-np.array([
(19.7000 + 24.4000)/2,
(22.8000 + 20.7750)/2,
(27.8000 + 19.9500)/2,
(31.9000 + 32.8500)/2,
(30.2000 + 40.8000)/2,
(34.5000 + 44.2000)/2,
])
w_steps_std = np.array([
(2.6000 + 0.6300)/2,
(1.7000 + 1.6250)/2,
(0.2000 + 1.4000)/2,
(0.0000 + 0.1500)/2,
(1.9000 + 5.5000)/2,
(1.6000 + 4.2000)/2,
])
wo_steps = 100-np.array([
(35.3500 + 30.5565)/2, 
(35.5000 + 34.9500)/2, 
(46.2000 + 50.5500)/2, 
(66.4000 + 89.9000)/2, 
(74.5000 + 93.5500)/2, 
(76.9500 + 99.1500)/2, 
])
wo_steps_std = np.array([
(2.7500 + 0.1546)/2,
(1.6000 + 0.5500)/2,
(4.2000 + 3.9500)/2,
(3.0000 + 1.1000)/2,
(0.9000 + 0.2500)/2,
(1.1500 + 0.1500)/2,
])
x = np.arange(len(w_steps))
for i in range(len(x)):
    scatter2 = ax2.scatter(x[i], w_steps[i], s=200, marker="D", edgecolors = "none", facecolors='orange')

for i in range(len(x)):
    scatter1 = ax2.scatter(x[i], wo_steps[i], s=200, marker="o", edgecolors = "none", facecolors='royalblue')


p1 = ax2.plot(x, wo_steps, lw = 4, color="royalblue", linestyle="dashed", label=r"$\mathrm{Trained~on~each~dataset~alone}$")
ax2.fill_between(
    x, 
    wo_steps+wo_steps_std,
    wo_steps-wo_steps_std, 
    color="royalblue", alpha=0.3
)

p2 = ax2.plot(x, w_steps, lw = 4, color="orange", linestyle="solid", label=r"$\mathrm{Trained~on~the~combined~dataset}$")
ax2.fill_between(
    x, 
    w_steps+w_steps_std,
    w_steps-w_steps_std, 
    color="orange", alpha=0.3
)

ax2.set_ylabel(r'$\mathrm{Error~rates}~(\%)$', fontsize =28)
ax2.set_xlabel(r'$\mathrm{Number~of~samples}$', fontsize =28) # '+r'$\mathrm{~number~of~sampled~sets}
plt.xticks(x, [r"$100$", r"$200$", r"$500$", r"$1000$", r"$2000$", r"$5000$"])
ax2.set_yticks(np.arange(20, 81, 20))
ax2.set_ylim((6, 110))
# ax.set_xlim((-2.5, 3.5))

ax2.tick_params(labelsize=28)
ax2.grid(ls=':', lw=0.8)
# plt.title(r"$\mathrm{Evaluated~on~Bellman}$" + "-"  + r"$\mathrm{Ford~and~DFS}$", fontsize=28)
plt.legend(fontsize=19)

plt.tight_layout()
plt.savefig("./figures/plot_compare_combining_bellman_ford_and_dfs.pdf", format="pdf", dpi=1200)
plt.show()