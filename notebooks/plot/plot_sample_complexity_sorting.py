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

f, ax2 = plt.subplots(figsize=(6.5, 5))


single_accuracy = np.array([0, 68.34, 83.26, 94.82, 98.61, 99.64])

finetuning_accuracy = np.array([0, 88.91, 96.14, 98.00, 98.99, 99.66])
x = np.arange(len(single_accuracy))

p2 = ax2.plot(x, finetuning_accuracy, lw = 4, color="royalblue", linestyle="solid", label=r"$\mathrm{Fine}$-$\mathrm{tuning~from~}m=5$")
p1 = ax2.plot(x, single_accuracy, lw = 4, color="royalblue", linestyle="dashed", label=r"$\mathrm{Training~from~scratch}$")
# ax2.fill_between(
#     x, 
#     neg+neg_std,
#     neg-neg_std, 
#     color="orange", alpha=0.3
# )

ax2.set_ylabel(r'$\mathrm{Accuracy}~(\%)$', fontsize = 32)
ax2.set_xlabel(r'$\mathrm{\#~training~examples~}(m=10)$', fontsize = 32) # '+r'$\mathrm{~number~of~sampled~sets}
# ax.set_yticks(np.arange(0, 1.1, .2))
plt.xticks(x, [r'$0$', r'$1k$', r'$2k$', r'$5k$', r'$10k$', r'$20k$'], fontsize=32)
ax2.set_yticks(np.arange(0, 101, 20))
# ax2.set_ylim((-3, 10))
# ax.set_xlim((-2.5, 3.5))

ax2.set_title(r'$\mathrm{Sorting~}(m=10)$', fontsize = 32)
ax2.tick_params(labelsize=32)
ax2.grid(ls=':', lw=0.8)
plt.legend(fontsize=24)

plt.tight_layout()
plt.savefig("./figures/sample_complexity_sorting_len10.pdf", format="pdf", dpi=1200)
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

f, ax2 = plt.subplots(figsize=(6.5, 5))



back_accuracy = np.array([99.99, 70.07, 66.84, 66.95, 59.21, 51.82])
back_accuracy_std = np.array([0, 1.21, 0.75, 0.43, 0.23, 0.12])
x = np.arange(len(single_accuracy))

p1 = ax2.plot(x, back_accuracy, lw = 4, color="crimson", linestyle="solid") # label=r"$\mathrm{Training~from~scratch}$"
ax2.fill_between(
    x, 
    back_accuracy+back_accuracy_std,
    back_accuracy-back_accuracy_std, 
    color="crimson", alpha=0.3
)

ax2.set_ylabel(r'$\mathrm{Accuracy}~(\%)$', fontsize = 32)
ax2.set_xlabel(r'$\mathrm{\#~training~examples~}(m=10)$', fontsize = 32) # '+r'$\mathrm{~number~of~sampled~sets}
# ax.set_yticks(np.arange(0, 1.1, .2))
plt.xticks(x, [r'$0$', r'$1k$', r'$2k$', r'$5k$', r'$10k$', r'$20k$'], fontsize=32)
# ax2.set_yticks(np.arange(0, 101, 20))
# ax2.set_ylim((-13.5, 9.5))
# ax.set_xlim((-2.5, 3.5))

ax2.set_title(r'$\mathrm{Sorting~}(m=5)$', fontsize = 32)
ax2.tick_params(labelsize=32)
ax2.grid(ls=':', lw=0.8)
# plt.legend(fontsize=16)

plt.tight_layout()
plt.savefig("./figures/backward_accuracy_sorting_len10.pdf", format="pdf", dpi=1200)
plt.show()