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


single_accuracy = np.array([0, 22.44, 58.83, 87.55, 99.99, 99.99, 99.99, 99.99])

finetuning_accuracy = np.array([0, 43.96, 92.04, 99.76, 99.9, 99.99, 99.99, 99.99])
x = np.arange(len(single_accuracy))

ax2.scatter(x, single_accuracy, s=150, color="royalblue", marker="o")
ax2.scatter(x, finetuning_accuracy, s=150, color="forestgreen", marker="o")
p2 = ax2.plot(x, finetuning_accuracy, lw = 4, color="forestgreen", linestyle="solid", label=r"$\mathrm{Fine}$-$\mathrm{tuning~from~}m=5$")
p1 = ax2.plot(x, single_accuracy, lw = 4, color="royalblue", linestyle="dashed", label=r"$\mathrm{Training~from~scratch}$")
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

ax2.set_title(r'$\mathrm{Addition~}(m=10)$', fontsize = 32)
ax2.tick_params(labelsize=32)
ax2.grid(ls=':', lw=0.8)
plt.legend(fontsize=24, loc="lower right")

plt.tight_layout()
plt.savefig("./figures/sample_complexity_addition_len10.pdf", format="pdf", dpi=1200)
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

f, ax2 = plt.subplots(figsize=(6.8, 5))


ours =  np.array([99.99, 44.9562, 34.1057, 32.64, 32.71, 31.5, 30.9, 30.1,])
back_accuracy = np.array([99.99, 24.9562, 14.1057, 12.64, 12.71, 12.5, 12.9, 12.1,])
back_accuracy_std = np.array([0, 1.21, 0.75, 0.43, 0.23, 0.12, 0.23, 0.12])
x = np.arange(len(single_accuracy))
ax2.scatter(x, ours, s=150, color="forestgreen", marker="o")
ax2.scatter(x, back_accuracy, s=150, color="royalblue", marker="o")
p1 = ax2.plot(x, ours, lw = 4, color="forestgreen", linestyle="solid", label=r"$\mathrm{Our~method}$") # label=r"$\mathrm{Fine}$-$\mathrm{tuning~on~}m=10$" 
p1 = ax2.plot(x, back_accuracy, lw = 4, color="royalblue", linestyle="dashed", label=r"$\mathrm{Fine}$" + r"{-}" + r"$\mathrm{tuning}$") #  
# ax2.fill_between(
#     x, 
#     back_accuracy+back_accuracy_std,
#     back_accuracy-back_accuracy_std, 
#     color="crimson", alpha=0.3
# )

ax2.set_ylabel(r'$\mathrm{Accuracy}~(\%)$', fontsize = 32)
ax2.set_xlabel(r'$\mathrm{\#~training~examples}$', fontsize = 32) # '+r'$\mathrm{~number~of~sampled~sets}
# ax.set_yticks(np.arange(0, 1.1, .2))
plt.xticks(x, [r'$0$', r'$10^3$', "", "", r'$10^4$', "", "", r'$10^5$'], fontsize=32)
ax2.set_yticks(np.arange(0, 101, 20))
# ax2.set_ylim((-12, 105))
# ax.set_xlim((-2.5, 3.5))

ax2.set_title(r'$\mathrm{Addition~}(m=5)$', fontsize = 32)
ax2.tick_params(labelsize=32)
ax2.grid(ls=':', lw=0.8)
plt.legend(fontsize=28)

plt.tight_layout()
plt.savefig("./figures/backward_accuracy_addition_len5.pdf", format="pdf", dpi=1200)
plt.show()
# %%
