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

output_accuracy = np.array([0, 78.34, 89.34, 98.57, 99.19, 99.69, 99.99, 99.99])

insertion_sort = np.array([0, 56.3680, 67.2002, 79.8100, 91.4673, 95.4097, 99.03, 99.71])
x = np.arange(len(output_accuracy))
ax2.scatter(x, output_accuracy, s=150, color="royalblue", marker="o")
ax2.scatter(x, insertion_sort, s=150, color="forestgreen", marker="o")
p2 = ax2.plot(x, output_accuracy, lw = 4, color="royalblue", linestyle="dashed", label=r"$\mathrm{Output~only}$")
p1 = ax2.plot(x, insertion_sort, lw = 4, color="forestgreen", linestyle="solid", label=r"$\mathrm{With~intermediate}$")

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
plt.savefig("./figures/sample_complexity_insertion_sort.pdf", format="pdf", dpi=1200)
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

output_accuracy = np.array([0, 78.34, 89.34, 98.57, 99.19, 99.69, 99.99, 99.99])

quick_sort = np.array([0, 56.33, 66.24, 79.36, 88.55, 93.73, 95.82, 99.71])
x = np.arange(len(output_accuracy))
ax2.scatter(x, output_accuracy, s=150, color="royalblue", marker="o")
ax2.scatter(x, quick_sort, s=150, color="forestgreen", marker="o")
p2 = ax2.plot(x, output_accuracy, lw = 4, color="royalblue", linestyle="dashed", label=r"$\mathrm{Output~only}$")
p1 = ax2.plot(x, quick_sort, lw = 4, color="forestgreen", linestyle="solid", label=r"$\mathrm{With~intermediate}$")

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

ax2.set_title(r'$\mathrm{Quick~sort}~(m=5)$', fontsize = 32)
ax2.tick_params(labelsize=32)
ax2.grid(ls=':', lw=0.8)
plt.legend(fontsize=28, loc="lower right")

plt.tight_layout()
plt.savefig("./figures/sample_complexity_quick_sort.pdf", format="pdf", dpi=1200)
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

output_accuracy = np.array([0, 78.34, 89.34, 98.57, 99.19, 99.69, 99.99, 99.99])

bubble_sort = np.array([0, 59.99385939, 72.18408953, 86.84, 93.40, 95.44, 99.42, 99.71])
x = np.arange(len(output_accuracy))
ax2.scatter(x, output_accuracy, s=150, color="royalblue", marker="o")
ax2.scatter(x, bubble_sort, s=150, color="forestgreen", marker="o")
p2 = ax2.plot(x, output_accuracy, lw = 4, color="royalblue", linestyle="dashed", label=r"$\mathrm{Output~only}$")
p1 = ax2.plot(x, bubble_sort, lw = 4, color="forestgreen", linestyle="solid", label=r"$\mathrm{With~intermediate}$")

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

ax2.set_title(r'$\mathrm{Bubble~sort~}(m=5)$', fontsize = 32)
ax2.tick_params(labelsize=32)
ax2.grid(ls=':', lw=0.8)
plt.legend(fontsize=28, loc="lower right")

plt.tight_layout()
plt.savefig("./figures/sample_complexity_bubble_sort.pdf", format="pdf", dpi=1200)
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

output_accuracy = np.array([0, 78.34, 89.34, 98.57, 99.19, 99.69, 99.99, 99.99])

selection_sort = np.array([0, 61.15725626, 70.89675565, 81.90, 88.39, 93.64, 98.64, 99.71])
x = np.arange(len(output_accuracy))
ax2.scatter(x, output_accuracy, s=150, color="royalblue", marker="o")
ax2.scatter(x, selection_sort, s=150, color="forestgreen", marker="o")
p2 = ax2.plot(x, output_accuracy, lw = 4, color="royalblue", linestyle="dashed", label=r"$\mathrm{Output~only}$")
p1 = ax2.plot(x, selection_sort, lw = 4, color="forestgreen", linestyle="solid", label=r"$\mathrm{With~intermediate}$")

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

ax2.set_title(r'$\mathrm{Selection~sort~}(m=5)$', fontsize = 32)
ax2.tick_params(labelsize=32)
ax2.grid(ls=':', lw=0.8)
plt.legend(fontsize=28, loc="lower right")

plt.tight_layout()
plt.savefig("./figures/sample_complexity_selection_sort.pdf", format="pdf", dpi=1200)
plt.show()