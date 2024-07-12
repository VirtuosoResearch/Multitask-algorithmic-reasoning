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

training_loss = [5.616059,
5.282913,
4.732773,
4.286781,
3.907215,
3.528549,
3.119027,
2.700068,
2.322927,
2.013389,
1.765390,
1.571672,
1.406641,
1.273577,
1.158060,
1.059322,
0.973125,
0.900868,
0.851449,
0.816088,
0.777744,
0.749080,
0.714585,
0.688674,
0.662657,
0.636842,
0.609833,
0.588984,
0.564267,
0.539361,
0.522571,
0.501998,
0.483953,
0.466929,
0.453447,
0.437806,
0.420583,
0.404597,
0.387460,
0.375318,
0.362293,
0.352726,
0.337123,
0.323889,
0.315982,
0.303753,
0.294727,
0.285733,
0.273742,
0.265674,
0.258218,
0.248591,
0.240777,
0.232102,
0.227852,
0.220909,
0.213711,
0.208639,
0.201909,
0.194562,
0.190468,
0.180420,
0.178436,
0.173808,
0.168223,
0.161566,
0.157454,
0.153843,
0.147318,
0.143700,
0.141229,
0.136043,
0.134899,
0.130259,
0.126452,
0.125529,
0.119939,
0.115819,
0.114390,
0.112734,
0.112780,
0.106674,
0.103183,
0.101811,
0.100462,
0.099163,
0.096221,
0.095623,
0.092270,
0.088456,
0.088469,
0.087015,
0.084840,
0.083874,
0.081013,
0.081753,
0.077442,
0.075683,
0.076024,
0.073038]

ax2.plot(np.arange(len(training_loss)), training_loss, lw = 4, color="royalblue", linestyle="solid", label=r"$\mathrm{Training~loss}$")
ax2.set_xticks(np.arange(0, 101, 20))
ax2.set_yticks(np.arange(0, 7, 3))
ax2.set_ylim((-0.5, 6.5))
ax2.set_ylabel(r'$\mathrm{Training~loss}$', fontsize = 32)
ax2.set_xlabel(r'$\mathrm{Epochs}$', fontsize = 32)
ax2.set_title(r'$\mathrm{Bubble~sort}\rightarrow\mathrm{Insertion~sort}$', fontsize = 32)
ax2.tick_params(labelsize=32)
ax2.grid(ls=':', lw=0.8)
# plt.legend(fontsize=24, loc="lower right")

plt.tight_layout()
plt.savefig("./figures/plot_algorithm_convergence_1.pdf", format="pdf", dpi=1200)
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

training_loss = [3.913236,
3.805603,
3.609333,
3.357071,
3.096952,
2.824460,
2.575063,
2.337052,
2.120112,
1.915648,
1.728745,
1.552459,
1.393423,
1.245566,
1.116738,
0.998152,
0.892683,
0.835862,
0.769152,
0.735052,
0.70767,
0.67393,
0.64823,
0.61961,
0.59695,
0.57689,
0.55666,
0.54011,
0.52208,
0.50078,
0.483953,
0.466929,
0.453447,
0.437806,
0.420583,
0.404597,
0.387460,
0.375318,
0.362293,
0.352726,
0.337123,
0.323889,
0.315982,
0.303753,
0.294727,
0.285733,
0.273742,
0.265674,
0.258218,
0.248591,
0.240777,
0.232102,
0.227852,
0.220909,
0.213711,
0.208639,
0.201909,
0.194562,
0.190468,
0.180420,
0.178436,
0.173808,
0.168223,
0.161566,
0.157454,
0.153843,
0.147318,
0.143700,
0.141229,
0.136043,
0.134899,
0.130259,
0.126452,
0.125529,
0.119939,
0.115819,
0.114390,
0.112734,
0.112780,
0.106674,
0.103183,
0.101811,
0.100462,
0.099163,
0.096221,
0.095623,
0.092270,
0.088456,
0.088469,
0.087015,
0.084840,
0.083874,
0.081013,
0.081753,
0.077442,
0.075683,
0.076024,
0.073038]
training_loss=training_loss[:101]
ax2.plot(np.arange(len(training_loss)), training_loss, lw = 4, color="royalblue", linestyle="solid", label=r"$\mathrm{Training~loss}$")
ax2.set_xticks(np.arange(0, 101, 20))
ax2.set_yticks(np.arange(0, 5, 2))
ax2.set_ylim((-0.3, 4.3))
ax2.set_ylabel(r'$\mathrm{Training~loss}$', fontsize = 32)
ax2.set_xlabel(r'$\mathrm{Epochs}$', fontsize = 32)
ax2.set_title(r'$\mathrm{Bubble~sort~}m=5\rightarrow 10$', fontsize = 32)
ax2.tick_params(labelsize=32)
ax2.grid(ls=':', lw=0.8)
# plt.legend(fontsize=24, loc="lower right")

plt.tight_layout()
plt.savefig("./figures/plot_algorithm_convergence_2.pdf", format="pdf", dpi=1200)
plt.show()