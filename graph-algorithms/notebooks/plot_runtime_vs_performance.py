# %%
import matplotlib.pyplot as plt
import numpy as np

import matplotlib as mpl

from matplotlib import rc
rc('font', **{'family':'sans-serif','sans-serif':['Helvetica']})
mpl.rcParams['savefig.dpi'] = 1200
mpl.rcParams['text.usetex'] = True  # not really needed


# Sample data points
x = [
30.476,
19.67087942, 
107.4831816	,
220.4783212	,
148.5070977 ,
60.8, 
77.16741243	,
]

y = [
86.0,
83.6,
84.9,
85.0,
86.4,
85.6,
87.3,
]
x = np.array(x) # convert to numpy array
y = 100 - np.array(y) # compute error rate

labels = [
          r'$\mathrm{STN}$', 
          r'$\mathrm{MTN}$', 
          r'$\mathrm{MMoE}$', 
          r'$\mathrm{TAG}$',
          r'$\mathrm{LearningToBranch}$',
          r'$\mathrm{GradTAG}$',
          r'$\mathrm{AutoBRANE}$',]

# Create scatter plot
f, ax = plt.subplots(figsize=(8, 6))
ax.scatter(x[1:-1], y[1:-1], color='orange', s=600)
ax.scatter(x[-1], y[-1], color='forestgreen', s=600, marker='D')
# Label each point
for i, label in enumerate(labels):
    if i in [6]:
        ax.text(x[i]+14, y[i]-0.3, label, fontsize=32, ha='left', va='bottom')
    if i in [5]:
        ax.text(x[i]+14, y[i]-0.3, label, fontsize=32, ha='left', va='bottom')
    elif i in [4]:
        ax.text(x[i]+12, y[i]-0.3, label, fontsize=32, ha='left', va='bottom')
    elif i in [3]:
        ax.text(x[i]+12, y[i]+0.1, label, fontsize=32, ha='left', va='top')
    elif i in [2]:
        ax.text(x[i]+10, y[i]+0.1, label, fontsize=32, ha='left', va='bottom')
    elif i in [1]:
        ax.text(x[i]+6, y[i]+0.1, label, fontsize=32, ha='left', va='bottom')

# set x-axis to log scale
# plt.xscale('log')
plt.ylim(12.3, 17)
plt.yticks([13, 14, 15, 16, 17])
plt.xticks([0, 50, 100, 150, 200, 250])
plt.xlim(-10, 350)

plt.title(r'$\mathrm{Edge~Transformer}$', fontsize=40)
plt.xlabel(r'$\mathrm{GPU~Hours}$', fontsize=40)
plt.ylabel(r'$\mathrm{Error~\%}$', fontsize=40)
ax.tick_params(labelsize=40)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("./figures/plot_runtime_vs_performance.pdf", format="pdf", dpi=1200)
plt.show()

# %%
# %%
import matplotlib.pyplot as plt
import numpy as np

import matplotlib as mpl

from matplotlib import rc
rc('font', **{'family':'sans-serif','sans-serif':['Helvetica']})
mpl.rcParams['savefig.dpi'] = 1200
mpl.rcParams['text.usetex'] = True  # not really needed


# Sample data points
x = [
3.278479903,
3.33093461,
33.40580624,	
14.92821967,
21.83927907,
4.5286, 
10.86864964	
]

y = [
75.2,
71.6,
71.7,
73.1,
74.0,
73.5,
74.6,
]
x = np.array(x) # convert to numpy array
y = 100 - np.array(y) # compute error rate

labels = [
          r'$\mathrm{STN}$', 
          r'$\mathrm{MTN}$', 
          r'$\mathrm{MMoE}$', 
          r'$\mathrm{TAG}$',
          r'$\mathrm{LearningToBranch}$',
          r'$\mathrm{GradTAG}$',
          r'$\mathrm{AutoBRANE}$',]

# Create scatter plot
f, ax = plt.subplots(figsize=(8, 6))
ax.scatter(x[1:-1], y[1:-1], color='orange', s=600)
ax.scatter(x[-1], y[-1], color='forestgreen', s=600, marker='D')
# Label each point
for i, label in enumerate(labels):
    if i in [6]:
        ax.text(x[i]+4, y[i]-0.3, label, fontsize=32, ha='left', va='bottom')
    if i in [5]:
        ax.text(x[i]+4, y[i]-0.3, label, fontsize=32, ha='left', va='bottom')
    elif i in [4]:
        ax.text(x[i]+4, y[i]-0.3, label, fontsize=32, ha='left', va='bottom')
    elif i in [3]:
        ax.text(x[i]+4, y[i]+0.3, label, fontsize=32, ha='left', va='top')
    elif i in [2]:
        ax.text(x[i]+3, y[i]+0.1, label, fontsize=32, ha='left', va='bottom')
    elif i in [1]:
        ax.text(x[i]+3, y[i]+0.1, label, fontsize=32, ha='left', va='bottom')

# set x-axis to log scale
# plt.xscale('log')
plt.ylim(25, 29)
plt.yticks([25, 26, 27, 28, 29])
plt.xticks([0, 10, 20, 30, 40, 50])
plt.xlim(-10, 65)

plt.title(r'$\mathrm{MPNN}$', fontsize=40)
plt.xlabel(r'$\mathrm{GPU~Hours}$', fontsize=40)
plt.ylabel(r'$\mathrm{Error~\%}$', fontsize=40)
ax.tick_params(labelsize=40)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("./figures/plot_runtime_vs_performance_mpnn.pdf", format="pdf", dpi=1200)
plt.show()
