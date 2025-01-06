# %%
import matplotlib as mpl
from matplotlib import rc

import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
mpl.rcParams['savefig.dpi'] = 1200
mpl.rcParams['text.usetex'] = True  # not really needed

# List of elements
elements = [
    "BFS", "DFS", "Topological sort", "SCC", "DAG shortest paths",
    "Articulation points", "Bridges", "MST Prim", "Floyd warshall",
    "MST Kruskal", "Dijkstra", "Bellman Ford"
]

# Define the manual linkage matrix for predefined clusters
linkage_matrix = [
    [2, 4, 1.0, 2],
    [0, 12, 1.0, 3],
    [1, 13, 1.5, 4],
    [3, 14, 1.5, 4], # -->15
    [5, 6, 1.0, 2],
    [7, 16, 1.0, 3],
    [8, 17, 1.0, 4], # -->18
    [11, 9, 1.0, 2],
    [10, 19, 1.5, 3], # -->20
    [18, 20, 3, 8], # -->21
    [15, 21, 3, 12]
]

# Plot the dendrogram
plt.figure(figsize=(10, 7))
dendrogram(linkage_matrix, labels=elements, leaf_rotation=90, leaf_font_size=28)
plt.yticks([])
plt.tight_layout()

plt.savefig('./figures/tree_structure_clrs_tasks.pdf', format='pdf', dpi=100)
plt.show()
