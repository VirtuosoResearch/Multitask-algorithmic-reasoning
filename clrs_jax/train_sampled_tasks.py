import random
import os

# List of all tasks
all_tasks = [
    "bfs", "dfs", "topological_sort", "articulation_points", "bridges",
    "strongly_connected_components", "mst_kruskal", "mst_prim",
    "dijkstra", "bellman_ford", "dag_shortest_paths"
]

# Set to store unique subsets
subsets = set()
while len(subsets) < 50:
    subset = tuple(sorted(random.sample(all_tasks, 4)))
    subsets.add(subset)

# Loop over each subset and run the command
for i, subset in enumerate(subsets):
    tasks_str = ",".join(f'"{task}"' for task in subset)
    command = (
        'CUDA_VISIBLE_DEVICES=1 python -m clrs.examples.run ' + f'--algorithms {tasks_str} ' + \
        '--use_projection --projection_dim 16 --processor_type "edge_t" --num_layers 5 --runs 1'
    )
    print(tasks_str)
    os.system(command)