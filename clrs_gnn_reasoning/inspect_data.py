# %%
import clrs
import os
import os.path as osp

#     'insertion_sort', 'bubble_sort', 'heapsort', 'quicksort', 
#     'activity_selector', 'articulation_points', 'bellman_ford', 'bfs',
#     'binary_search', 'bridges', 'dag_shortest_paths', 'dfs', 'dijkstra',
#     'find_maximum_subarray_kadane', 'floyd_warshall', 'graham_scan', 
#     'jarvis_march', 'kmp_matcher', 'lcs_length', 'matrix_chain_order', 
#     'minimum', 'mst_kruskal', 'mst_prim', 'naive_string_matcher',
#     'optimal_bst', 'quickselect', 'segments_intersect', 'strongly_connected_components',
#     'task_scheduling', 'topological_sort',]:

# 'bfs', "dfs", "topological_sort", "articulation_points", "bridges", 
#        "strongly_connected_components", "mst_kruskal", "mst_prim", 
#        "dijkstra", "bellman_ford", 'dag_shortest_paths', "floyd_warshall"

for algorithm in ['bellman_ford']:

    train_ds, num_samples, spec = clrs.create_dataset(
        folder='./data/CLRS', algorithm=algorithm,
        split='train', batch_size=1)

    root_dir = "./data/CLRS/processed"
    processed_dir = osp.join(root_dir, algorithm)
    if not osp.exists(processed_dir):
        os.makedirs(processed_dir)

    for i, feedback in enumerate(train_ds.as_numpy_iterator()):
        features = feedback.features
        outputs = feedback.outputs
        inputs = features.inputs # inputs "key" and "pos"
        hints = features.hints
        lengths = features.lengths
        break
    print(algorithm, outputs)

# bfs (DataPoint(name="pi",	location=node,	type=pointer,	data=Array(1, 16)),)
# dfs (DataPoint(name="pi",	location=node,	type=pointer,	data=Array(1, 16)),)
# topological_sort (DataPoint(name="topo",	location=node,	type=pointer,	data=Array(1, 16)), DataPoint(name="topo_head",	location=node,	type=mask_one,	data=Array(1, 16)))
# articulation_points (DataPoint(name="is_cut",	location=node,	type=mask,	data=Array(1, 16)),)
# bridges (DataPoint(name="is_bridge",	location=edge,	type=mask,	data=Array(1, 16, 16)),)
# strongly_connected_components (DataPoint(name="scc_id",	location=node,	type=pointer,	data=Array(1, 16)),)
# mst_kruskal (DataPoint(name="in_mst",	location=edge,	type=mask,	data=Array(1, 16, 16)),)
# mst_prim (DataPoint(name="pi",	location=node,	type=pointer,	data=Array(1, 16)),)
# dijkstra (DataPoint(name="pi",	location=node,	type=pointer,	data=Array(1, 16)),)
# bellman_ford (DataPoint(name="pi",	location=node,	type=pointer,	data=Array(1, 16)),)
# dag_shortest_paths (DataPoint(name="pi",	location=node,	type=pointer,	data=Array(1, 16)),)
# floyd_warshall (DataPoint(name="Pi",	location=edge,	type=pointer,	data=Array(1, 16, 16)),)

# %%
# Compare the hints of two algorithms
algorithm_1 = "bfs"
train_ds_1, num_samples_1, spec_1 = clrs.create_dataset(
        folder='./data/CLRS', algorithm=algorithm_1,
        split='train', batch_size=1)

algorithm_2 = "bellman_ford"
train_ds_2, num_samples_2, spec_2 = clrs.create_dataset(
        folder='./data/CLRS', algorithm=algorithm_2,
        split='train', batch_size=1)

iter_1 = train_ds_1.as_numpy_iterator()
iter_2 = train_ds_2.as_numpy_iterator()
for i, feedback in enumerate(iter_1):
    features_1 = feedback.features
    hints_1 = feedback.features.hints

    features_2 = next(iter_2).features
    hints_2 = features_2.hints

    break