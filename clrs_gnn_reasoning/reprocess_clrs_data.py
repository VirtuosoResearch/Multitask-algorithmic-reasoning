# %%
import torch
import clrs
from torch_geometric.data import Data
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from scipy.sparse import coo_matrix
import numpy as np
import networkx as nx

class CLRSData(Data):
    """A data object for CLRS data."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

def to_torch(value):
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    elif isinstance(value, torch.Tensor):
        return value
    else:
        return torch.tensor(value)
    
def infer_type(dp_type, data):
    return data.astype(np.float32) # convert to float32

def pointer_to_one_hot(pointer, n):
    """Convert a pointer to a one-hot vector."""
    return (np.arange(n) == pointer.reshape(-1, 1)).astype(float)

def to_data(inputs, hints, outputs, use_hints=True):
    data_dict = {}
    input_attributes = []
    hint_attributes = []
    output_attributes = []
    data_dict['length'] = inputs[0].data[0].shape[0]
    
    # first get the edge index; create a fully connected graph 
    input_keywords = [dp.name for dp in inputs]
    if "adj" in input_keywords:
        graph = nx.from_numpy_array(inputs[input_keywords.index("adj")].data[0])
    else:
        graph = nx.complete_graph(data_dict['length'])
    data_dict['edge_index'] = torch.tensor(np.array(list(graph.edges())).T, dtype=torch.long)

    # Parse inputs
    for dp in inputs:
        if dp.name == "adj":
            continue
        elif dp.name == "A":
            # add self loops
            unique_values = np.unique(dp.data[0])
            is_weighted = unique_values.size != 2 or not np.all(unique_values == np.array([0,1]))
            if is_weighted:
                data_dict["weights"] = infer_type("A", (dp.data[0] + np.eye(dp.data[0].shape[0]))[data_dict["edge_index"][0], data_dict["edge_index"][1]])
        elif dp.location == clrs.Location.EDGE:
            data_dict[dp.name] = infer_type(dp.type_, dp.data[0][data_dict["edge_index"][0], data_dict["edge_index"][1]])
            input_attributes.append(dp.name)
        elif dp.location == clrs.Location.NODE:
            if dp.type_ == clrs.Type.POINTER:
                # Convert pointers to one-hot edge masks
                n = dp.data[0].shape[0]
                pointer_matrix = pointer_to_one_hot(dp.data[0], n)
                data_dict[dp.name] = pointer_matrix[data_dict["edge_index"][0], data_dict["edge_index"][1]]
            else:
                data_dict[dp.name] = infer_type(dp.type_, dp.data[0])
            input_attributes.append(dp.name)
        else: # Graph
            data_dict[dp.name] = infer_type(dp.type_, dp.data[0])
    # Parse outputs
    for dp in outputs:
        output_attributes.append(dp.name)
        if dp.location == clrs.Location.EDGE:
            data_dict[dp.name] = infer_type(dp.type_, dp.data[0][data_dict["edge_index"][0], data_dict["edge_index"][1]])
        elif dp.location == clrs.Location.NODE:
            if dp.type_ == clrs.Type.POINTER:
                # Convert pointers to one-hot edge masks
                n = dp.data[0].shape[0]
                pointer_matrix = pointer_to_one_hot(dp.data[0], n)
                data_dict[dp.name] = pointer_matrix[data_dict["edge_index"][0], data_dict["edge_index"][1]]
            else:
                data_dict[dp.name] = infer_type(dp.type_, dp.data[0])
        else: # Graph
            data_dict[dp.name] = infer_type(dp.type_, dp.data[0])
    if use_hints:
        # Parse hints
        for dp in hints:
            hint_attributes.append(dp.name)
            if dp.location == clrs.Location.EDGE or (dp.location == clrs.Location.NODE and dp.type_ == clrs.Type.POINTER):
                arr = dp.data.squeeze(1) # Hints, N, N, D (...)
                if dp.location == clrs.Location.NODE:
                    # arr is Hints, N, D (...)
                    # Convert pointers to one-hot edge masks
                    stages = []
                    for hd in range(arr.shape[0]):
                        n = arr.shape[1]
                        pointer_matrix = pointer_to_one_hot(arr[hd], n)
                        stages.append(pointer_matrix)
                    
                    arr = np.stack(stages, axis=0)
                
                # Determine the number of dimensions of the array
                num_dims = arr.ndim
                transpose_indices = tuple(range(num_dims))
                transpose_indices = (1, 2, 0) + transpose_indices[3:]
                data_dict[dp.name] = infer_type(dp.type_, arr.transpose(*transpose_indices)[data_dict["edge_index"][0], data_dict["edge_index"][1]])
            elif dp.location == clrs.Location.NODE and not dp.type_ == clrs.Type.POINTER:
                arr = dp.data.squeeze(1) # Hints, N, D (...)
                # Determine the number of dimensions of the array
                num_dims = arr.ndim
                # Create a tuple of indices to swap the first two dimensions
                transpose_indices = tuple(range(num_dims))
                transpose_indices = (1, 0) + transpose_indices[2:]
                data_dict[dp.name] = infer_type(dp.type_, arr.transpose(*transpose_indices))
            else:
                data_dict[dp.name] = infer_type(dp.type_, dp.data.squeeze(1)[np.newaxis, ...])

        
    data_dict = {k: to_torch(v) for k,v in data_dict.items()}
    data = CLRSData(**data_dict)    
    data.hints = hint_attributes
    data.inputs = input_attributes
    data.outputs = output_attributes
    return data


# %%
import clrs
import os
import os.path as osp

for algorithm in [
    'insertion_sort', 'bubble_sort', 'heapsort', 'quicksort', 
    'activity_selector', 'articulation_points', 'bellman_ford', 'bfs',
    'binary_search', 'bridges', 'dag_shortest_paths', 'dfs', 'dijkstra',
    'find_maximum_subarray_kadane', 'floyd_warshall', 'graham_scan', 
    'jarvis_march', 'kmp_matcher', 'lcs_length', 'matrix_chain_order', 
    'minimum', 'mst_kruskal', 'mst_prim', 'naive_string_matcher',
    'optimal_bst', 'quickselect', 'segments_intersect', 'strongly_connected_components',
    'task_scheduling', 'topological_sort',]:

    train_ds, num_samples, spec = clrs.create_dataset(
        folder='./data/CLRS', algorithm=algorithm,
        split='train', batch_size=1)
    
    root_dir = "./data/CLRS/processed"
    processed_dir = osp.join(root_dir, algorithm)
    if not osp.exists(processed_dir):
        os.makedirs(processed_dir)

    for i, feedback in enumerate(train_ds.as_numpy_iterator()):
        if i >= num_samples:
            break
        features = feedback.features
        outputs = feedback.outputs
        inputs = features.inputs # inputs "key" and "pos"
        hints = features.hints
        lengths = features.lengths
        
        data = to_data(inputs, hints, outputs)
        # print(algorithm, data)
        torch.save(data, osp.join(processed_dir, f'data_{i}.pt'))
# %%

for algorithm in [
    'insertion_sort', 'bubble_sort', 'heapsort', 'quicksort', 
    'activity_selector', 'articulation_points', 'bellman_ford', 'bfs',
    'binary_search', 'bridges', 'dag_shortest_paths', 'dfs', 'dijkstra',
    'find_maximum_subarray_kadane', 'floyd_warshall', 'graham_scan', 
    'jarvis_march', 'kmp_matcher', 'lcs_length', 'matrix_chain_order', 
    'minimum', 'mst_kruskal', 'mst_prim', 'naive_string_matcher',
    'optimal_bst', 'quickselect', 'segments_intersect', 'strongly_connected_components',
    'task_scheduling', 'topological_sort',]:

    train_ds, num_samples, spec = clrs.create_dataset(
        folder='./data/CLRS', algorithm=algorithm,
        split='eval', batch_size=1)
    
    root_dir = "./data/CLRS/processed_eval"
    processed_dir = osp.join(root_dir, algorithm)
    if not osp.exists(processed_dir):
        os.makedirs(processed_dir)

    for i, feedback in enumerate(train_ds.as_numpy_iterator()):
        if i >= num_samples:
            break
        features = feedback.features
        outputs = feedback.outputs
        inputs = features.inputs # inputs "key" and "pos"
        hints = features.hints
        lengths = features.lengths
        
        data = to_data(inputs, hints, outputs)
        # print(algorithm, data)
        torch.save(data, osp.join(processed_dir, f'data_{i}.pt'))

for algorithm in [
    'insertion_sort', 'bubble_sort', 'heapsort', 'quicksort', 
    'activity_selector', 'articulation_points', 'bellman_ford', 'bfs',
    'binary_search', 'bridges', 'dag_shortest_paths', 'dfs', 'dijkstra',
    'find_maximum_subarray_kadane', 'floyd_warshall', 'graham_scan', 
    'jarvis_march', 'kmp_matcher', 'lcs_length', 'matrix_chain_order', 
    'minimum', 'mst_kruskal', 'mst_prim', 'naive_string_matcher',
    'optimal_bst', 'quickselect', 'segments_intersect', 'strongly_connected_components',
    'task_scheduling', 'topological_sort',]:

    train_ds, num_samples, spec = clrs.create_dataset(
        folder='./data/CLRS', algorithm=algorithm,
        split='test', batch_size=1)
    
    root_dir = "./data/CLRS/processed_test"
    processed_dir = osp.join(root_dir, algorithm)
    if not osp.exists(processed_dir):
        os.makedirs(processed_dir)

    for i, feedback in enumerate(train_ds.as_numpy_iterator()):
        if i >= num_samples:
            break
        features = feedback.features
        outputs = feedback.outputs
        inputs = features.inputs # inputs "key" and "pos"
        hints = features.hints
        lengths = features.lengths
        
        data = to_data(inputs, hints, outputs)
        # print(algorithm, data)
        torch.save(data, osp.join(processed_dir, f'data_{i}.pt'))