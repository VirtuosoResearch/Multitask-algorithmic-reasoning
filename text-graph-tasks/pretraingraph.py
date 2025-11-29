import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.data import DataLoader
from torch_geometric.data import Data

import numpy as np
from sklearn.cluster import KMeans
import random
import networkx as nx   

from src.model.gnn_models import GINEncoder, MultiTaskHeads, MultiTaskGIN

def pyg_data_from_nx(num_nodes, p, feature_dim=32):
    """
    Converts a NetworkX graph G to a PyTorch Geometric Data object.
    The node feature matrix (data.x) is set to all ones.
    
    Additionally, the following labels are attached:
      - graph_node_count: |V|
      - graph_edge_count: |E| (unique undirected edges)
      - node_degree: degree for each node
      - cycle_check: 1 if a cycle (of length ≥ 3) exists, else 0
      - triangle_count: number of triangles in G
      - edge_pair_tasks: a dictionary with keys:
            "pairs": tensor of shape [num_pairs, 2] containing unordered node pairs,
            "edge_existence": tensor of shape [num_pairs] (1 if edge exists, else 0),
            "connectivity": tensor of shape [num_pairs] (1 if a path exists, else 0),
            "shortest_path": tensor of shape [num_pairs] (shortest path length or -1)
    """
    num_nodes = 10
    p = 0.3
    G = nx.erdos_renyi_graph(n=num_nodes, p=p)
    # Ensure at least one edge exists (if necessary)
    if G.number_of_edges() == 0 and num_nodes > 1:
        u, v = np.random.choice(num_nodes, 2, replace=False)
        G.add_edge(u, v)
    n = G.number_of_nodes()
    if n == 0:
        return None

    # Create node features: all ones, shape [n, 1]
    x = torch.ones((n, feature_dim), dtype=torch.float)
    
    # Create edge_index: list all edges in both directions
    edges = list(G.edges())
    edge_index_list = []
    for u, v in edges:
        edge_index_list.append([u, v])
        edge_index_list.append([v, u])
    if len(edge_index_list) > 0:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    
    # Graph-level labels:
    graph_node_count = torch.tensor([n], dtype=torch.long)
    graph_edge_count = torch.tensor([G.number_of_edges()], dtype=torch.long)
    
    # Node-level label: degree for each node
    node_degrees = [d for _, d in G.degree()]
    node_degree = torch.tensor(node_degrees, dtype=torch.long)
    
    # Cycle check: use cycle_basis (each cycle has length >= 3)
    cycles = nx.cycle_basis(G)
    cycle_check = torch.tensor([1 if len(cycles) > 0 else 0], dtype=torch.long)
    
    # Triangle count: each triangle is counted at each vertex, so divide sum by 3.
    triangles_dict = nx.triangles(G)
    triangle_count = int(sum(triangles_dict.values()) / 3)
    triangle_count = torch.tensor([triangle_count], dtype=torch.float)
    
    # Edge-level tasks: Compute for all unordered pairs (u,v) with u < v.
    edge_existence = {}
    connectivity = {}
    shortest_path = {}
    nodes = list(G.nodes())
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            u, v = nodes[i], nodes[j]
            # Edge existence: 1 if the edge exists, 0 otherwise.
            edge_existence[(u, v)] = 1 if G.has_edge(u, v) else 0
            # Connectivity and shortest path:
            if nx.has_path(G, u, v):
                connectivity[(u, v)] = 1
                sp_length = nx.shortest_path_length(G, source=u, target=v)
                shortest_path[(u, v)] = sp_length
            else:
                connectivity[(u, v)] = 0
                shortest_path[(u, v)] = 0

    # Convert edge-level dictionaries to tensors.
    if len(edge_existence) > 0:
        pairs = torch.tensor(list(edge_existence.keys()), dtype=torch.long)
        edge_existence_tensor = torch.tensor(list(edge_existence.values()), dtype=torch.long)
        connectivity_tensor = torch.tensor(list(connectivity.values()), dtype=torch.long)
        shortest_path_tensor = torch.tensor(list(shortest_path.values()), dtype=torch.long)
    else:
        pairs = torch.empty((0, 2), dtype=torch.long)
        edge_existence_tensor = torch.empty((0,), dtype=torch.long)
        connectivity_tensor = torch.empty((0,), dtype=torch.long)
        shortest_path_tensor = torch.empty((0,), dtype=torch.long)
        
    edge_pair_tasks = {
        "pairs": pairs,
        "edge_existence": edge_existence_tensor,
        "connectivity": connectivity_tensor,
        "shortest_path": shortest_path_tensor
    }
    
    # Create the PyG Data object and attach the labels.
    data = Data(x=x, edge_index=edge_index)
    data.node_count = graph_node_count
    data.edge_count = graph_edge_count
    data.node_degree = node_degree
    data.cycle_check = cycle_check
    data.triangle_count = triangle_count
    data.edge_pair_tasks = edge_pair_tasks
    data.edge_existence = edge_existence_tensor
    data.connectivity = connectivity_tensor
    data.shortest_path = shortest_path_tensor
    
    return data

# =============================================================================
# Assume these are our task names.
# (They correspond to: node_count, edge_count, edge_existence, node_degree,
#  connectivity, cycle_check, shortest_path, triangle_count)
# =============================================================================
TASK_LIST = [
    "node_count", "edge_count", "edge_existence", "node_degree",
    "connectivity", "cycle_check", "shortest_path", "triangle_count"
]

# =============================================================================
# Example of a dictionary-based GINEncoder.
# (This version was provided in a previous answer.)
# =============================================================================


# =============================================================================
# Stub functions for training and evaluation.
# In a real implementation, these would include full training loops,
# loss computations (only for the active tasks) and evaluation metrics.
# =============================================================================
def compute_loss_for_tasks(outputs, data, active_tasks):
    # Stub: Sum the losses for each task in active_tasks.
    loss = 0.0
    # For example, if "node_count" is active, add its loss.
    for task in active_tasks:
        if task in outputs:
            if task == 'triangle_count':
                #print(outputs[task])
                #print(getattr(data, task))
                loss += F.mse_loss(outputs[task], getattr(data, task))
            else:
                #print(outputs[task].shape)
                #print(getattr(data, task).min())
                loss += F.cross_entropy(outputs[task], getattr(data, task))
            #loss += F.mse_loss(outputs[task], getattr(data, task))

    return loss

def train_model(model, train_loader, active_tasks, task_branch, num_epochs=10):

    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-3)
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for data in train_loader:
            optimizer.zero_grad()
            outputs = model(data, task_branch)
            loss = compute_loss_for_tasks(outputs, data, active_tasks)
            #print(outputs[active_tasks[0]])
            #print(data)
            #loss = F.cross_entropy(outputs[active_tasks[0]], getattr(data, active_tasks[0]))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {epoch_loss/len(train_loader):.4f}")
    # Return some evaluation metric (e.g., final loss) as a proxy.
    return epoch_loss / len(train_loader)

def evaluate_model(model, test_loader, active_tasks, task_branch):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for data in test_loader:
            outputs = model(data, task_branch)
            loss = compute_loss_for_tasks(outputs, data, active_tasks)
            total_loss += loss.item()
    return total_loss / len(test_loader)

# =============================================================================
# Pipeline Functions
# =============================================================================
def train_single_task(task, train_loader, test_loader):
    print(f"Training single task: {task}")
    # Create a model that uses only the branch for this task.
    # We use a model with a single encoder branch (specified by task).
    model = MultiTaskGIN(num_features=32, hidden_dim=32, num_layers=2, task_names=[task])
    train_loss = train_model(model, train_loader, active_tasks=[task], task_branch=task, num_epochs=10)
    test_loss = evaluate_model(model, test_loader, active_tasks=[task], task_branch=task)
    # Here, lower loss means better performance.
    print(f"Task {task}: Train Loss={train_loss:.4f}, Test Loss={test_loss:.4f}")
    return test_loss

def train_pair_tasks(task1, task2, train_loader, test_loader):
    print(f"Training task pair: {task1} & {task2}")
    # Create a model that has encoder branches for both tasks.
    # For simplicity, we assume the two tasks share the same encoder branch.
    # (You might decide to use separate branches and combine later.)
    model = MultiTaskGIN(num_features=32, hidden_dim=32, num_layers=2, task_names=[task1, task2])
    active_tasks = [task1, task2]
    # Here, we arbitrarily choose to use the branch for task1 as the encoder.
    train_loss = train_model(model, train_loader, active_tasks, task_branch=task1, num_epochs=10)
    test_loss = evaluate_model(model, test_loader, active_tasks, task_branch=task1)
    print(f"Task pair ({task1}, {task2}): Train Loss={train_loss:.4f}, Test Loss={test_loss:.4f}")
    return test_loss  # Use test loss as a proxy performance metric

def pipeline(train_loader, test_loader):
    # Step 1: Train each task separately.
    single_task_perf = {}
    for task in TASK_LIST:
        perf = train_single_task(task, train_loader, test_loader)
        single_task_perf[task] = perf
    print("Single task performance:", single_task_perf)

    # Step 2: Train every pair of tasks.
    n = len(TASK_LIST)
    pair_matrix = np.zeros((n, n))
    for i, t1 in enumerate(TASK_LIST):
        for j, t2 in enumerate(TASK_LIST):
            if i == j:
                pair_matrix[i, j] = single_task_perf[t1]
            elif i < j:
                perf_pair = train_pair_tasks(t1, t2, train_loader, test_loader)
                pair_matrix[i, j] = perf_pair
                pair_matrix[j, i] = perf_pair  # symmetric matrix
    print("Pair performance matrix:\n", pair_matrix)

    # Step 3: Cluster the tasks based on the pair matrix.
    # Here we use k-means clustering (number of clusters chosen arbitrarily).
    num_clusters = 3
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(pair_matrix)
    labels = kmeans.labels_
    clusters = {}
    for idx, task in enumerate(TASK_LIST):
        clusters.setdefault(labels[idx], []).append(task)
    print("Clustered task groups:", clusters)

    # Step 4: For each cluster, train a single model with a dict of branches for the tasks in that cluster.
    group_models = {}
    for cluster_id, tasks in clusters.items():
        print(f"Training model for task group {cluster_id}: {tasks}")
        # Create a model that includes branches for each task in the group.
        model = MultiTaskGIN(num_features=4, hidden_dim=32, num_layers=3, task_names=tasks)
        # Train the model with losses computed only for the tasks in this group.
        _ = train_model(model, train_loader, active_tasks=tasks, task_branch=tasks[0], num_epochs=10)
        test_perf = evaluate_model(model, test_loader, active_tasks=tasks, task_branch=tasks[0])
        print(f"Group {cluster_id} test performance: {test_perf:.4f}")
        group_models[cluster_id] = model

    return group_models

def train_node_count(train_loader, test_loader, feature_dim):
    model = MultiTaskGIN(num_features=feature_dim, hidden_dim=32, num_layers=2, task_names=["node_count"])
    train_loss = train_model(model, train_loader, active_tasks=["node_count"], task_branch="node_count", num_epochs=10)
    test_loss = evaluate_model(model, test_loader, active_tasks=["node_count"], task_branch="node_count")
    print(f"Task node_count: Train Loss={train_loss:.4f}, Test Loss={test_loss:.4f}")
    return test_loss

def train_single_task_pipeline(task_list, train_loader, test_loader):
    model = MultiTaskGIN(num_features=32, hidden_dim=32, num_layers=2, task_names=TASK_LIST)
    for task in task_list:
        print(f"Training single task: {task}")
        
        train_loss = train_model(model, train_loader, active_tasks=[task], task_branch=task, num_epochs=10)
        test_loss = evaluate_model(model, test_loader, active_tasks=[task], task_branch=task)
        print(f"Task {task}: Train Loss={train_loss:.4f}, Test Loss={test_loss:.4f}")
    torch.save(model.state_dict(), 'MultiGIN.pth')
    return test_loss

# =============================================================================
# Main: (Here we assume you have created train_loader and test_loader for your dataset)
# =============================================================================
if __name__ == "__main__":
    # For demonstration, we assume you have a dataset and DataLoader.
    # In practice, replace the following with your actual dataset code.
    data = pyg_data_from_nx(num_nodes=10, p=0.3, feature_dim=32)

    # --- Dummy dataset creation ---
    # Suppose each data sample is a graph with 10 nodes and random features.
    n_samples = 100
    min_nodes = 15
    max_nodes = 25
    dummy_data_list = []
    for _ in range(n_samples):
        n = np.random.randint(min_nodes, max_nodes)
        data = pyg_data_from_nx(num_nodes=n, p=0.3, feature_dim=32)
        if data is not None:
            dummy_data_list.append(data)
    
    # Create DataLoaders.
    train_split = 0.8
    batch_size = 32
    train_loader = DataLoader(dummy_data_list[:int(n_samples*train_split)], batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dummy_data_list[int(n_samples*train_split):], batch_size=batch_size, shuffle=False)
    
    # Run the pipeline.
    #group_models = pipeline(train_loader, test_loader)
    #node_count_loss = train_node_count(train_loader, test_loader, 32)
    #print(node_count_loss)
    train_single_task_pipeline(TASK_LIST, train_loader, test_loader)
    

