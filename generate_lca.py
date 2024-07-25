import numpy as np
import pandas as pd
import os
import argparse
from collections import defaultdict

def generate_tree(n):
    edges = []
    nodes = list(range(n))
    np.random.shuffle(nodes)
    
    for i in range(1, n):
        parent = np.random.choice(nodes[:i])
        child = nodes[i]
        edges.append((parent, child))
    
    return edges

def preprocess_lca(n, edges):
    LOG = 20  # For n up to around 10^6
    parent = [[-1] * LOG for _ in range(n)]
    depth = [0] * n
    adjacency_list = defaultdict(list)
    
    for u, v in edges:
        adjacency_list[u].append(v)
        adjacency_list[v].append(u)
    
    def dfs(v, p, d):
        parent[v][0] = p
        depth[v] = d
        for i in range(1, LOG):
            if parent[v][i - 1] != -1:
                parent[v][i] = parent[parent[v][i - 1]][i - 1]
            else:
                break
        for u in adjacency_list[v]:
            if u != p:
                dfs(u, v, d + 1)
    
    dfs(0, -1, 0)
    
    return parent, depth

def lca(u, v, parent, depth):
    LOG = 20
    if depth[u] < depth[v]:
        u, v = v, u
    
    diff = depth[u] - depth[v]
    for i in range(LOG):
        if (diff >> i) & 1:
            u = parent[u][i]
    
    if u == v:
        return u
    
    for i in range(LOG - 1, -1, -1):
        if parent[u][i] != parent[v][i]:
            u = parent[u][i]
            v = parent[v][i]
    
    return parent[u][0]

def main(args):
    data_size = args.data_size
    nodes = args.nodes
    pairs_per_tree = args.pairs_per_tree
    file_dir = "./data/lca/"
    
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    file_name = os.path.join(file_dir, f"lca_data_{nodes}.csv")
    
    df = None
    for _ in range(data_size):
        n = nodes
        edges = generate_tree(n)
        parent, depth = preprocess_lca(n, edges)
        
        instances = []
        # for _ in range(pairs_per_tree):
        u, v = np.random.choice(n, 2, replace=False)
        edges = " ".join([f"{p}-{c}" for p, c in edges])
        input = "node_pair: "+ f"{u}-{v}" + " edges: " + edges
        
        
        lca_node = lca(u, v, parent, depth)
        instances.append({
            "nodes": n,
            "input" : input,
            "output": lca_node
        })
        
        for instance in instances:
            for key, val in instance.items():
                instance[key] = [val, ]
            tmp_df = pd.DataFrame(instance)
            df = pd.concat([df, tmp_df], ignore_index=True) if df is not None else tmp_df

        if len(df) >= 1000:
            if not os.path.exists(file_name):
                df.to_csv(file_name, index=False)
            else:
                result_df = pd.read_csv(file_name)
                result_df = pd.concat([result_df, df], ignore_index=True)
                result_df.to_csv(file_name, index=False)
                if result_df.shape[0] >= data_size:
                    exit()
            df = None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_size", type=int, default=100000)
    parser.add_argument("--nodes", type=int, default=10)
    parser.add_argument("--pairs_per_tree", type=int, default=1)
    args = parser.parse_args()
    
    main(args)
