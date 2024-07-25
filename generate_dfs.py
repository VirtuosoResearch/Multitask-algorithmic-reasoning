import numpy as np
import pandas as pd
import os
import argparse

def generate_tree(n):
    edges = []
    nodes = list(range(n))
    np.random.shuffle(nodes)
    
    for i in range(1, n):
        parent = np.random.choice(nodes[:i])
        child = nodes[i]
        edges.append((parent, child))
    
    return edges

def dfs(adjacency_list, start_node):
    visited = [False] * len(adjacency_list)
    dfs_order = []
    
    def dfs_recursive(node):
        if visited[node]:
            return
        visited[node] = True
        dfs_order.append(node)
        for neighbor in adjacency_list[node]:
            if not visited[neighbor]:
                dfs_recursive(neighbor)
    
    dfs_recursive(start_node)
    return dfs_order

def main(args):
    data_size = args.data_size
    nodes = args.nodes
    file_dir = "./data/dfs/"
    
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    file_name = os.path.join(file_dir, f"dfs_first_data_{nodes}.csv")
    
    df = None
    for _ in range(data_size):
        n = nodes
        edges = generate_tree(n)
        
        adjacency_list = {i: [] for i in range(n)}
        for parent, child in edges:
            adjacency_list[parent].append(child)
            adjacency_list[child].append(parent)
        
        dfs_order = dfs(adjacency_list, 0)
        
        instance = {
            "nodes": n,
            "input": " ".join([f"{parent}-{child}" for parent, child in edges]),
            "output": " ".join(map(str, dfs_order))
        }
        
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
    parser.add_argument("--data_size", type=int, default=10000)
    parser.add_argument("--nodes", type=int, default=20)
    args = parser.parse_args()
    
    main(args)
