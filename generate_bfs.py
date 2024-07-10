import numpy as np
import pandas as pd
import os
import argparse
from collections import deque

def generate_tree(n):
    edges = []
    nodes = list(range(n))
    np.random.shuffle(nodes)
    
    for i in range(1, n):
        parent = np.random.choice(nodes[:i])
        child = nodes[i]
        edges.append((parent, child))
    
    return edges

def bfs(edges, n):
    adjacency_list = {i: [] for i in range(n)}
    for parent, child in edges:
        adjacency_list[parent].append(child)
        adjacency_list[child].append(parent)
    
    start_node = 0
    queue = deque([start_node])
    visited = [False] * n
    visited[start_node] = True
    bfs_order = []
    steps = []

    while queue:
        node = queue.popleft()
        bfs_order.append(node)
        steps.append(list(bfs_order))
        # print(steps)
        for neighbor in adjacency_list[node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                queue.append(neighbor)
    
    return bfs_order, steps

#
def main(args):
    data_size = args.data_size
    nodes = args.nodes
    file_dir = "./data/bfs/"
    
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    file_name = os.path.join(file_dir, f"bfs_data_{nodes}.csv")
    print(file_name)
    df = None
    for _ in range(data_size):
        n = nodes
        edges = generate_tree(n)
        bfs_order, steps = bfs(edges, n)

        instance = {
            "n": nodes,
            "input": " ".join([f"{parent}-{child}" for parent, child in edges]),
            "output": " ".join(map(str, bfs_order))
        }

        for k, step in enumerate(steps):
            instance[f"step_{k}"] = " ".join(map(str, step))
            

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
    parser.add_argument("--data_size", type=int, default=1000)
    parser.add_argument("--nodes", type=int, default=10)
    args = parser.parse_args()
    
    main(args)
