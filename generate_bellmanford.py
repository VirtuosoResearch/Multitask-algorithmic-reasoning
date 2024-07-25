import numpy as np
import pandas as pd
import os
import argparse

def generate_graph(num_nodes, num_edges, max_weight=10):
    edges = []
    for _ in range(num_edges):
        u = np.random.randint(0, num_nodes)
        v = np.random.randint(0, num_nodes)
        while u == v:
            v = np.random.randint(0, num_nodes)
        weight = np.random.randint(1, max_weight + 1)
        edges.append((u, v, weight))
    return edges

def bellman_ford(num_nodes, edges, src):
    dist = [float('inf')] * num_nodes
    dist[src] = 0
    
    for _ in range(num_nodes - 1):
        for u, v, weight in edges:
            if dist[u] != float('inf') and dist[u] + weight < dist[v]:
                dist[v] = dist[u] + weight
    
    for u, v, weight in edges:
        if dist[u] != float('inf') and dist[u] + weight < dist[v]:
            return None  # Indicates a negative weight cycle
    
    return dist

def main(args):
    data_size = args.data_size
    num_nodes = args.num_nodes
    num_edges = args.num_edges
    file_dir = "./data/bellman/"
    
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    file_name = os.path.join(file_dir, f"bellman_data_{num_nodes}_{num_edges}.csv")
    
    df = None
    for _ in range(data_size):
        edges = generate_graph(num_nodes, num_edges)
        src = np.random.randint(0, num_nodes)
        distances = bellman_ford(num_nodes, edges, src)
        
        if distances is None:
            continue  # Skip graphs with negative weight cycles
        
        edges_str = " ".join([f"{u}-{v}-{w}" for u, v, w in edges])
        input_data = f"nodes: {num_nodes} edges: {edges_str} source: {src}"
        dis = " ".join(map(str, distances))
        output_data = "distance: "+dis
        
        instance = {
            "node" : num_nodes,
            "input": input_data,
            "output": output_data
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
    parser.add_argument("--data_size", type=int, default=100000)
    parser.add_argument("--num_nodes", type=int, default=8)
    parser.add_argument("--num_edges", type=int, default=16)
    args = parser.parse_args()
    
    main(args)
