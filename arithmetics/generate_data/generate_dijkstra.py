import numpy as np
import pandas as pd
import os
import argparse
import heapq

def generate_graph(n, m):
    edges = []
    for _ in range(m):
        u, v = np.random.choice(n, 2, replace=False)
        weight = np.random.randint(1, 101)
        edges.append((u, v, weight))
        edges.append((v, u, weight)) 
    return edges

def dijkstra(n, edges, start_node):
    adjacency_list = {i: [] for i in range(n)}
    for u, v, weight in edges:
        adjacency_list[u].append((v, weight))
    
    dist = [float('inf')] * n
    dist[start_node] = 0
    visited = [False] * n
    min_heap = [(0, start_node)]
    intermediate_steps = []

    while min_heap:
        current_dist, u = heapq.heappop(min_heap)
        if visited[u]:
            continue
        visited[u] = True
        intermediate_steps.append((u, current_dist))

        for v, weight in adjacency_list[u]:
            if not visited[v] and current_dist + weight < dist[v]:
                dist[v] = current_dist + weight
                heapq.heappush(min_heap, (dist[v], v))

    return dist, intermediate_steps

def main(args):
    data_size = args.data_size
    nodes = args.nodes
    edges_count = args.edges
    file_dir = "./data/dijkstra/"
    
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    file_name = os.path.join(file_dir, f"dijkstra_data_{nodes}_{edges_count}.csv")
    print(file_name)
    df = None
    for _ in range(data_size):
        n = nodes
        m = edges_count
        edges = generate_graph(n, m)
        start_node = np.random.randint(n)
        dist, intermediate_steps = dijkstra(n, edges, start_node)
        
        edges = " ".join([f"{u}-{v}-{weight}" for u, v, weight in edges])
        input = str(start_node) + " " + edges
        
        instance = {
            "nodes": n,
            # "edges": " ".join([f"{u}-{v}-{weight}" for u, v, weight in edges]),
            # "start_node": start_node,
            "input": input,
            "output": " ".join(map(str, dist)),
            # "intermediate_steps": " | ".join([f"{u}-{dist}" for u, dist in intermediate_steps])
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
    parser.add_argument("--nodes", type=int, default=10)
    parser.add_argument("--edges", type=int, default=15)
    args = parser.parse_args()
    
    main(args)
