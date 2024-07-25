import numpy as np
import pandas as pd
import os
import argparse

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, u):
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])
        return self.parent[u]

    def union(self, u, v):
        root_u = self.find(u)
        root_v = self.find(v)
        if root_u != root_v:
            if self.rank[root_u] > self.rank[root_v]:
                self.parent[root_v] = root_u
            elif self.rank[root_u] < self.rank[root_v]:
                self.parent[root_u] = root_v
            else:
                self.parent[root_v] = root_u
                self.rank[root_u] += 1

def generate_graph(n, m):
    edges = []
    for _ in range(m):
        u, v = np.random.choice(n, 2, replace=False)
        weight = np.random.randint(1, 101)
        edges.append((u, v, weight))
    return edges

def kruskal(n, edges):
    edges = sorted(edges, key=lambda x: x[2])
    uf = UnionFind(n)
    mst_edges = []
    intermediate_steps = []
    
    for u, v, weight in edges:
        if uf.find(u) != uf.find(v):
            uf.union(u, v)
            mst_edges.append((u, v, weight))
            intermediate_steps.append((u, v, weight))
        if len(mst_edges) == n - 1:
            break
    
    return mst_edges, intermediate_steps

def main(args):
    data_size = args.data_size
    nodes = args.nodes
    edges_count = args.edges
    file_dir = "./data/kruskal/"
    
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    file_name = os.path.join(file_dir, f"kruskal_data_{nodes}_{edges_count}.csv")
    print(file_name)
    df = None
    for _ in range(data_size):
        n = nodes
        m = edges_count
        edges = generate_graph(n, m)
        mst_edges, intermediate_steps = kruskal(n, edges)
        
        instance = {
            "nodes": n,
            "input": " ".join([f"{u}-{v}-{weight}" for u, v, weight in edges]),
            "output": " ".join([f"{u}-{v}-{weight}" for u, v, weight in mst_edges]),
            # "intermediate_steps": " | ".join([f"{u}-{v}-{weight}" for u, v, weight in intermediate_steps])
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
