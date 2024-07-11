import numpy as np
import pandas as pd
import os
import argparse
import collections
import sys

class Dinic:
    def __init__(self, n):
        self.n = n
        self.adj = [[] for _ in range(n)]
        self.level = [0] * n
        self.ptr = [0] * n

    def add_edge(self, u, v, cap):
        self.adj[u].append([v, cap, len(self.adj[v])])
        self.adj[v].append([u, 0, len(self.adj[u]) - 1])

    def bfs(self, src, sink):
        self.level = [-1] * self.n
        self.level[src] = 0
        queue = collections.deque([src])
        while queue:
            u = queue.popleft()
            for v, cap, _ in self.adj[u]:
                if cap and self.level[v] == -1:
                    self.level[v] = self.level[u] + 1
                    queue.append(v)
        return self.level[sink] != -1

    def dfs(self, u, sink, flow):
        if u == sink:
            return flow
        while self.ptr[u] < len(self.adj[u]):
            v, cap, rev = self.adj[u][self.ptr[u]]
            if cap and self.level[v] == self.level[u] + 1:
                pushed = self.dfs(v, sink, min(flow, cap))
                if pushed:
                    self.adj[u][self.ptr[u]][1] -= pushed
                    self.adj[v][rev][1] += pushed
                    return pushed
            self.ptr[u] += 1
        return 0

    def max_flow(self, src, sink):
        flow = 0
        while self.bfs(src, sink):
            self.ptr = [0] * self.n
            while True:
                pushed = self.dfs(src, sink, float('inf'))
                if not pushed:
                    break
                flow += pushed
        return flow

def generate_graph(n, m):
    edges = []
    for _ in range(m):
        u, v = np.random.choice(n, 2, replace=False)
        capacity = np.random.randint(1, 101)
        edges.append((u, v, capacity))
    return edges

def main(args):
    data_size = args.data_size
    nodes = args.nodes
    edges_count = args.edges
    file_dir = "./data/dinic_wll/"
    
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    file_name = os.path.join(file_dir, f"dinic_data_{nodes}_{edges_count}.csv")
    print(file_name)
    df = None
    for _ in range(data_size):
        n = nodes
        m = edges_count
        edges = generate_graph(n, m)
        src = np.random.randint(n)
        sink = np.random.randint(n)
        while sink == src:
            sink = np.random.randint(n)
        
        dinic = Dinic(n)
        for u, v, capacity in edges:
            dinic.add_edge(u, v, capacity)
        
        max_flow_value = dinic.max_flow(src, sink)
        
        instance = {
            "nodes": n,
            "edges": " ".join([f"{u}-{v}-{capacity}" for u, v, capacity in edges]),
            "source": src,
            "sink": sink,
            "max_flow": max_flow_value
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
    parser.add_argument("--data_size", type=int, default=1000)
    parser.add_argument("--nodes", type=int, default=10)
    parser.add_argument("--edges", type=int, default=15)
    args = parser.parse_args()
    
    main(args)
