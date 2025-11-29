# %%
"""Utility to extract intermediate predecessor label snapshots during Bellman-Ford.

The main function `bellman_ford_predecessor_steps` executes the Bellman-Ford
algorithm on an undirected weighted graph (represented by a NumPy adjacency
matrix) starting from a specified source node. It records a snapshot of the
current predecessor array each time exactly one node's predecessor (and thus its
shortest-path estimate) changes due to a relaxation.

Assumptions / Conventions
-------------------------
- The graph is provided as a square NumPy array `adj` of shape (n, n).
- `adj[i, j]` is the weight of the edge between i and j. The graph is treated as
  undirected, so an edge exists if either `adj[i, j]` or `adj[j, i]` is finite;
  by convention the matrix should be symmetric, but we do not strictly require
  exact equality (we'll take the minimum weight if asymmetry is found).
- Use `numpy.inf` (or a very large value) to denote *no edge*.
- Negative edge weights are allowed. Negative weight cycles reachable from the
  source will raise a `ValueError`.
- The predecessor of each node is initialized to the node itself (self-loop
  convention). The source node's predecessor therefore is itself as well.
- Distances are initialized to +inf except the source which is 0.
- Snapshots: Each snapshot is a Python list of length n containing the current
  predecessor indices AFTER a single successful relaxation that changes exactly
  one node's predecessor. If multiple relaxations occur across iterations,
  each produces its own snapshot.

Returns
-------
List[List[int]]
    A list of predecessor label lists, in chronological order of changes.

Example
-------
>>> import numpy as np
>>> adj = np.array([
...   [0,   4,  np.inf, 5],
...   [4,   0,  1,      np.inf],
...   [np.inf, 1, 0,    2],
...   [5, np.inf, 2,    0]
... ])
>>> steps = bellman_ford_predecessor_steps(adj, source=0)
>>> for s in steps:
...     print(s)
[0, 0, 2, 0]
[0, 1, 2, 0]

(The exact sequence depends on edge ordering.)
"""
from __future__ import annotations

from typing import List
import numpy as np


def bellman_ford_predecessor_steps(adj: np.ndarray, source: int) -> List[List[int]]:
    """Run Bellman-Ford and record predecessor snapshots after each single change.

    Parameters
    ----------
    adj : np.ndarray
        Square (n, n) adjacency matrix. `np.inf` indicates no edge. Should
        represent an undirected graph (symmetric weights), but slight
        asymmetries are tolerated; the smaller weight is used. Diagonal can be
        zeros.
    source : int
        Index of the source node (0 <= source < n).

    Returns
    -------
    List[List[int]]
        Chronological list of predecessor arrays (as Python lists) after each
        single-node relaxation that changes its predecessor.

    Raises
    ------
    ValueError
        If the input is invalid (non-square, bad source) or a negative weight
        cycle is detected reachable from the source.
    """
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("Adjacency matrix must be square (n x n)")

    n = adj.shape[0]
    if not (0 <= source < n):
        raise ValueError("Source index out of range")

    # Copy to float array for safety.
    adj = np.asarray(adj, dtype=float)

    # Build undirected edge list (u, v, w) for ordered pairs (treat as directed both ways).
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            w_ij = adj[i, j]
            w_ji = adj[j, i]
            w = min(w_ij, w_ji)
            if np.isfinite(w):
                edges.append((i, j, w))
                edges.append((j, i, w))

    # Distances and predecessors.
    dist = np.full(n, np.inf, dtype=float)
    dist[source] = 0.0
    pred = np.arange(n)  # predecessor initialized to self

    snapshots: List[List[int]] = []

    # Relax edges up to n-1 times.
    for _ in range(n - 1):
        any_change = False
        for u, v, w in edges:
            # Relaxation check: path to u plus w better than dist[v]
            if np.isfinite(dist[u]) and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                pred[v] = u
                snapshots.append(pred.copy().tolist())
                any_change = True
        if not any_change:
            break  # Early stop

    # Negative cycle detection: if we can still relax, there is a cycle.
    for u, v, w in edges:
        if np.isfinite(dist[u]) and dist[u] + w < dist[v]:
            raise ValueError("Negative weight cycle detected")

    return snapshots


from __future__ import annotations

from collections import deque
from typing import List
import numpy as np


def bfs_predecessor_steps(adj: np.ndarray, source: int) -> List[List[int]]:
    """Run BFS and record predecessor snapshots after each discovery.

    Parameters
    ----------
    adj : np.ndarray
        Square adjacency matrix (n x n). Finite entries indicate an edge.
        Not required to be symmetric; asymmetries are tolerated (edge exists if
        either direction finite). We treat graph as undirected.
    source : int
        Starting node index.

    Returns
    -------
    List[List[int]]
        Each entry is a list of current predecessors after a single new node is
        discovered.
    """
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("Adjacency matrix must be square")

    n = adj.shape[0]
    if not (0 <= source < n):
        raise ValueError("Source index out of range")

    adj = np.asarray(adj)

    # Build undirected neighbor lists for efficiency.
    neighbors = [[] for _ in range(n)]
    for i in range(n):
        # Only iterate j>i, then add both ways when edge exists.
        for j in range(i + 1, n):
            a = adj[i, j]
            b = adj[j, i]
            # Edge exists if either finite and not zero on diagonal; treat 0 off-diagonal as edge only
            # if explicitly present (weight == 0 but still an edge). We'll define existence as finite.
            if (np.isfinite(a) and i != j) or (np.isfinite(b) and i != j):
                neighbors[i].append(j)
                neighbors[j].append(i)

    pred = np.arange(n)  # self predecessors
    visited = np.zeros(n, dtype=bool)
    visited[source] = True

    snapshots: List[List[int]] = []

    q = deque([source])
    while q:
        u = q.popleft()
        for v in neighbors[u]:
            if not visited[v]:
                visited[v] = True
                pred[v] = u
                snapshots.append(pred.copy().tolist())
                q.append(v)

    return snapshots



def dfs_predecessor_steps(adj: np.ndarray, source: int) -> List[List[int]]:
    """Iterative DFS with predecessor snapshots per new discovery.

    Parameters
    ----------
    adj : np.ndarray
        Square adjacency matrix (n x n). Finite off-diagonal entries denote an
        undirected edge (symmetry not strictly required).
    source : int
        Starting node index.

    Returns
    -------
    List[List[int]]
        Predecessor array copies after each new node discovery.
    """
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("Adjacency matrix must be square")
    n = adj.shape[0]
    if not (0 <= source < n):
        raise ValueError("Source index out of range")

    adj = np.asarray(adj)

    # Build neighbor lists (sorted) treating graph as undirected.
    neighbors = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if i == j:
                continue
            a = adj[i, j]
            b = adj[j, i]
            if (np.isfinite(a)) or (np.isfinite(b)):
                neighbors[i].append(j)
                neighbors[j].append(i)
    for lst in neighbors:
        lst.sort()

    pred = np.arange(n)
    visited = np.zeros(n, dtype=bool)
    snapshots: List[List[int]] = []

    stack = [source]
    visited[source] = True  # source discovered; no snapshot for initial state.

    while stack:
        u = stack.pop()
        # Traverse neighbors in ascending order; to preserve order, push in reverse.
        for v in reversed(neighbors[u]):
            if not visited[v]:
                visited[v] = True
                pred[v] = u
                snapshots.append(pred.copy().tolist())
                stack.append(v)
    return snapshots



# %%
import numpy as np

def demo():
    # Example graph (undirected) with 5 nodes
    # 0-1 (4), 0-3 (5), 1-2 (1), 2-3 (2), 2-4 (3), 3-4 (1)
    inf = np.inf
    adj = np.array([
        [0,   4,  inf, 5,   inf],
        [4,   0,  1,   inf, inf],
        [inf, 1,  0,   2,   3],
        [5,   inf, 2,  0,   1],
        [inf, inf, 3,  1,   0],
    ], dtype=float)

    steps = bellman_ford_predecessor_steps(adj, source=0)
    print("Snapshots of predecessors after each single relaxation:")
    for i, s in enumerate(steps):
        print(f"Step {i+1}: {s}")

    # Simple sanity checks
    assert steps, "Expected at least one snapshot"
    n = adj.shape[0]
    for s in steps:
        assert len(s) == n
        for idx, p in enumerate(s):
            assert 0 <= p < n, "Predecessor out of range"  # (self allowed)

def demo_bfs():
    # Reuse a graph similar to the Bellman-Ford demo (weights ignored for BFS):
    inf = np.inf
    adj = np.array([
        [0,   4,  inf, 5,   inf],
        [4,   0,  1,   inf, inf],
        [inf, 1,  0,   2,   3],
        [5,   inf, 2,  0,   1],
        [inf, inf, 3,  1,   0],
    ], dtype=float)

    steps = bfs_predecessor_steps(adj, source=0)
    print("BFS predecessor snapshots (after each discovery):")
    for i, s in enumerate(steps):
        print(f"Step {i+1}: {s}")

    # Sanity checks
    assert steps, "Expected at least one snapshot"
    n = adj.shape[0]
    for s in steps:
        assert len(s) == n
        for p in s:
            assert 0 <= p < n


def demo_dfs():
    inf = np.inf
    # Same demo graph as BFS (weights ignored for DFS structure):
    adj = np.array([
        [0,   4,  inf, 5,   inf],
        [4,   0,  1,   inf, inf],
        [inf, 1,  0,   2,   3],
        [5,   inf, 2,  0,   1],
        [inf, inf, 3,  1,   0],
    ], dtype=float)

    steps = dfs_predecessor_steps(adj, source=0)
    print("DFS predecessor snapshots (after each discovery):")
    for i, s in enumerate(steps):
        print(f"Step {i+1}: {s}")

    # Sanity checks
    assert steps, "Expected at least one snapshot"
    n = adj.shape[0]
    for s in steps:
        assert len(s) == n
        for p in s:
            assert 0 <= p < n


# %%
import numpy as np
from typing import List, Tuple, Dict

def er_adjacency(n: int, p: float, weight_low: float = 1.0, weight_high: float = 2, rng: np.random.Generator | None = None) -> np.ndarray:
    """Generate an undirected weighted Erdos-Renyi graph adjacency matrix.

    Absent edges -> np.inf, diagonal zeros, symmetric weights.
    Weights sampled uniformly in [weight_low, weight_high].
    """
    if rng is None:
        rng = np.random.default_rng()
    adj = np.full((n, n), np.inf, dtype=float)
    np.fill_diagonal(adj, 0.0)
    mask = rng.uniform(size=(n, n)) < p
    # Only keep upper triangle (i<j) to enforce undirected without duplicates.
    for i in range(n):
        for j in range(i + 1, n):
            if mask[i, j]:
                w = rng.uniform(weight_low, weight_high)
                adj[i, j] = w
                adj[j, i] = w
    return adj


def snapshots_to_set(snapshots: List[List[int]]) -> set[Tuple[int, ...]]:
    return {tuple(s) for s in snapshots}


def pair_similarity_ordered(a: List[List[int]], b: List[List[int]]) -> float:
    """Positional similarity: fraction of matching snapshots at identical indices.

    similarity = matches / max(len(a), len(b)); empty-empty => 1.0.
    """
    la, lb = len(a), len(b)
    L = max(la, lb)
    if L == 0:
        return 1.0
    matches = 0
    upto = min(la, lb)
    for i in range(upto):
        if a[i] == b[i]:
            matches += 1
    return matches / L


def evaluate(num_graphs: int = 1000, n: int = 16, p: float = 0.1, seed: int = 3) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    pairs = [("BF", "BFS"), ("BF", "DFS"), ("BFS", "DFS")]
    accum = {f"{a}-{b}": [] for a, b in pairs}

    for _ in range(num_graphs):
        adj = er_adjacency(n, p, rng=rng)
        try:
            bf = bellman_ford_predecessor_steps(adj, 0)
        except ValueError:
            # Negative cycles shouldn't occur with positive weights; skip if any unexpected error.
            continue
        bfs = bfs_predecessor_steps(adj, 0)
        dfs = dfs_predecessor_steps(adj, 0)

        alg_map = {"BF": bf, "BFS": bfs, "DFS": dfs}
        for a, b in pairs:
            accum[f"{a}-{b}"].append(pair_similarity_ordered(alg_map[a], alg_map[b]))

    # Aggregate (mean of similarities). Provide std dev for context.
    summary: Dict[str, float] = {}
    for key, vals in accum.items():
        if vals:
            summary[f"{key}_mean"] = float(np.mean(vals))
            summary[f"{key}_std"] = float(np.std(vals))
        else:
            summary[f"{key}_mean"] = float('nan')
            summary[f"{key}_std"] = float('nan')
    return summary

summary = evaluate()
print("Similarity evaluation over random ER graphs:")
for k in sorted(summary):
    print(f"{k}: {summary[k]:.4f}")
