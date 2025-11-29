# %%
import os
import pandas as pd
import numpy as np

# def estimate_task_affinities(cur_set, args):
"""Compute affinity scores between algorithms based on their joint performance in subsets.

Returns:
    np.ndarray: An n x n matrix where entry (i,j) represents the affinity between algorithms i and j
"""
results_dir = './results'
# Construct specific filename
csv_path = os.path.join(results_dir, "processor_edge_t_layers_5_dim_192_full_training_random_subsets_experiment.csv")
cur_set = ["bfs", "dfs", "articulation_points", "bellman_ford", "mst_kruskal"]
if not os.path.exists(csv_path):
    raise ValueError(f"CSV results file not found at: {csv_path}")

# Load the results
df = pd.read_csv(csv_path)

# We'll focus on validation accuracy as the metric
df = df[df['split'] == 'val']
df = df[df['metric'] == 'score']

# Initialize affinity matrix
n = len(cur_set)
cur_set = list(cur_set)  # Convert to list for consistent indexing
affinity_matrix = np.zeros((n, n))

# For each pair of algorithms
for i in range(n):
    for j in range(n):
        algo1, algo2 = cur_set[i], cur_set[j]
        
        # Find subsets containing both algorithms
        mask = df['training_algorithms'].apply(lambda x: algo1 in x and algo2 in x)
        relevant_results = df[mask]
        
        if len(relevant_results) > 0:
            # Average the performance when both algorithms are present
            joint_performance = relevant_results[
                (relevant_results['target_algorithm'] == algo1) 
            ]['value'].mean()
            affinity_matrix[i, j] = joint_performance


            joint_performance = relevant_results[
                (relevant_results['target_algorithm'] == algo2) 
            ]['value'].mean()
            affinity_matrix[j, i] = joint_performance  

# %%
import cvxpy as cp
import numpy as np

def run_regularized_sdp_clustering(task_affinities, size_lam):
    def sdp_clustering(T):
        n = T.shape[0]

        A = []
        b = []
        # remove the trace constraint
        # first constraint: trace(X) = k
        # A.append(np.eye(n))
        # b.append(k)

        # second constraint: Xe = e
        for i in range(n):
            tmp_A = np.zeros((n, n))
            tmp_A[:, i] = 1
            A.append(tmp_A)
            b.append(1)

        # Define and solve the CVXPY problem.
        # Create a symmetric matrix variable.
        X = cp.Variable((n,n), symmetric=True)
        # The operator >> denotes matrix inequality.
        constraints = [X >> 0, X>=0]
        # remove the trace constraint
        constraints += [
            cp.trace(A[i] @ X) == b[i] for i in range(len(A))
        ]
        prob = cp.Problem(cp.Minimize(-cp.trace(T @ X) + size_lam*cp.trace(X)),
                        constraints)
        prob.solve()

        # Print result.
        print("The optimal value is", prob.value)
        X_final = X.value
        X_final = X_final > 1/n
        return X_final, X.value

    maximum = np.max(task_affinities)
    X_final, X_value = sdp_clustering(task_affinities)

    # generate cluster labels
    assignment = {}; cluster_idx = 0; assigned_before = np.zeros(X_final.shape[0])
    for i in range(X_final.shape[0]):
        assigned_count = 0
        for j in range(i, X_final.shape[1]):
            if X_final[i, j] and assigned_before[j] == 0:
                if assigned_before[i] == 0: 
                    if cluster_idx in assignment:
                        assignment[cluster_idx].append(i) 
                    else:
                        assignment[cluster_idx] = [i]
                    assigned_count += 1
                    assigned_before[i] = 1
                if assigned_before[j] == 0:
                    if cluster_idx in assignment:
                        assignment[cluster_idx].append(j) 
                    else:
                        assignment[cluster_idx] = [j]
                    assigned_count += 1
                    assigned_before[j] = 1
        if assigned_count > 0:
            cluster_idx += 1

    for cluster_idx in assignment:
        print(" ".join([str(idx) for idx in assignment[cluster_idx]]))

    return X_final, assignment

# affinity_matrix = np.array([
#     [0.799316, 0.819336, 0.922363, 0.707031, 0.456055, 0.558594],
#     [0.740723, 0.187500, 0.304199, 0.275879, 0.364746, 0.413086],
#     [0.878906, 0.604004, 0.498535, 0.160645, 0.938477, 0.641113],
#     [1.000000, 0.999512, 0.990723, 0.984863, 0.999023, 0.999512],
#     [0.252930, 0.340332, 0.342285, 0.355469, 0.479980, 0.373047],
#     [0.918945, 0.895508, 0.914551, 0.869141, 0.815430, 0.881836],
# ])

X_final, assignment = run_regularized_sdp_clustering(affinity_matrix, size_lam=2)

def compute_inner_cluster_affinity(affinity_matrix, assignment):
    cluster_affinities = 0
    for cluster_idx in assignment:
        indices = assignment[cluster_idx]
        if len(indices) < 1:
            cluster_affinities += affinity_matrix[indices, indices].sum()
            continue
        submatrix = affinity_matrix[np.ix_(indices, indices)]
        avg_affinity = np.sum(submatrix)/len(indices)
        cluster_affinities += avg_affinity
    return cluster_affinities

cluster_affinities = compute_inner_cluster_affinity(affinity_matrix, assignment)
print("Cluster affinities:", cluster_affinities)

# %%
# 0 1
# 2 3 4

# # %%
# from scipy.stats import wilcoxon

# # LearningToBranch accuracies (Edge Transformer, Table 10)
# ltb = [100.0, 36.7, 98.7, 91.2, 99.2, 81.4, 87.1, 93.7, 92.4, 99.3, 78.7]

# # AutoBRANE accuracies (Edge Transformer, Table 10)
# auto = [99.8, 42.6, 96.2, 98.1, 97.3, 81.9, 92.8, 98.7, 96.7, 99.0, 82.8]

# # Two-sided Wilcoxon signed-rank test (paired)
# W, p_two = wilcoxon(auto, ltb, zero_method="wilcox", alternative="two-sided")
# # One-sided ("AutoBRANE > LearningToBranch")
# _, p_greater = wilcoxon(auto, ltb, zero_method="wilcox", alternative="greater")

# print(W, p_two, p_greater)

# # %%
# import os
# import numpy as np

# gradients_dir = f"./gradients/processor_edge_t_layers_5_dim_192_seed_0_projection_dim_16_bfs"

# # load gradients & solve logistic regression
# gradients, labels = [], []
# count = 0
# file_list = list(os.listdir(gradients_dir))
# file_list.sort()
# for file in file_list:
#     if "gradients" in file:
#         gradients.append(np.load(os.path.join(gradients_dir, file)))
#         print(file, gradients[-1].shape)
#         count += 1
#     if count >= 20: 
#         break

# count = 0
# file_list = list(os.listdir(gradients_dir))
# file_list.sort()
# for file in file_list:
#     if "labels" in file:
#         labels.append(np.load(os.path.join(gradients_dir, file)))
#         print(file, labels[-1].shape)
#         count += 1
#     if count >= 20:
#         break

# gradients = np.concatenate(gradients, axis=0)
# labels = np.concatenate(labels, axis=0)
