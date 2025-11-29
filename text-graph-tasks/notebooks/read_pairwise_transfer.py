# %%
import pandas as pd
import numpy as np

# task_list = [
#     'edge_existence', 'node_degree', 'node_count', 'edge_count', 'connected_nodes', 'cycle_check',
#     'disconnected_nodes', 'reachability', 'shortest_path', 'maximum_flow', 'triangle_counting', 'node_classification'
# ]
# task_list = [ f"{task}_zero_shot" for task in task_list ]

task_list = [
    "bfs", "dfs", "topological_sort", "articulation_points", "bridges", "strongly_connected_components", "mst_kruskal", "mst_prim", "dijkstra", "bellman_ford", 'dag_shortest_paths', "floyd_warshall"
]

df = pd.read_csv('../results/Qwen-Qwen2.5-1.5B_clrs_pair_lora_r_16/results.csv', index_col=0)

# stl_results = [
#     98.0000, 99.5000, 100.0000, 70.0000, 100.0000, 99.8000, 84.7783, 97.6000, 86.2000, 47.2000,	60.2000, 97.9000
# ]
pairwise_results = np.zeros((12, 12))

for i, task in enumerate(task_list):
    tmp_df = df[df['Task name'] == task]
    if len(tmp_df) == 0:
        continue
    for j, task2 in enumerate(task_list):
        if i == j:
            if len(tmp_df[tmp_df['Trained with'] == "{}".format(task)]) > 0:
                pairwise_results[i, j] = tmp_df[tmp_df['Trained with'] == "{}".format(task)]['test_accuracy'].values[0]
        else:
            if len(tmp_df[tmp_df['Trained with'] == "{} {}".format(task, task2)]) > 0:
                pairwise_results[i, j] = tmp_df[tmp_df['Trained with'] == "{} {}".format(task, task2)]['test_accuracy'].values[0]
            elif len(tmp_df[tmp_df['Trained with'] == "{} {}".format(task2, task)]) > 0:
                pairwise_results[i, j] = tmp_df[tmp_df['Trained with'] == "{} {}".format(task2, task)]['test_accuracy'].values[0]

# %%
for i in range(len(pairwise_results)):
    tmp_results = pairwise_results[i]
    tmp_results[tmp_results==0] = tmp_results[tmp_results!=0].mean() 

# %%
for i, task_name in enumerate(task_list):
    neg_tasks = np.where(pairwise_results[i, :] < -1)[0]
    pos_tasks = np.where(pairwise_results[i, :] > 1)[0]
    neg_tasks = [task_list[idx] for idx in neg_tasks]
    pos_tasks = [task_list[idx] for idx in pos_tasks]
    print(task_name)
    print(pairwise_results[i, :])
    print("Positive: ", pos_tasks)
    print("Negative: ", neg_tasks)
    print("====================")

# %%
def get_average_min(pairwise_results, start_idx_1, end_idx_1, start_idx_2, end_idx_2):
    max_results = np.min(pairwise_results[start_idx_1:end_idx_1, start_idx_2:end_idx_2], axis=1)
    return np.mean(max_results)

transfer_results = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        transfer_results[i, j] = get_average_min(pairwise_results, 4*i, 4*i+4, 4*j, 4*j+4)
print(transfer_results)

# %%
def get_average_max(pairwise_results, start_idx_1, end_idx_1, start_idx_2, end_idx_2):
    max_results = np.max(pairwise_results[start_idx_1:end_idx_1, start_idx_2:end_idx_2], axis=1)
    return np.mean(max_results)

transfer_results = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        transfer_results[i, j] = get_average_max(pairwise_results, 4*i, 4*i+4, 4*j, 4*j+4)
print(transfer_results)

# %%
def get_num_positive(pairwise_results, start_idx_1, end_idx_1, start_idx_2, end_idx_2):
    return np.sum(pairwise_results[start_idx_1:end_idx_1, start_idx_2:end_idx_2] > 0)

num_positive = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        num_positive[i, j] = get_num_positive(pairwise_results, 4*i, 4*i+4, 4*j, 4*j+4)
    
print(num_positive)
# %%

import cvxpy as cp
import numpy as np

def run_sdp_clustering(task_affinities, k):

    def sdp_clustering(T, k):
        n = T.shape[0]

        A = []
        b = []
        # first constraint 
        A.append(np.eye(n))
        b.append(k)

        # second constraint
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
        constraints += [
            cp.trace(A[i] @ X) == b[i] for i in range(len(A))
        ]
        prob = cp.Problem(cp.Minimize(cp.trace(T @ X)),
                        constraints)
        prob.solve()

        # Print result.
        print("The optimal value is", prob.value)
        X_final = X.value
        X_final = X_final > 1/n
        return X_final, X.value

    maximum = np.max(task_affinities)
    X_final, X_value = sdp_clustering(maximum-task_affinities, k)

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

    final_assignments = []
    for cluster_idx in assignment:
        print(" ".join([str(idx) for idx in assignment[cluster_idx]]))
        final_assignments.append(assignment[cluster_idx])
    return final_assignments

n_clusters = 6
final_assignments = run_sdp_clustering(pairwise_results, n_clusters)

for assignment in final_assignments:
    print(" ".join([task_list[idx] for idx in assignment]))


# %%


''' 
edge_existence_zero_shot
[ 0.          1.09999847 -1.29999924 -2.90000153  1.60000229 -0.5
  1.20000076 -1.          1.09999847 -4.29999924 -2.69999695 -1.20000076]
Positive:  ['node_degree_zero_shot', 'connected_nodes_zero_shot', 'disconnected_nodes_zero_shot', 'shortest_path_zero_shot']
Negative:  ['node_count_zero_shot', 'edge_count_zero_shot', 'maximum_flow_zero_shot', 'triangle_counting_zero_shot', 'node_classification_zero_shot']
====================
node_degree_zero_shot
[-0.60000229  0.         -0.20000076 -0.80000305  0.10000229 -0.5
  0.         -0.29999924 -0.70000076 -1.90000153 -1.29999924  0.20000076]
Positive:  []
Negative:  ['maximum_flow_zero_shot', 'triangle_counting_zero_shot']
====================
node_count_zero_shot
[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
Positive:  []
Negative:  []
====================
edge_count_zero_shot
[ 0.20000076  1.          0.09999847  0.         -0.40000153  3.19999695
 -3.         -2.80000305 -3.59999847 -5.80000305 -0.40000153  1.        ]
Positive:  ['cycle_check_zero_shot']
Negative:  ['disconnected_nodes_zero_shot', 'reachability_zero_shot', 'shortest_path_zero_shot', 'maximum_flow_zero_shot']
====================
connected_nodes_zero_shot
[-0.02000046 -0.11428452  0.         -0.19999695  0.          0.
  0.         -0.40000153  0.         -0.07499695 -0.05000305  0.        ]
Positive:  []
Negative:  []
====================
cycle_check_zero_shot
[-0.35999756 -0.20000153 -0.16000061 -0.03999786 -0.20000153  0.
 -0.35999756 -0.20000153 -0.11999969 -0.43999939  0.03999634 -0.20000153]
Positive:  []
Negative:  []
====================
disconnected_nodes_zero_shot
[ -1.1635272    7.1364377    0.98770647 -13.25522108   9.91014604
   4.66640215   0.          -7.42807264   6.59354143 -11.70888777
 -12.71100111  -6.75528212]
Positive:  ['node_degree_zero_shot', 'connected_nodes_zero_shot', 'cycle_check_zero_shot', 'shortest_path_zero_shot']
Negative:  ['edge_existence_zero_shot', 'edge_count_zero_shot', 'reachability_zero_shot', 'maximum_flow_zero_shot', 'triangle_counting_zero_shot', 'node_classification_zero_shot']
====================
reachability_zero_shot
[ 6.99999237e-01 -1.00000000e-01  2.00003052e-01 -1.99998474e-01
 -1.99998474e-01  2.00003052e-01 -1.52587890e-06  0.00000000e+00
  4.00000000e-01 -4.00003052e-01 -6.00000000e-01  5.99996948e-01]
Positive:  []
Negative:  []
====================
shortest_path_zero_shot
[  3.59999924   3.09999924   3.19999771 -10.78000183   3.60000305
   0.75333099   4.20000153   1.8          0.           1.60000305
  -1.79999847  -7.2       ]
Positive:  ['edge_existence_zero_shot', 'node_degree_zero_shot', 'node_count_zero_shot', 'connected_nodes_zero_shot', 'disconnected_nodes_zero_shot', 'reachability_zero_shot', 'maximum_flow_zero_shot']
Negative:  ['edge_count_zero_shot', 'triangle_counting_zero_shot', 'node_classification_zero_shot']
====================
maximum_flow_zero_shot
[-1.2         0.49999886  0.3         0.59999924  0.20000153 -0.79999847
 -0.60000153 -0.79999847 -0.2         0.         -0.60000153 -0.99999924]
Positive:  []
Negative:  ['edge_existence_zero_shot']
====================
triangle_counting_zero_shot
[ 1.69999962e+00 -8.99998856e-01 -4.00000763e-01 -6.00001526e-01
  3.99998474e-01 -1.40000076e+00  8.00000000e-01  2.00001526e-01
 -7.99998474e-01  1.20000153e+00  0.00000000e+00  7.62939450e-07]
Positive:  ['edge_existence_zero_shot', 'maximum_flow_zero_shot']
Negative:  ['cycle_check_zero_shot']
====================
node_classification_zero_shot
[-0.10000076 -0.30000153  0.90000305 -6.30000153  1.29999695 -0.09999695
  0.29999695  0.29999695 -0.70000305 -0.30000153 -0.9         0.        ]
Positive:  ['connected_nodes_zero_shot']
Negative:  ['edge_count_zero_shot']
====================
''' 