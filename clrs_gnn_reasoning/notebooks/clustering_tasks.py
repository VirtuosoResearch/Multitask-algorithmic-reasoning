# %%
import cvxpy as cp
import numpy as np

# task_names = ['bfs', "dfs", "topological_sort", "articulation_points", "bridges", "strongly_connected_components", "mst_kruskal", "mst_prim", "dijkstra", "bellman_ford", 'dag_shortest_paths', "floyd_warshall"]
task_names = [ 'bfs', "dfs", "topological_sort", "strongly_connected_components", 'dag_shortest_paths']
num_task = 5
avg_task_models = \
np.array([[1.0000, 0.3652, 0.6875, 0.9297, 0.9102],
[0.9961, 0.3711, 0.6484, 0.9238, 0.9082],
[0.9824, 0.3359, 0.6270, 0.9199, 0.9141],
[0.9961, 0.3594, 0.6641, 0.9199, 0.9043],
[0.9902, 0.3516, 0.6211, 0.9238, 0.7852],])



# [ "mst_kruskal", "dijkstra", "bellman_ford",]
# np.array([[0.6624, 0.7148, 0.5859],
# [0.6624, 0.7129, 0.5625],
# [0.6624, 0.7051, 0.5742],])

# ["articulation_points", "bridges",  "mst_prim", "floyd_warshall"]
# np.array([[0.9199,	0.7088,	0.5371,	0.5078],
# [0.9238,	0.6785,	0.541,	0.5117],
# [0.9883,	0.7868,	0.5449,	0.5488],
# [0.9121,	0.7048,	0.541,	0.4844],])


# Share layer 1 
# np.array(
# [[1.0000,	0.3828,	0.6816,	0.9531,	0.7360,	0.9316,	0.6624,	0.5625,	0.7012,	0.5898,	0.9238,	0.5215,],
# [0.9941,	0.3711,	0.7012,	0.9121,	0.8150,	0.9219,	0.6624,	0.5371,	0.6934,	0.5840,	0.8945,	0.5996,],
# [0.9980,	0.3379,	0.6270,	0.9434,	0.6830,	0.9258,	0.6624,	0.5273,	0.6914,	0.5879,	0.9102,	0.6113,],
# [0.9863,	0.3652,	0.7422,	0.9199,	0.7378,	0.9102,	0.6624,	0.5352,	0.6777,	0.5879,	0.8281,	0.5625,],
# [0.9902,	0.3359,	0.6855,	0.9941,	0.6785,	0.9180,	0.6624,	0.5410,	0.6816,	0.5938,	0.8691,	0.5137,],
# [0.9961,	0.3809,	0.7148,	0.9473,	0.7355,	0.9277,	0.6624,	0.5352,	0.7051,	0.5918,	0.9043,	0.5703,],
# [0.9980,	0.3848,	0.6562,	0.9316,	0.8443,	0.9297,	0.6624,	0.5430,	0.7188,	0.5957,	0.8984,	0.5605,],
# [0.9902,	0.3750,	0.7188,	0.9570,	0.8999,	0.9336,	0.6624,	0.5449,	0.6973,	0.5977,	0.9043,	0.5996,],
# [0.9922,	0.3652,	0.6582,	0.9395,	0.7693,	0.9316,	0.6624,	0.5508,	0.7129,	0.5820,	0.8047,	0.6230,],
# [0.9902,	0.3516,	0.6270,	0.9121,	0.6617,	0.9199,	0.6624,	0.5391,	0.6934,	0.5742,	0.8086,	0.5840,],
# [0.9980,	0.3691,	0.6133,	0.9121,	0.6621,	0.9238,	0.6624,	0.5332,	0.6973,	0.543,	0.7852,	0.6309,],
# [0.9941,	0.3906,	0.6758,	0.9375,	0.9617,	0.9238,	0.6624,	0.541,	0.6934,	0.5977,	0.7617,	0.4844,],])

# Full model pariwise
# np.array(
# [[1.0000, 0.4004, 0.6875, 0.9941, 0.8437, 0.9258, 0.6877, 0.5293, 0.7012, 0.5801, 0.9121, 0.7109],
# [0.9961, 0.5124, 0.7109, 0.9941, 0.6548, 0.9296, 0.6624, 0.5527, 0.7050, 0.5820, 0.9062, 0.7402],
# [0.9980, 0.2930, 0.6602, 0.9883, 0.3408, 0.9043, 0.6624, 0.5254, 0.7012, 0.5293, 0.9043, 0.6992],
# [0.9863, 0.2930, 0.6992, 0.9987, 0.6710, 0.9238, 0.6817, 0.5156, 0.7070, 0.5898, 0.8984, 0.7148],
# [0.9922, 0.4219, 0.6992, 0.9121, 0.8260, 0.9219, 0.6624, 0.5254, 0.6992, 0.5898, 0.9121, 0.4238],
# [0.9961, 0.3671, 0.7266, 0.9785, 0.8238, 0.9307, 0.6624, 0.5293, 0.7031, 0.5781, 0.8916, 0.6943],
# [0.9941, 0.3320, 0.7148, 0.9980, 0.6475, 0.9355, 0.6989, 0.5273, 0.6934, 0.5820, 0.9102, 0.7246],
# [0.9922, 0.3164, 0.7363, 0.9941, 0.7593, 0.9238, 0.7363, 0.5381, 0.6895, 0.5762, 0.9102, 0.7363],
# [0.9961, 0.3750, 0.6855, 0.9980, 0.7915, 0.9414, 0.6871, 0.5332, 0.7090, 0.5879, 0.8867, 0.7129],
# [0.9941, 0.3066, 0.7266, 0.9766, 0.7972, 0.9043, 0.6624, 0.5215, 0.7090, 0.6387, 0.9004, 0.6758],
# [0.9941, 0.3027, 0.6777, 0.9902, 0.4991, 0.9307, 0.6672, 0.5098, 0.6953, 0.5762, 0.8916, 0.6387],
# [0.9805, 0.2675, 0.6973, 0.9902, 0.8971, 0.9307, 0.6624, 0.4883, 0.6836, 0.5449, 0.9082, 0.6943],]
# )




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


T = avg_task_models -  np.reshape(avg_task_models[np.arange(num_task), np.arange(num_task)], (-1, 1))
# T = T / avg_task_models[np.arange(num_task), np.arange(num_task)].reshape(-1, 1)
n_clusters = 3
final_assignments =  run_sdp_clustering(T, n_clusters)

for assignment in final_assignments:
    print(" ".join([task_names[idx] for idx in assignment]))

def compute_groups_estimated_performance(assignments):
    single_performance = np.diag(avg_task_models)
    T = np.copy(avg_task_models); num_task = T.shape[0]
    T[np.arange(num_task), np.arange(num_task)] = 0

    group_performance = []
    for assignment in assignments:
        if len(assignment) == 1:
            group_performance.append(single_performance[assignment[0]].reshape(1))
            continue
        group_performance.append(np.sum(T[assignment, :][:, assignment], axis=1)/(len(assignment)-1))
    group_performance = np.concatenate(group_performance)
    return group_performance

group_performance = compute_groups_estimated_performance(final_assignments)
print(group_performance)
print(np.mean(group_performance))
# %%

# First layer pairwise

# k=2 0.7300229166666666 (Choose this)
# bfs dfs topological_sort strongly_connected_components dag_shortest_paths
# articulation_points bridges mst_prim floyd_warshall
# mst_kruskal dijkstra bellman_ford

# k=3 0.7301 
# bfs dfs topological_sort strongly_connected_components dag_shortest_paths
# mst_kruskal
# articulation_points
# bridges mst_prim floyd_warshall
# dijkstra bellman_ford

# k=4 0.7345437499999999
# bfs dfs topological_sort strongly_connected_components dag_shortest_paths
# articulation_points mst_prim
# bridges floyd_warshall
# mst_kruskal bellman_ford
# dijkstra

# k=5 0.7375791666666666
# bfs dfs dag_shortest_paths
# mst_kruskal
# topological_sort articulation_points strongly_connected_components
# bridges mst_prim floyd_warshall
# bellman_ford
# dijkstra


# Full model pairwise

# 2 0.7387066666666667
# bfs dfs bridges strongly_connected_components dijkstra bellman_ford
# topological_sort articulation_points mst_kruskal mst_prim dag_shortest_paths floyd_warshall

# 3 0.7527504166666668
# bfs bridges strongly_connected_components dijkstra bellman_ford
# dfs
# topological_sort articulation_points mst_kruskal mst_prim dag_shortest_paths floyd_warshall

# 4 0.7583041666666667
# bfs bridges strongly_connected_components
# dfs
# topological_sort articulation_points mst_kruskal mst_prim dijkstra dag_shortest_paths floyd_warshall
# bellman_ford

# 5 0.7513011111111112
# 6 0.7472791666666666