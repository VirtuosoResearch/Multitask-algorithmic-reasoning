import argparse
import queue
import os

import numpy as np
import pandas as pd
from clustering import *

def estimate_task_affinities(cur_set, args):
    """Compute affinity scores between algorithms based on their joint performance in subsets.
    
    Returns:
        np.ndarray: An n x n matrix where entry (i,j) represents the affinity between algorithms i and j
    """
    results_dir = './results'
    # Construct specific filename
    csv_path = os.path.join(results_dir,
        f'processor_{args.processor_type}_layers_{args.num_layers}_dim_{args.hidden_size}_'
        f'seed_{args.gradient_projection_seed}_projection_dim_{args.gradient_projection_dim}_'
        f'subset_{args.num_subsets}_subset_size_{args.subset_size}.csv')
    
    if not os.path.exists(csv_path):
        raise ValueError(f"CSV results file not found at: {csv_path}")
    
    # Load the results
    df = pd.read_csv(csv_path)
    
    # We'll focus on validation accuracy as the metric
    df = df[df['split'] == 'test']
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
                # Average the performance when both algorithms are present
                joint_performance = relevant_results[
                    (relevant_results['target_algorithm'] == algo1) 
                ]['value'].mean()
                affinity_matrix[i, j] = joint_performance


                joint_performance = relevant_results[
                    (relevant_results['target_algorithm'] == algo2) 
                ]['value'].mean()
                affinity_matrix[j, i] = joint_performance  
    
    return affinity_matrix

def fast_approx_partition(cur_set, cur_layer, cur_checkpoint, args):
    # randomly split cur_set into 2 subsets
    # n = len(cur_set)
    # if n == 1:
    #     return [cur_set]
    # first_num = np.random.randint(1, n)
    # first_subset = [str(algo) for algo in np.random.choice(list(cur_set), first_num, replace=False)]
    # first_subset = set(first_subset)
    # second_subset = cur_set - first_subset
    # return [first_subset, second_subset]

    # Step 1: train a model with freezing l-1 layers 
    cur_set = list(cur_set)
    os.system("CUDA_VISIBLE_DEVICES=2 python -m clrs.examples.run \
            --algorithms {} --processor_type {} --num_layers {} --hidden_size {} \
            --load_checkpoint_path {} --freeze_processor --freeze_layers {} \
            --use_projection --projection_dim 16".format(
                ",".join(cur_set), 
                args.processor_type, 
                args.num_layers, 
                args.hidden_size,
                cur_checkpoint if cur_checkpoint is not None else "test",
                cur_layer - 1
            ))
    
    # Step 2: Compute the gradients of the model
    tmp_checkpoint = f"processor_{args.processor_type}_layers_{args.num_layers}_dim_{args.hidden_size}_" \
                    + "_".join([algorithm[:3] for algorithm in cur_set])
    for i, algo in enumerate(cur_set):
        os.system("CUDA_VISIBLE_DEVICES=2 python -m clrs.examples.fast_estimation_compute_gradients \
        --algorithms {} --processor_type {} --num_layers {} --hidden_size {}\
        --use_projection --projection_dim 16 --batch_size 1 \
        --load_checkpoint_path {} --train_steps 50\
        --change_algo_index {} --gradient_projection_dim {}".format(
            algo,
            args.processor_type,
            args.num_layers,
            args.hidden_size,
            tmp_checkpoint,
            i,
            args.gradient_projection_dim
        ))
    
    # Step 3: Estimate the performance of the model on subsets, freezing the l layers
    os.system("CUDA_VISIBLE_DEVICES=2 python -m clrs.examples.fast_estimation_linear_regression\
        --algorithms {} --processor_type {} --num_layers {} --hidden_size {}\
        --use_projection --projection_dim 16 --batch_size 1 \
        --load_checkpoint_path {}\
        --layer {} --gradient_projection_dim {} --regularization_lambda 1e3 \
        --num_subsets {} --num_subset_size {}".format(
            ",".join(cur_set),
            args.processor_type,
            args.num_layers,
            args.hidden_size,
            tmp_checkpoint,
            cur_layer,
            args.gradient_projection_dim,
            args.num_subsets,
            args.subset_size
    ))

    # Step 4: Estimate the task affinity and clustering the tasks
    task_affinities = estimate_task_affinities(cur_set, args)

    # Step 5: tune the clustering hyperparameters
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

    optimal_lam = None; max_affinity = -1e9
    for lam in np.arange(0.1, 1.1, 0.1):
        X_final, assignment = run_regularized_sdp_clustering(task_affinities, size_lam=lam)
        cluster_affinities = compute_inner_cluster_affinity(task_affinities, assignment)
        print("Cluster affinities:", cluster_affinities)
        if cluster_affinities > max_affinity:
            max_affinity = cluster_affinities
            optimal_lam = lam
    
    X_final, assignment = run_regularized_sdp_clustering(task_affinities, size_lam=optimal_lam)
    task_paritions = []
    for cluster in assignment:
        task_paritions.append(set([cur_set[idx] for idx in cluster]))
    return task_paritions


def main(args):
    Q = queue.Queue()
    init_set = set(args.algorithms)
    Q.put((init_set, 1)) # indexed from 1

    branching_structure = [args.algorithms]
    cur_checkpoint = None # no checkpoint at the beginning
    while not Q.empty():
        cur_set, cur_layer = Q.get()
        print(cur_layer, cur_set)

        if cur_layer <= args.num_layers:
            task_partitions = fast_approx_partition(cur_set, cur_layer, cur_checkpoint, args)
            cur_checkpoint = "processor_{}_layers_{}_dim_{}_{}".format(
                args.processor_type,
                args.num_layers,
                args.hidden_size,
                "_".join([algorithm[:3] for algorithm in cur_set])
            )
            for next_subset in task_partitions:
                Q.put((next_subset, cur_layer+1))
            
            # adding the task partitions to the branching structure
            if len(branching_structure) <= cur_layer:
                branching_structure.append([])
            branching_structure[cur_layer+1].extend(task_partitions)

    # write the branching structure to a file
    file_name = "tree_structure_{}".format([algo[:3] for algo in args.algorithms])
    with open(f"./tree_configs/{file_name}.txt", "w") as f:
        for i, layer in enumerate(branching_structure):
            for subset in layer:
                f.write("{}: {}\n".format(i, " ".join(list(subset))))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BranchingNN search")
    parser.add_argument("--algorithms", type=str, nargs="+", 
                        default=["bfs","dfs","topological_sort","articulation_points","bridges","strongly_connected_components","mst_kruskal","mst_prim","dijkstra","bellman_ford",'dag_shortest_paths',"floyd_warshall"])
    parser.add_argument("--processor_type", type=str, default="edge_t")
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--hidden_size", type=int, default=192)
    parser.add_argument("--gradient_projection_dim", type=int, default=400)
    parser.add_argument("--num_subsets", type=int, default=50)
    parser.add_argument("--subset_size", type=int, default=4)
    parser.add_argument("--gradient_projection_seed", type=int, default=0)
    args = parser.parse_args()
    print(args)
    main(args)