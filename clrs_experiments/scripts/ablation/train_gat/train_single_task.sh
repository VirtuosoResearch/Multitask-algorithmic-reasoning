#!/bin/bash

# Array of algorithms
algorithms=("dfs" "topological_sort" "articulation_points" "bridges" "strongly_connected_components" "mst_kruskal" "dijkstra" "bellman_ford" "dag_shortest_paths")
# "bfs" "mst_prim" "floyd_warshall"
# "dfs" "topological_sort" "articulation_points" "bridges" "strongly_connected_components" "mst_kruskal" "dijkstra" "bellman_ford" "dag_shortest_paths"
# CUDA device to use
CUDA_DEVICE=0

# Training parameters
PROCESSOR_TYPE="gatv2"
NUM_LAYERS=3
RUNS=3
TRAIN_STEPS=10000
PROJECTION_DIM=16 # no projection

# # Create results directory if it doesn't exist
# mkdir -p results

subset_id=0
for algorithm in "${algorithms[@]}"; do
    echo "Algorithm: $algorithm (subset_id: $subset_id)"
    
    # Run training
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python -m clrs.examples.run \
        --algorithms "$algorithm" \
        --processor_type "$PROCESSOR_TYPE" \
        --num_layers $NUM_LAYERS \
        --runs $RUNS \
        --train_steps $TRAIN_STEPS \
        --subset_id $subset_id \
        --save_name "single_task"
    
    subset_id=$((subset_id + 1))
    echo "------------------------------------------"
done

# algorithms=("bfs" "mst_prim" "floyd_warshall" "dfs" "topological_sort" "articulation_points" "bridges" "strongly_connected_components" "mst_kruskal" "dijkstra" "bellman_ford" "dag_shortest_paths")
# total_algorithms=${#algorithms[@]}

# for ((i=0; i<$total_algorithms; i++)); do
#     for ((j=i+1; j<$total_algorithms; j++)); do
#         algo1="${algorithms[$i]}"
#         algo2="${algorithms[$j]}"
        
#         echo "=========================================="
#         echo "Training pair: $algo1 + $algo2 (subset_id: $subset_id)"
#         echo "=========================================="
        
#         # Run training with both algorithms
#         CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python -m clrs.examples.run \
#             --algorithms "$algo1" "$algo2" \
#             --processor_type "$PROCESSOR_TYPE" \
#             --num_layers $NUM_LAYERS \
#             --runs $RUNS \
#             --train_steps $TRAIN_STEPS \
#             --subset_id $subset_id \
#             --save_name "pair_tasks"
        
#         subset_id=$((subset_id + 1))
#         echo "------------------------------------------"
#     done
# done

echo "=========================================="
echo "All training completed!"
echo "=========================================="