#!/bin/bash

# Array of algorithms
algorithms=("topological_sort" "articulation_points" "bridges" "strongly_connected_components" "dag_shortest_paths" "floyd_warshall")
target_algorithms=("bfs" "dfs" "bellman_ford")
#  ("mst_prim" "mst_kruskal" "dijkstra" "bfs" "dfs" "bellman_ford")

# CUDA device to use
CUDA_DEVICE=0

# Training parameters
PROCESSOR_TYPE="edge_t"
NUM_LAYERS=5
RUNS=3
TRAIN_STEPS=10000
PROJECTION_DIM=16

# Create results directory if it doesn't exist
mkdir -p results

# Train on each pair of tasks
subset_id=0
total_algorithms=${#algorithms[@]}
total_target_algorithms=${#target_algorithms[@]}

# subset_id=0
# for algorithm in "${algorithms[@]}"; do
#     echo "Algorithm: $algorithm (subset_id: $subset_id)"
    
#     # Run training
#     CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python -m clrs.examples.run \
#         --algorithms "$algorithm" \
#         --processor_type "$PROCESSOR_TYPE" \
#         --num_layers $NUM_LAYERS \
#         --runs $RUNS \
#         --train_steps $TRAIN_STEPS \
#         --subset_id $subset_id \
#         --save_name "single_task_edge_t_test"
    
#     subset_id=$((subset_id + 1))
#     echo "------------------------------------------"
# done

for ((i=0; i<$total_algorithms; i++)); do
    for ((j=0; j<$total_target_algorithms; j++)); do
        # if [ $i -lt 4 ] && [ $j -lt 4 ]; then
        #     continue
        # fi
        algo1="${algorithms[$i]}"
        algo2="${target_algorithms[$j]}"
        
        echo "=========================================="
        echo "Training pair: $algo1 + $algo2 (subset_id: $subset_id)"
        echo "=========================================="
        
        # Run training with both algorithms
        CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python -m clrs.examples.run \
            --algorithms "$algo1","$algo2" \
            --processor_type "$PROCESSOR_TYPE" \
            --num_layers $NUM_LAYERS \
            --runs $RUNS \
            --train_steps $TRAIN_STEPS \
            --subset_id $subset_id \
            --save_name "pair_tasks_edge_t"
        
        subset_id=$((subset_id + 1))
        echo "------------------------------------------"
    done
done

echo "=========================================="
echo "All pair training completed!"
echo "Total pairs trained: $subset_id"
echo "=========================================="
