#!/bin/bash

# Array of algorithms
algorithms=("bfs" "mst_prim" "floyd_warshall")
# "dfs" "topological_sort" "articulation_points" "bridges" "strongly_connected_components" "mst_kruskal" "dijkstra" "bellman_ford" "dag_shortest_paths"
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

# Train on each single task
subset_id=0
for algorithm in "${algorithms[@]}"; do
    echo "Algorithm: $algorithm (subset_id: $subset_id)"
    
    # Run training
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python -m clrs.examples.run \
        --algorithms "$algorithm" \
        --use_projection \
        --projection_dim $PROJECTION_DIM \
        --processor_type "$PROCESSOR_TYPE" \
        --num_layers $NUM_LAYERS \
        --runs $RUNS \
        --train_steps $TRAIN_STEPS \
        --subset_id $subset_id \
        --save_name "single_task_edge_t_test"
    
    subset_id=$((subset_id + 1))
    echo "------------------------------------------"
done

subset_id=0
for algorithm in "${algorithms[@]}"; do
    echo "Algorithm: $algorithm (subset_id: $subset_id)"
    
    # Run training
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python -m clrs.examples.run \
        --algorithms "$algorithm" \
        --use_projection \
        --projection_dim $PROJECTION_DIM \
        --processor_type "$PROCESSOR_TYPE" \
        --num_layers 3 \
        --runs $RUNS \
        --train_steps $TRAIN_STEPS \
        --subset_id $subset_id \
        --save_name "single_task_edge_t_test"
    
    subset_id=$((subset_id + 1))
    echo "------------------------------------------"
done


subset_id=0
for algorithm in "${algorithms[@]}"; do
    echo "Algorithm: $algorithm (subset_id: $subset_id)"
    
    # Run training
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python -m clrs.examples.run \
        --algorithms "$algorithm" \
        --processor_type "$PROCESSOR_TYPE" \
        --num_layers 3 \
        --runs $RUNS \
        --train_steps $TRAIN_STEPS \
        --subset_id $subset_id \
        --save_name "single_task_edge_t_test"
    
    subset_id=$((subset_id + 1))
    echo "------------------------------------------"
done

echo "=========================================="
echo "All training completed!"
echo "=========================================="
