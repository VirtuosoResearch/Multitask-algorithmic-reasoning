#!/bin/bash

# Array of algorithms
CUDA_DEVICE=2

# Training parameters
PROCESSOR_TYPE="gatv2"
NUM_LAYERS=3
RUNS=3
TRAIN_STEPS=10000
PROJECTION_DIM=16 # no projection


algorithms=("bfs" "mst_prim" "floyd_warshall" "dfs" "topological_sort" "articulation_points" "bridges" "strongly_connected_components" "mst_kruskal" "dijkstra" "bellman_ford" "dag_shortest_paths")
total_algorithms=${#algorithms[@]}

# Run training with both algorithms
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python -m clrs.examples.run \
    --algorithms "bfs","mst_prim","floyd_warshall","dfs","topological_sort","articulation_points","bridges","strongly_connected_components","mst_kruskal","dijkstra","bellman_ford","dag_shortest_paths" \
    --processor_type "$PROCESSOR_TYPE" \
    --num_layers $NUM_LAYERS \
    --runs $RUNS \
    --train_steps $TRAIN_STEPS \
    --subset_id 0 \
    --save_name "mtn"


echo "=========================================="
echo "All training completed!"
echo "=========================================="