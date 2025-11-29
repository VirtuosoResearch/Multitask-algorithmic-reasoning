#!/bin/bash

# Array of algorithms
algorithms=("dfs")
# "bfs" "mst_prim" "floyd_warshall"
# "dfs" "topological_sort" "articulation_points" "bridges" "strongly_connected_components" "mst_kruskal" "dijkstra" "bellman_ford" "dag_shortest_paths"
# CUDA device to use
CUDA_DEVICE=0

# Training parameters
PROCESSOR_TYPE="mpnn"
NUM_LAYERS=3
RUNS=1
TRAIN_STEPS=100
PROJECTION_DIM=16 # no projection

subset_id=0
algorithms=("bfs")
total_algorithms=${#algorithms[@]}

# for ((i=0; i<$total_algorithms; i++)); do
#     echo "=========================================="
#     echo "Training pair: ${algorithms[i]} (subset_id: $subset_id)"
#     echo "=========================================="
    
#     # Run training with both algorithms
#     CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python -m clrs.examples.run \
#         --algorithms "${algorithms[i]}" \
#         --processor_type "$PROCESSOR_TYPE" \
#         --num_layers $NUM_LAYERS \
#         --runs $RUNS \
#         --train_steps $TRAIN_STEPS \
#         --subset_id $subset_id \
#         --save_name "test"
    
#     subset_id=$((subset_id + 1))
#     echo "------------------------------------------"
# done

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python -m clrs.examples.run \
    --algorithms "bfs","dfs","topological_sort","articulation_points","bridges","strongly_connected_components","mst_kruskal","mst_prim","dijkstra","bellman_ford",'dag_shortest_paths',"floyd_warshall"\
    --use_branching_structure --branching_structure_dir "gat_tree" --processor_type branching_gatv2 \
    --num_layers 3 \
    --runs 1 \
    --train_steps 100 \
    --save_name "test"


echo "=========================================="
echo "All training completed!"
echo "=========================================="