#!/bin/bash

# Array of algorithms
algorithms=("bfs" "mst_prim" "floyd_warshall" "dfs" "topological_sort" "articulation_points" "bridges" "strongly_connected_components" "mst_kruskal" "dijkstra" "bellman_ford" "dag_shortest_paths")
total_algorithms=${#algorithms[@]}
# CUDA device to use
CUDA_DEVICE=2

# Training parameters
PROCESSOR_TYPE="gatv2"
NUM_LAYERS=3
RUNS=3
TRAIN_STEPS=10000
PROJECTION_DIM=16 # no projection

# subset_id=0
# groupings=("bfs,dfs,bridges,strongly_connected_components,mst_kruskal,dag_shortest_paths" 
#            "floyd_warshall,topological_sort,articulation_points,dijkstra"
#            "mst_prim,bellman_ford")
# for grouping in "${groupings[@]}"; do
#     echo "=========================================="
#     echo "Training grouping: $grouping (subset_id: $subset_id)"
#     echo "=========================================="
#         # Run training with both algorithms
#         CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python -m clrs.examples.run \
#             --algorithms "$grouping" \
#             --processor_type "$PROCESSOR_TYPE" \
#             --num_layers $NUM_LAYERS \
#             --runs $RUNS \
#             --train_steps $TRAIN_STEPS \
#             --subset_id $subset_id \
#             --save_name "grouping_layer_1_clusters_3"
        
#         subset_id=$((subset_id + 1))
#         echo "------------------------------------------"
# done

subset_id=0
groupings=("bfs,dfs,bridges,strongly_connected_components,mst_kruskal" 
           "floyd_warshall,dag_shortest_paths"
           "topological_sort,articulation_points"
           "mst_prim,bellman_ford,dijkstra")
for grouping in "${groupings[@]}"; do
    echo "=========================================="
    echo "Training grouping: $grouping (subset_id: $subset_id)"
    echo "=========================================="
        # Run training with both algorithms
        CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python -m clrs.examples.run \
            --algorithms "$grouping" \
            --processor_type "$PROCESSOR_TYPE" \
            --num_layers $NUM_LAYERS \
            --runs $RUNS \
            --train_steps $TRAIN_STEPS \
            --subset_id $subset_id \
            --save_name "grouping_layer_1_clusters_4"
        
        subset_id=$((subset_id + 1))
        echo "------------------------------------------"
done


echo "=========================================="
echo "All training completed!"
echo "=========================================="