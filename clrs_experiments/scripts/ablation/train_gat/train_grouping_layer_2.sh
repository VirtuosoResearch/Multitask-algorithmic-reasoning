#!/bin/bash

# Array of algorithms
algorithms=("bfs" "mst_prim" "floyd_warshall" "dfs" "topological_sort" "articulation_points" "bridges" "strongly_connected_components" "mst_kruskal" "dijkstra" "bellman_ford" "dag_shortest_paths")
total_algorithms=${#algorithms[@]}
# CUDA device to use
CUDA_DEVICE=0

# Training parameters
PROCESSOR_TYPE="gatv2"
NUM_LAYERS=3
RUNS=3
TRAIN_STEPS=10000
PROJECTION_DIM=16 # no projection

subset_id=0
groupings=("bfs"
           "dfs,bridges"
           "strongly_connected_components"
           "mst_kruskal,dag_shortest_paths")
mapping_indices=("0"
                 "1,2"
                 "3"
                 "4,5")
remap_indices=("0"
                "0,1"
                "0"
                "0,1")
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
            --save_name "grouping_layer_2_clusters_9"\
            --load_checkpoint_path "processor_gatv2_layers_3_dim_192_bfs_dfs_bri_str_mst_dag"\
            --freeze_processor True --freeze_layers 1\
            --remap_algo_from_list "${mapping_indices[$subset_id]}" --remap_algo_to_list "${remap_indices[$subset_id]}"
        
        subset_id=$((subset_id + 1))
        echo "------------------------------------------"
done

groupings=("floyd_warshall" 
           "topological_sort,articulation_points" 
           "dijkstra")
mapping_indices=("0"
                 "1,2"
                 "3")
remap_indices=("0"
                "0,1"
                "0")
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
            --save_name "grouping_layer_2_clusters_9"\
            --load_checkpoint_path "processor_gatv2_layers_3_dim_192_flo_top_art_dij"\
            --freeze_processor True --freeze_layers 1\
            --remap_algo_from_list "${mapping_indices[$subset_id]}" --remap_algo_to_list "${remap_indices[$subset_id]}"
        
        subset_id=$((subset_id + 1))
        echo "------------------------------------------"
done



echo "=========================================="
echo "All training completed!"
echo "=========================================="