#!/bin/bash

# Array of algorithms
algorithms=("bfs" "dfs" "bridges" "strongly_connected_components" "mst_kruskal" "dag_shortest_paths")
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
for ((i=0; i<$total_algorithms; i++)); do
    for ((j=i+1; j<$total_algorithms; j++)); do
        algo1="${algorithms[$i]}"
        algo2="${algorithms[$j]}"
        
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
            --subset_id $subset_id\
            --save_name "pair_tasks_layer_1"\
            --load_checkpoint_path "processor_gatv2_layers_3_dim_192_bfs_dfs_bri_str_mst_dag"\
            --freeze_processor True --freeze_layers 1\
            --remap_algo_from_list "${i},${j}" --remap_algo_to_list "0,1"
        
        subset_id=$((subset_id + 1))
        echo "------------------------------------------"
    done
done


# Array of algorithms
algorithms=("floyd_warshall" "topological_sort" "articulation_points" "dijkstra")
total_algorithms=${#algorithms[@]}
# CUDA device to use
CUDA_DEVICE=0

# Training parameters
PROCESSOR_TYPE="gatv2"
NUM_LAYERS=3
RUNS=3
TRAIN_STEPS=10000
PROJECTION_DIM=16 # no projection

for ((i=0; i<$total_algorithms; i++)); do
    for ((j=i+1; j<$total_algorithms; j++)); do
        algo1="${algorithms[$i]}"
        algo2="${algorithms[$j]}"
        
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
            --save_name "pair_tasks_layer_1"\
            --load_checkpoint_path "processor_gatv2_layers_3_dim_192_flo_top_art_dij"\
            --freeze_processor True --freeze_layers 1\
            --remap_algo_from_list "${i},${j}" --remap_algo_to_list "0,1"
        
        subset_id=$((subset_id + 1))
        echo "------------------------------------------"
    done
done


# Array of algorithms
algorithms=("mst_prim" "bellman_ford")
total_algorithms=${#algorithms[@]}
# CUDA device to use
CUDA_DEVICE=0

# Training parameters
PROCESSOR_TYPE="gatv2"
NUM_LAYERS=3
RUNS=3
TRAIN_STEPS=10000
PROJECTION_DIM=16 # no projection

for ((i=0; i<$total_algorithms; i++)); do
        algo1="${algorithms[$i]}"
        
        echo "=========================================="
        echo "Training pair: $algo1 (subset_id: $subset_id)"
        echo "=========================================="
        
        # Run training with both algorithms
        CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python -m clrs.examples.run \
            --algorithms "$algo1" \
            --processor_type "$PROCESSOR_TYPE" \
            --num_layers $NUM_LAYERS \
            --runs $RUNS \
            --train_steps $TRAIN_STEPS \
            --subset_id $subset_id \
            --save_name "pair_tasks_layer_1"\
            --load_checkpoint_path "processor_gatv2_layers_3_dim_192_mst_bel"\
            --freeze_processor True --freeze_layers 1\
            --remap_algo_from_list "${i}" --remap_algo_to_list "0"
        
        subset_id=$((subset_id + 1))
        echo "------------------------------------------"
done

echo "=========================================="
echo "All training completed!"
echo "=========================================="