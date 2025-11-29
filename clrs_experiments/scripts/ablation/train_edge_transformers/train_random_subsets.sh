#!/bin/bash

# Array of algorithms
algorithms=("bfs" "dfs" "topological_sort" "articulation_points" "bridges" "strongly_connected_components" "mst_kruskal" "mst_prim" "dijkstra" "bellman_ford" "dag_shortest_paths" "floyd_warshall")

# Number of algorithms to sample
subset_size=3

# Number of random subsets to generate
num_subsets=200

# CUDA device to use
CUDA_DEVICE=1

# Training parameters
PROCESSOR_TYPE="edge_t"
NUM_LAYERS=5
RUNS=3
TRAIN_STEPS=5000
PROJECTION_DIM=16

# Create results directory if it doesn't exist
mkdir -p results

# Array to track seen subsets (using sorted algorithm names as key)
declare -A seen_subsets

# Function to shuffle and select N algorithms
sample_algorithms() {
    local n=$1
    local shuffled=($(printf '%s\n' "${algorithms[@]}" | shuf))
    echo "${shuffled[@]:0:$n}"
}

# Function to create a sorted key from algorithms
create_subset_key() {
    local algos=("$@")
    printf '%s\n' "${algos[@]}" | sort | tr '\n' ',' | sed 's/,$//'
}

# Generate and train on random subsets
subset_id=0
attempts=0
max_attempts=$((num_subsets * 10))  # Prevent infinite loop

while [ $subset_id -lt $num_subsets ] && [ $attempts -lt $max_attempts ]; do
    attempts=$((attempts + 1))
    
    # Sample random algorithms
    sampled_algos=($(sample_algorithms $subset_size))
    
    # Create a unique key for this subset (sorted to detect duplicates)
    subset_key=$(create_subset_key "${sampled_algos[@]}")
    
    # Check if we've seen this subset before
    if [ -n "${seen_subsets[$subset_key]}" ]; then
        echo "Skipping duplicate subset (attempt $attempts): $subset_key"
        continue
    fi
    
    # Mark this subset as seen
    seen_subsets[$subset_key]=1
    
    echo "=========================================="
    echo "Training subset $subset_id (attempt $attempts)"
    echo "=========================================="
    
    # Convert array to comma-separated string
    algo_string=$(IFS=,; echo "${sampled_algos[*]}")
    
    echo "Selected algorithms: $algo_string"
    
    # Run training
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python -m clrs.examples.run \
        --algorithms "$algo_string" \
        --use_projection \
        --projection_dim $PROJECTION_DIM \
        --processor_type "$PROCESSOR_TYPE" \
        --num_layers $NUM_LAYERS \
        --runs $RUNS \
        --train_steps $TRAIN_STEPS \
        --subset_id $subset_id \
        --save_name "random_subsets_experiment"
    
    echo "Completed subset $subset_id"
    echo ""
    
    # Increment subset_id only after successful unique subset
    subset_id=$((subset_id + 1))
done

# Check if we hit max attempts
if [ $attempts -ge $max_attempts ]; then
    echo "=========================================="
    echo "Warning: Reached maximum attempts ($max_attempts)"
    echo "Only generated $subset_id unique subsets out of $num_subsets requested"
    echo "=========================================="
fi

echo "=========================================="
echo "All training completed!"
echo "=========================================="
