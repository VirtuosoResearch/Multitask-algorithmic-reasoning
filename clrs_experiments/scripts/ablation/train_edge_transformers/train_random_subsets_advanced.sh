#!/bin/bash

# ============================================
# Configuration
# ============================================

# Array of algorithms
algorithms=("bfs" "dfs" "topological_sort" "articulation_points" "bridges" 
            "strongly_connected_components" "mst_kruskal" "mst_prim" 
            "dijkstra" "bellman_ford" "dag_shortest_paths" "floyd_warshall")

# Subset configuration
SUBSET_SIZE=3          # Number of algorithms per subset
NUM_SUBSETS=10         # Number of random subsets to generate
SEED=42                # Random seed for reproducibility

# Hardware configuration
CUDA_DEVICE=1

# Model parameters
PROCESSOR_TYPE="edge_t"
NUM_LAYERS=5
HIDDEN_SIZE=192
PROJECTION_DIM=16
NB_HEADS=12

# Training parameters
RUNS=1
TRAIN_STEPS=2000
BATCH_SIZE=4
LEARNING_RATE=2.5e-4
EVAL_EVERY=50
TEST_EVERY=500

# Experiment name
EXPERIMENT_NAME="random_subsets_experiment"

# ============================================
# Script
# ============================================

# Set random seed
RANDOM=$SEED

# Create results directory
mkdir -p results
mkdir -p saved

# Log file
LOG_FILE="results/training_log_${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S).txt"

echo "Starting random subset training experiment" | tee -a "$LOG_FILE"
echo "Experiment name: $EXPERIMENT_NAME" | tee -a "$LOG_FILE"
echo "Total algorithms: ${#algorithms[@]}" | tee -a "$LOG_FILE"
echo "Subset size: $SUBSET_SIZE" | tee -a "$LOG_FILE"
echo "Number of subsets: $NUM_SUBSETS" | tee -a "$LOG_FILE"
echo "Seed: $SEED" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Function to shuffle and select N algorithms
sample_algorithms() {
    local n=$1
    local shuffled=($(printf '%s\n' "${algorithms[@]}" | shuf -n "$n" --random-source=<(yes $RANDOM)))
    echo "${shuffled[@]}"
}

# Generate and train on random subsets
for subset_id in $(seq 0 $((NUM_SUBSETS - 1))); do
    echo "==========================================" | tee -a "$LOG_FILE"
    echo "Training subset $subset_id / $((NUM_SUBSETS - 1))" | tee -a "$LOG_FILE"
    echo "==========================================" | tee -a "$LOG_FILE"
    
    # Sample random algorithms
    sampled_algos=($(sample_algorithms $SUBSET_SIZE))
    
    # Convert array to comma-separated string
    algo_string=$(IFS=,; echo "${sampled_algos[*]}")
    
    echo "Selected algorithms: $algo_string" | tee -a "$LOG_FILE"
    echo "Training started at: $(date)" | tee -a "$LOG_FILE"
    
    # Run training
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python -m clrs.examples.run \
        --algorithms "$algo_string" \
        --use_projection \
        --projection_dim $PROJECTION_DIM \
        --processor_type "$PROCESSOR_TYPE" \
        --num_layers $NUM_LAYERS \
        --hidden_size $HIDDEN_SIZE \
        --nb_heads $NB_HEADS \
        --runs $RUNS \
        --train_steps $TRAIN_STEPS \
        --batch_size $BATCH_SIZE \
        --learning_rate $LEARNING_RATE \
        --eval_every $EVAL_EVERY \
        --test_every $TEST_EVERY \
        --subset_id $subset_id \
        --save_name "$EXPERIMENT_NAME" \
        --seed $((SEED + subset_id)) \
        2>&1 | tee -a "$LOG_FILE"
    
    exit_code=${PIPESTATUS[0]}
    
    if [ $exit_code -eq 0 ]; then
        echo "✓ Completed subset $subset_id successfully" | tee -a "$LOG_FILE"
    else
        echo "✗ Subset $subset_id failed with exit code $exit_code" | tee -a "$LOG_FILE"
    fi
    
    echo "Training finished at: $(date)" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
done

echo "==========================================" | tee -a "$LOG_FILE"
echo "All training completed!" | tee -a "$LOG_FILE"
echo "Results saved in: results/${EXPERIMENT_NAME}.csv" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
