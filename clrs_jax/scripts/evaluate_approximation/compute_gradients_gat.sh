task_names=("bfs" "dfs" "bellman_ford" "dijkstra" "mst_kruskal" "mst_prim")

length=${#task_names[@]}

# Checkpoint path indicates the path of the loaded initialization
# train_steps indicate the number of batches of gradients to compute and save
# change_algo_index indicates the index of the algorithm, in order to load decoder of the task 
for ((i = 0; i < $length; i++)); do
CUDA_VISIBLE_DEVICES=0 python -m clrs.examples.fast_estimation_compute_gradients \
        --algorithms "${task_names[$i]}" --change_algo_index $i --processor_type "gat" --num_layers 5 --hidden_size 192\
        --use_projection --projection_dim 16 --batch_size 1 \
        --load_checkpoint_path "processor_gat_layers_5_dim_192_bfs_dfs_bel_dij_mst_mst" --train_steps 50\
        --gradient_projection_dim 400 
done

for ((i = 0; i < $length; i++)); do
CUDA_VISIBLE_DEVICES=0 python -m clrs.examples.fast_estimation_compute_gradients \
        --algorithms "${task_names[$i]}" --change_algo_index $i --processor_type "triplet_mpnn" --num_layers 5 --hidden_size 192\
        --use_projection --projection_dim 16 --batch_size 1 \
        --load_checkpoint_path "processor_triplet_mpnn_layers_5_dim_192_bfs_dfs_bel_dij_mst_mst" --train_steps 50\
        --gradient_projection_dim 400
done