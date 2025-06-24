task_names=("bfs" "dfs" "articulation_points" "bellman_ford" "mst_kruskal")

length=${#task_names[@]}

for ((i = 0; i < $length; i++)); do
CUDA_VISIBLE_DEVICES=2 python -m clrs.examples.fast_estimation_compute_gradients \
        --algorithms "${task_names[$i]}" --processor_type "edge_t" --num_layers 5 --hidden_size 192\
        --use_projection --projection_dim 16 --batch_size 1 \
        --load_checkpoint_path "processor_edge_t_layers_5_dim_192_bfs_dfs_art_bel_mst" --train_steps 20\
        --change_algo_index $i
done

# CUDA_VISIBLE_DEVICES=2 python -m clrs.examples.fast_estimation_compute_gradients \
#         --algorithms "bfs" --processor_type "edge_t" --num_layers 5 --hidden_size 192\
#         --use_projection --projection_dim 16 --batch_size 1 \
#         --load_checkpoint_path "processor_edge_t_layers_5_dim_192_bfs_dfs_art_bel_mst" --train_steps 10\
#         --change_algo_index 0