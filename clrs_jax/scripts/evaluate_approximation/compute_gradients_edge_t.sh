task_names=("bfs" "dfs" "topological_sort" "articulation_points" "bridges" "strongly_connected_components" "mst_kruskal" "mst_prim" "dijkstra" "bellman_ford" 'dag_shortest_paths' "floyd_warshall")

length=${#task_names[@]}

for ((i = 0; i < $length; i++)); do
CUDA_VISIBLE_DEVICES=0 python -m clrs.examples.fast_estimation_compute_gradients --algorithms "${task_names[$i]}"\
    --processor_type 'edge_t' --hidden_size 24 --num_layers 6 --use_projection --projection_dim 4 --batch_size 1 --load_checkpoint_path "processor_edge_t_layers_6_dim_24_bub_ins_hea_qui_bfs_dfs_top_art_bri_str_mst_mst_dij_bel_dag_flo" --train_steps 50\
    --change_algo_index $i
done

for ((i = 0; i < $length; i++)); do
CUDA_VISIBLE_DEVICES=0 python -m clrs.examples.fast_estimation_compute_gradients --algorithms "${task_names[$i]}"\
    --processor_type 'edge_t' --hidden_size 48 --num_layers 6 --use_projection --projection_dim 4 --batch_size 1 --load_checkpoint_path "processor_edge_t_layers_6_dim_48_bub_ins_hea_qui_bfs_dfs_top_art_bri_str_mst_mst_dij_bel_dag_flo" --train_steps 50\
    --change_algo_index $i
done

for ((i = 0; i < $length; i++)); do
CUDA_VISIBLE_DEVICES=0 python -m clrs.examples.fast_estimation_compute_gradients --algorithms "${task_names[$i]}"\
    --processor_type 'edge_t' --hidden_size 96 --num_layers 6 --use_projection --projection_dim 8 --batch_size 1 --load_checkpoint_path "processor_edge_t_layers_6_dim_96_bub_ins_hea_qui_bfs_dfs_top_art_bri_str_mst_mst_dij_bel_dag_flo" --train_steps 50\
    --change_algo_index $i
done


for ((i = 0; i < $length; i++)); do
CUDA_VISIBLE_DEVICES=0 python -m clrs.examples.fast_estimation_compute_gradients --algorithms "${task_names[$i]}"\
    --processor_type 'edge_t' --hidden_size 192 --num_layers 6 --use_projection --projection_dim 16 --batch_size 1 --load_checkpoint_path "processor_edge_t_layers_6_dim_192_bub_ins_hea_qui_bfs_dfs_top_art_bri_str_mst_mst_dij_bel_dag_flo" --train_steps 50\
    --change_algo_index $i
done


# CUDA_VISIBLE_DEVICES=0 python -m clrs.examples.fast_estimation_compute_gradients --algorithms "bfs"\
#     --processor_type 'edge_t' --hidden_size 24 --num_layers 6 --use_projection --projection_dim 4 --batch_size 1 --load_checkpoint_path "processor_edge_t_layers_6_dim_24_bub_ins_hea_qui_bfs_dfs_top_art_bri_str_mst_mst_dij_bel_dag_flo" --train_steps 50\
#     --change_algo_index 0