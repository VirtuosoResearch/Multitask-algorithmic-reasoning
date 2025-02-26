task_names=("bfs" "dfs" "topological_sort" "articulation_points" "bridges" "strongly_connected_components" "mst_kruskal" "mst_prim" "dijkstra" "bellman_ford" 'dag_shortest_paths' "floyd_warshall")

length=${#task_names[@]}

for ((i = 0; i < $length; i++)); do
CUDA_VISIBLE_DEVICES=0 python -m clrs.examples.fast_estimation_compute_gradients --algorithms "${task_names[$i]}"\
    --use_projection --projection_dim 16 --processor_type "mpnn" --batch_size 1 --load_checkpoint_path "processor_mpnn_layers_3_dim_192_bfs_dfs_top_art_bri_str_mst_mst_dij_bel_dag_flo" --train_steps 50\
    --change_algo_index $i
done

# "bubble_sort" "insertion_sort" "heapsort" "quicksort" "bfs"

# python -m clrs.examples.fast_estimation_compute_gradients --algorithms "bridges"\
#     --use_projection --projection_dim 16 --processor_type "mpnn" --batch_size 1 --load_checkpoint_path "processor_mpnn_layers_3_dim_192_bfs_dfs_top_art_bri_str_mst_mst_dij_bel_dag_flo" --train_steps 20\
#     --change_algo_index 4