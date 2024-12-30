
task_names=('bfs' "dfs" "topological_sort" "articulation_points" "bridges" "strongly_connected_components" "mst_kruskal" "mst_prim" "dijkstra" "bellman_ford" 'dag_shortest_paths' "floyd_warshall")
length=${#task_names[@]}

# for ((j = 0; j < $length; j++)); do
#   python train_mtl.py --algorithms "floyd_warshall" "${task_names[$j]}" \
#   --cfg "./configs/SAGE.yml" --lr 5e-5 --hidden_dim 128 --gnn_layers 3 --enable_gru --devices 1 --batch_size 8 --epochs 50 --runs 1 --loss_weight_hint 2\
#   --save_name "pairwise_layer_1" \
#   --load_checkpoint_dir "bfs_dfs_topological_sort_articulation_points_bridges_strongly_connected_components_mst_kruskal_mst_p/SAGE-hints-run0-mtl/seed892-epoch=19-step=30000.pt" \
#   --load_layer 1
# done


# for ((j = 0; j < $length; j++)); do
#   python train_mtl.py --algorithms "floyd_warshall" "${task_names[$j]}" \
#   --cfg "./configs/SAGE.yml" --lr 5e-5 --hidden_dim 128 --gnn_layers 3 --enable_gru --devices 1 --batch_size 8 --epochs 50 --runs 1 --loss_weight_hint 2\
#   --save_name "pairwise_layer_2" \
#   --load_checkpoint_dir "bfs_dfs_topological_sort_articulation_points_bridges_strongly_connected_components_mst_kruskal_mst_p/SAGE-hints-run0-mtl/seed892-epoch=19-step=30000.pt" \
#   --load_layer 2
# done

for ((j = 0; j < $length; j++)); do
  python train_mtl.py --algorithms "strongly_connected_components" "${task_names[$j]}" \
  --cfg "./configs/SAGE.yml" --lr 5e-5 --hidden_dim 128 --gnn_layers 3 --enable_gru --devices 0 --batch_size 8 --epochs 50 --runs 1 --loss_weight_hint 2\
  --save_name "pairwise_layer_1" \
  --load_checkpoint_dir "bfs_dfs_topological_sort_articulation_points_bridges_strongly_connected_components_mst_kruskal_mst_p/SAGE-hints-run0-mtl/seed892-epoch=19-step=30000.pt" \
  --load_layer 1
done

for ((j = 0; j < $length; j++)); do
  python train_mtl.py --algorithms "mst_kruskal" "${task_names[$j]}" \
  --cfg "./configs/SAGE.yml" --lr 5e-5 --hidden_dim 128 --gnn_layers 3 --enable_gru --devices 0 --batch_size 8 --epochs 50 --runs 1 --loss_weight_hint 2\
  --save_name "pairwise_layer_1" \
  --load_checkpoint_dir "bfs_dfs_topological_sort_articulation_points_bridges_strongly_connected_components_mst_kruskal_mst_p/SAGE-hints-run0-mtl/seed892-epoch=19-step=30000.pt" \
  --load_layer 1
done