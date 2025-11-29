task_names=('bfs' "dfs" "topological_sort" "articulation_points" "bridges" "strongly_connected_components" "mst_kruskal" "mst_prim" "dijkstra" "bellman_ford" 'dag_shortest_paths' "floyd_warshall")

length=${#task_names[@]}

for ((i = 6; i < $length; i++)); do
  for ((j = i + 1; j < $length; j++)); do
    python train_mtl.py --algorithms "${task_names[$i]}" "${task_names[$j]}" \
    --cfg "./configs/SAGE.yml" --lr 5e-5 --hidden_dim 128 --gnn_layers 3 --enable_gru --devices 2 --batch_size 8 --epochs 50 --runs 1 --loss_weight_hint 2\
    --save_name "pairwise"
  done
done

# python train_mtl.py --algorithms 'bfs' "dfs" \
#   --cfg "./configs/SAGE.yml" --lr 5e-5 --hidden_dim 128 --gnn_layers 3 --enable_gru --devices 2 --batch_size 8 --epochs 10 --runs 1 --loss_weight_hint 2\
#   --save_name "pairwise"