task_names=('bfs' "dfs" "topological_sort" "articulation_points" "bridges" "strongly_connected_components" "mst_kruskal" "mst_prim" "dijkstra" "bellman_ford" 'dag_shortest_paths' "floyd_warshall")

# python train_mtl.py --algorithms 'bfs' "dfs" "topological_sort" \
#   --cfg "./configs/SAGE.yml" --lr 5e-5 --hidden_dim 128 --gnn_layers 3 --enable_gru --devices 1 --batch_size 8 --epochs 100 --runs 1 --loss_weight_hint 2\
#   --save_name "higher_order"

# python train_mtl.py --algorithms 'bfs' "dfs" "topological_sort" "articulation_points"\
#   --cfg "./configs/SAGE.yml" --lr 5e-5 --hidden_dim 128 --gnn_layers 3 --enable_gru --devices 1 --batch_size 8 --epochs 100 --runs 1 --loss_weight_hint 2\
#   --save_name "higher_order"

python train_mtl.py --algorithms 'bfs' "dfs" "topological_sort" "articulation_points"  "bridges"\
  --cfg "./configs/SAGE.yml" --lr 5e-5 --hidden_dim 128 --gnn_layers 3 --enable_gru --devices 1 --batch_size 8 --epochs 100 --runs 1 --loss_weight_hint 2\
  --save_name "higher_order"

python train_mtl.py --algorithms 'bfs' "dfs" "topological_sort" "articulation_points"  "bridges" "strongly_connected_components"\
  --cfg "./configs/SAGE.yml" --lr 5e-5 --hidden_dim 128 --gnn_layers 3 --enable_gru --devices 1 --batch_size 8 --epochs 100 --runs 1 --loss_weight_hint 2\
  --save_name "higher_order"

python train_mtl.py --algorithms 'bfs' "dfs" "topological_sort" "articulation_points"  "bridges" "strongly_connected_components" "mst_kruskal"\
  --cfg "./configs/SAGE.yml" --lr 5e-5 --hidden_dim 128 --gnn_layers 3 --enable_gru --devices 1 --batch_size 8 --epochs 100 --runs 1 --loss_weight_hint 2\
  --save_name "higher_order"

python train_mtl.py --algorithms 'bfs' "dfs" "topological_sort" "articulation_points"  "bridges" "strongly_connected_components" "mst_kruskal" "mst_prim"\
  --cfg "./configs/SAGE.yml" --lr 5e-5 --hidden_dim 128 --gnn_layers 3 --enable_gru --devices 1 --batch_size 8 --epochs 100 --runs 1 --loss_weight_hint 2\
  --save_name "higher_order"