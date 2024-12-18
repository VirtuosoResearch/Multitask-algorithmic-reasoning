python train_mtl.py --algorithms 'bfs' "dfs" "topological_sort" "articulation_points" "bridges" "strongly_connected_components" "mst_kruskal" "mst_prim" "dijkstra" "bellman_ford" 'dag_shortest_paths' "floyd_warshall"\
  --cfg "./configs/SAGE.yml" --lr 5e-5 --hidden_dim 128 --gnn_layers 3 --enable_gru --devices 1 --batch_size 8 --epochs 50 --runs 1 --loss_weight_hint 2\
  --save_name "test" --train_branched_network --tree_config_dir "group_3_layer_1.txt"

python train_mtl.py --algorithms 'bfs' "dfs" "topological_sort" "articulation_points" "bridges" "strongly_connected_components" "mst_kruskal" "mst_prim" "dijkstra" "bellman_ford" 'dag_shortest_paths' "floyd_warshall"\
  --cfg "./configs/SAGE.yml" --lr 5e-5 --hidden_dim 128 --gnn_layers 3 --enable_gru --devices 1 --batch_size 8 --epochs 50 --runs 1 --loss_weight_hint 2\
  --save_name "test" --train_branched_network --tree_config_dir "group_3_layer_2.txt"