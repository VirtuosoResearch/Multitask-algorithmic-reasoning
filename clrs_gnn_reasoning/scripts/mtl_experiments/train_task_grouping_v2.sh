python train_mtl.py --algorithms bfs dfs bridges strongly_connected_components dijkstra bellman_ford\
  --cfg "./configs/SAGE.yml" --lr 5e-5 --hidden_dim 128 --gnn_layers 3 --enable_gru --enbale_gru_task_wise --devices 2 --batch_size 8 --epochs 100 --runs 2 --loss_weight_hint 2\
  --save_name "clustered"

python train_mtl.py --algorithms topological_sort articulation_points mst_kruskal mst_prim dag_shortest_paths floyd_warshall\
  --cfg "./configs/SAGE.yml" --lr 5e-5 --hidden_dim 128 --gnn_layers 3 --enable_gru --enbale_gru_task_wise --devices 2 --batch_size 8 --epochs 100 --runs 2 --loss_weight_hint 2\
  --save_name "clustered"

python train_mtl.py --algorithms bfs bridges strongly_connected_components dijkstra bellman_ford\
  --cfg "./configs/SAGE.yml" --lr 5e-5 --hidden_dim 128 --gnn_layers 3 --enable_gru --enbale_gru_task_wise --devices 2 --batch_size 8 --epochs 100 --runs 2 --loss_weight_hint 2\
  --save_name "clustered"
