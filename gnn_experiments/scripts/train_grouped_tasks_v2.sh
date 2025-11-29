# Group 4

python train_mtl.py --algorithms bfs bridges strongly_connected_components\
  --cfg "./configs/SAGE.yml" --lr 5e-5 --hidden_dim 128 --gnn_layers 3 --enable_gru --devices 2 --batch_size 8 --epochs 50 --runs 1 --loss_weight_hint 2\
  --save_name "clustered"

python train_mtl.py --algorithms topological_sort articulation_points mst_kruskal mst_prim dijkstra dag_shortest_paths floyd_warshall\
  --cfg "./configs/SAGE.yml" --lr 5e-5 --hidden_dim 128 --gnn_layers 3 --enable_gru --devices 2 --batch_size 8 --epochs 50 --runs 1 --loss_weight_hint 2\
  --save_name "clustered"
