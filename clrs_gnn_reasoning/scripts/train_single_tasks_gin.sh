for algorithm in "mst_kruskal" "mst_prim" "dijkstra" "bellman_ford" 'dag_shortest_paths' "floyd_warshall" # 'bfs' "dfs" "topological_sort" "articulation_points" "bridges" "strongly_connected_components" 
do
for layer in 3 
do
    python train.py --algorithm $algorithm --cfg "./configs/GIN.yml" --lr 5e-5\
            --hidden_dim 128 --gnn_layers $layer --enable_gru --devices 1 --batch_size 8 --epochs 50 --runs 2 --loss_weight_hint 2
done
done

# python train.py --algorithm "bfs" --cfg "./configs/GINE.yml" --lr 0.0004239\
#         --hidden_dim 128 --gnn_layers 1 --enable_gru --devices 0 --batch_size 8 --epochs 100 --runs 3 --use_complete_graph

# python train.py --algorithm "topological_sort" --cfg "./configs/GIN.yml" --lr 1e-5\
#     --hidden_dim 128 --gnn_layers 2 --enable_gru --devices 0 --batch_size 8 --epochs 100 --runs 3 --loss_weight_hint 2

# python train.py --algorithm "dfs" --cfg "./configs/GCN.yml" --lr 5e-5\
#         --hidden_dim 128 --gnn_layers 3 --enable_gru --devices 1 --batch_size 8 --epochs 100 --runs 1 --loss_weight_hint 2
