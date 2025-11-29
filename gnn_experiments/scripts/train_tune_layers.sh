for layer in 8 9 10
do
for algorithm in "dfs" "topological_sort" "dijkstra" # "strongly_connected_components" "mst_kruskal" "mst_prim" "dijkstra" "bellman_ford" 'dag_shortest_paths' "floyd_warshall" # 'bfs' "dfs" "topological_sort" "articulation_points" "bridges"
do
    python train.py --algorithm $algorithm --cfg "./configs/SAGE.yml" --lr 5e-5\
            --hidden_dim 128 --gnn_layers $layer --enable_gru --devices 1 --batch_size 8 --epochs 50 --runs 1 --loss_weight_hint 2
done
done
