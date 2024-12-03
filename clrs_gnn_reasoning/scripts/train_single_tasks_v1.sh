for algorithm in 'bfs' "dfs" "topological_sort" "articulation_points" "bridges" "strongly_connected_components"
do
for layer in 1 2 3 
do
    python train.py --algorithm "bfs" --cfg "./configs/GIN.yml" --lr 0.0004239\
            --hidden_dim 128 --gnn_layers $layer --enable_gru --devices 0 --batch_size 8 --epochs 100 --runs 3
done
done
