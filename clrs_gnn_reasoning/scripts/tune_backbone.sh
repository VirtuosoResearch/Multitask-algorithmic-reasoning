
for algorithm in "bfs" "bellman_ford" "mst_kruskal" "dijkstra"
do
for backbone in "GCN" "SAGE" "SAGE_normalize"
do
python train.py --algorithm $algorithm --cfg "./configs/${backbone}.yml" --lr 5e-5\
        --hidden_dim 128 --gnn_layers 3 --enable_gru --devices 1 --batch_size 8 --epochs 100 --runs 2 --loss_weight_hint 2
done
done


# python train.py --algorithm "bfs" --cfg "./configs/SAGE.yml" --lr 5e-5\
#         --hidden_dim 128 --gnn_layers 3 --enable_gru --devices 1 --batch_size 8 --epochs 100 --runs 1 --loss_weight_hint 2

