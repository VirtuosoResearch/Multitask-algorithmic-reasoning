for algorithm in "floyd_warshall" "mst_kruskal" 
do
for layer in 3 
do
    python train.py --algorithm $algorithm --cfg "./configs/GIN.yml" --lr 5e-5\
            --hidden_dim 128 --gnn_layers $layer --enable_gru --devices 1 --batch_size 8 --epochs 100 --runs 3 --loss_weight_hint 2
done
done


# python train.py --algorithm "bfs" --cfg "./configs/GIN.yml" --lr 5e-5\
#             --hidden_dim 128 --gnn_layers 3 --enable_gru --devices 1 --batch_size 8 --epochs 100 --runs 3 --loss_weight_hint 2