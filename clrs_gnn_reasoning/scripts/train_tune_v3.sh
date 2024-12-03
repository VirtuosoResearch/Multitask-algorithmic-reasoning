
python train.py --algorithm "bfs" --cfg "./configs/GIN.yml" --lr 0.0004239\
    --hidden_dim 128 --gnn_layers 1 --enable_gru --devices 0 --batch_size 32 --epochs 400

python train.py --algorithm "dijkstra" --cfg "./configs/GINE.yml" --lr 5e-5\
    --hidden_dim 128 --gnn_layers 1 --enable_gru --devices 0 
