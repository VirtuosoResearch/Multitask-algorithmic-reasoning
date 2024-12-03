for lr in 1e-5 2e-5 5e-5 1e-4
do
python train.py --algorithm "bellman_ford" --cfg "./configs/GINE.yml" --lr $lr\
    --hidden_dim 128 --gnn_layers 1 --devices 0

python train.py --algorithm "bellman_ford" --cfg "./configs/GINE.yml" --lr $lr\
    --hidden_dim 128 --gnn_layers 1 --enable_gru --devices 0 
done