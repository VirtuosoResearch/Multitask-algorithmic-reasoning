# for dim in 128 256 512 1024
# do
# python train.py --algorithm "bellman_ford" --cfg "./configs/GINE.yml" --lr 5e-5\
#     --hidden_dim $dim --gnn_layers 1 --enable_gru --devices 1 
# done
