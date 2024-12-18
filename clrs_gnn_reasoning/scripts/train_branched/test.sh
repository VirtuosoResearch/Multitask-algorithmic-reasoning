python train_mtl.py --algorithms 'bfs' "dfs" \
  --cfg "./configs/SAGE.yml" --lr 5e-5 --hidden_dim 128 --gnn_layers 3 --enable_gru --devices 1 --batch_size 8 --epochs 3 --runs 1 --loss_weight_hint 2\
  --save_name "test" --train_branched_network
