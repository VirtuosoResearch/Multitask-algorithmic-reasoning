python train_mtl.py --algorithms mst_kruskal dijkstra bellman_ford \
    --cfg "./configs/SAGE.yml" --lr 5e-5 --hidden_dim 128 --gnn_layers 3 --enable_gru --devices 2 --batch_size 8 --epochs 50 --runs 3 --loss_weight_hint 2\
    --save_name "mtl" \
    --load_checkpoint_dir "bfs_dfs_topological_sort_articulation_points_bridges_strongly_connected_components_mst_kruskal_mst_p/SAGE-hints-run0-mtl/seed892-epoch=19-step=30000.pt" \
    --load_layer 1
