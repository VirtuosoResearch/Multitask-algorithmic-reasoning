CUDA_VISIBLE_DEVICES=0 python -m clrs.examples.fast_estimation_linear_regression\
        --algorithms "bfs","dfs","bellman_ford","dijkstra","mst_prim"\
        --processor_type "gat" --num_layers 5 --hidden_size 192\
        --use_projection --projection_dim 16 --batch_size 1 \
        --load_checkpoint_path "processor_gat_layers_5_dim_192_bfs_dfs_bel_dij_mst_mst"\
        --layer 0 --gradient_projection_dim 400 --regularization_lambda 1e3 \
        --num_subsets 10 --num_subset_size 4