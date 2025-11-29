CUDA_VISIBLE_DEVICES=1 python -m clrs.examples.fast_estimation_linear_regression \
        --algorithms "bfs","dfs","articulation_points","bellman_ford","mst_kruskal"\
        --processor_type "edge_t" --num_layers 5 --hidden_size 192 \
        --use_projection --projection_dim 16 --batch_size 1 \
        --load_checkpoint_path "processor_edge_t_layers_5_dim_192_bfs_dfs_art_bel_mst"\
        --layer 0 --gradient_projection_dim 400 --regularization_lambda 1e3 \
        --num_subsets 20 --num_subset_size 3