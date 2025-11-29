CUDA_VISIBLE_DEVICES=0 python -m clrs.examples.run \
    --algorithms "bfs","dfs","articulation_points","bellman_ford","mst_kruskal"\
    --processor_type "edge_t" --num_layers 5 --hidden_size 192 \
    --use_projection --projection_dim 16  --num_layers 5 --runs 1

# processor_edge_t_layers_5_dim_192_bfs_dfs_art_bel_mst
# CUDA_VISIBLE_DEVICES=0 python -m clrs.examples.run \
#             --algorithms {} --processor_type {} --num_layers {} --hidden_size {} \
#             --load_checkpoint_path {} --freeze_processor --freeze_layer {} \
#             --use_projection --projection_dim 16