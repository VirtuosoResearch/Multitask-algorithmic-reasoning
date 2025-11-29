# "bfs","dfs","articulation_points","bellman_ford","mst_kruskal"

# every time compute gradients for one algorithm
algo_list=("bfs" "dfs" "articulation_points" "bellman_ford" "mst_kruskal")

for i in {0..4}
do
CUDA_VISIBLE_DEVICES=1 python -m clrs.examples.fast_estimation_compute_gradients \
        --algorithms "${algo_list[$i]}" \
        --processor_type "edge_t" --num_layers 5 --hidden_size 192 \
        --use_projection --projection_dim 16 --batch_size 1 \
        --load_checkpoint_path "processor_edge_t_layers_5_dim_192_bfs_dfs_art_bel_mst" --train_steps 50\
        --change_algo_index $i --gradient_projection_dim 400
done