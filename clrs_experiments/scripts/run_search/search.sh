CUDA_VISIBLE_DEVICES=1 python branchnn_search.py \
        --algorithms "bfs" "dfs" "bellman_ford" "mst_kruskal" \
        --processor_type "edge_t" --num_layers 3 --hidden_size 192 \
        --gradient_projection_dim 400 --num_subsets 20 --subset_size 3