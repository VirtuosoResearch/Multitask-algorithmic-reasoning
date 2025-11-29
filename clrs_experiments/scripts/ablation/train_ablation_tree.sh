CUDA_VISIBLE_DEVICES=0  python -m clrs.examples.run --algorithms "bfs","dfs","mst_kruskal","bellman_ford"\
    --use_branching_structure --branching_structure_dir "ablate_tree_1" --processor_type branching_edge_t\
    -use_projection --projection_dim 16 --num_layers 3 --runs 2

CUDA_VISIBLE_DEVICES=0  python -m clrs.examples.run --algorithms "bfs","dfs","mst_kruskal","bellman_ford"\
    --use_branching_structure --branching_structure_dir "ablate_tree_2" --processor_type branching_edge_t\
    -use_projection --projection_dim 16 --num_layers 3 --runs 2

CUDA_VISIBLE_DEVICES=0  python -m clrs.examples.run --algorithms "bfs","dfs","mst_kruskal","bellman_ford"\
    --use_branching_structure --branching_structure_dir "ablate_tree_3" --processor_type branching_edge_t\
    -use_projection --projection_dim 16 --num_layers 3 --runs 2

