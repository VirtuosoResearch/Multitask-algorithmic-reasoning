# bfs
# dfs topological_sort bridges strongly_connected_components dijkstra floyd_warshall
# mst_kruskal mst_prim bellman_ford dag_shortest_paths
# articulation_points

CUDA_VISIBLE_DEVICES=1 python -m clrs.examples.run \
    --algorithms "bfs"\
    --use_projection --projection_dim 16 --processor_type "edge_t" --num_layers 5 --runs 1 --train_steps 2000

CUDA_VISIBLE_DEVICES=1 python -m clrs.examples.run \
    --algorithms "dfs","topological_sort","bridges","strongly_connected_components","dijkstra","floyd_warshall"\
    --use_projection --projection_dim 16 --processor_type "edge_t" --num_layers 5 --runs 1 --train_steps 2000

CUDA_VISIBLE_DEVICES=1 python -m clrs.examples.run \
    --algorithms "mst_kruskal","mst_prim","bellman_ford","dag_shortest_paths"\
    --use_projection --projection_dim 16 --processor_type "edge_t" --num_layers 5 --runs 1 --train_steps 2000

CUDA_VISIBLE_DEVICES=1 python -m clrs.examples.run \
    --algorithms "articulation_points"\
    --use_projection --projection_dim 16 --processor_type "edge_t" --num_layers 5 --runs 1 --train_steps 2000

