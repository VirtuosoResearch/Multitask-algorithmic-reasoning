# bfs articulation_points
# dfs topological_sort bridges mst_kruskal mst_prim dijkstra bellman_ford floyd_warshall
# strongly_connected_components dag_shortest_paths

CUDA_VISIBLE_DEVICES=0 python -m clrs.examples.run \
    --algorithms "bfs","articulation_points"\
    --use_projection --projection_dim 16 --processor_type "edge_t" --num_layers 5 --runs 1 --train_steps 2000

CUDA_VISIBLE_DEVICES=0 python -m clrs.examples.run \
    --algorithms "dfs","topological_sort","bridges","mst_kruskal","mst_prim","dijkstra","bellman_ford","floyd_warshall"\
    --use_projection --projection_dim 16 --processor_type "edge_t" --num_layers 5 --runs 1 --train_steps 2000

CUDA_VISIBLE_DEVICES=0 python -m clrs.examples.run \
    --algorithms "strongly_connected_components","dag_shortest_paths"\
    --use_projection --projection_dim 16 --processor_type "edge_t" --num_layers 5 --runs 1 --train_steps 2000
