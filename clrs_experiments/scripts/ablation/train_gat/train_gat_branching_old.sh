# bfs articulation_points
# dfs topological_sort bridges mst_kruskal mst_prim dijkstra bellman_ford floyd_warshall
# strongly_connected_components dag_shortest_paths

CUDA_VISIBLE_DEVICES=1 python -m clrs.examples.run \
    --algorithms "bfs","articulation_points"\
    --use_projection --projection_dim 16 --processor_type "gat" --num_layers 5 --runs 1 --train_steps 2000

CUDA_VISIBLE_DEVICES=1 python -m clrs.examples.run \
    --algorithms "dfs","topological_sort","bridges","mst_kruskal","mst_prim","dijkstra","bellman_ford","floyd_warshall"\
    --use_projection --projection_dim 16 --processor_type "gat" --num_layers 5 --runs 1 --train_steps 2000

CUDA_VISIBLE_DEVICES=1 python -m clrs.examples.run \
    --algorithms "strongly_connected_components","dag_shortest_paths"\
    --use_projection --projection_dim 16 --processor_type "gat" --num_layers 5 --runs 1 --train_steps 2000

CUDA_VISIBLE_DEVICES=1 python -m clrs.examples.run \
    --algorithms "bfs"\
    --use_projection --projection_dim 16 --processor_type "gat" --num_layers 5 --runs 1 --train_steps 2000

CUDA_VISIBLE_DEVICES=1 python -m clrs.examples.run \
    --algorithms "dfs","topological_sort","bridges","strongly_connected_components","dijkstra","floyd_warshall"\
    --use_projection --projection_dim 16 --processor_type "gat" --num_layers 5 --runs 1 --train_steps 2000

CUDA_VISIBLE_DEVICES=1 python -m clrs.examples.run \
    --algorithms "mst_kruskal","mst_prim","bellman_ford","dag_shortest_paths"\
    --use_projection --projection_dim 16 --processor_type "gat" --num_layers 5 --runs 1 --train_steps 2000

CUDA_VISIBLE_DEVICES=1 python -m clrs.examples.run \
    --algorithms "articulation_points"\
    --use_projection --projection_dim 16 --processor_type "gat" --num_layers 5 --runs 1 --train_steps 2000

CUDA_VISIBLE_DEVICES=1 python -m clrs.examples.run \
    --algorithms "bfs","dfs","topological_sort","articulation_points","bridges","strongly_connected_components","mst_kruskal","mst_prim","dijkstra","bellman_ford","dag_shortest_paths","floyd_warshall"\
    --use_projection --projection_dim 16 --processor_type "gat" --num_layers 5 --runs 1 --train_steps 2000