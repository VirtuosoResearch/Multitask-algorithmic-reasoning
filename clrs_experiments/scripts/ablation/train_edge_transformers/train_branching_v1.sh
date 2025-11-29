CUDA_DEVICE=1

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python -m clrs.examples.run \
    --algorithms "bfs","dfs","topological_sort","articulation_points","bridges","strongly_connected_components","mst_kruskal","mst_prim","dijkstra","bellman_ford",'dag_shortest_paths',"floyd_warshall"\
    --use_branching_structure --branching_structure_dir "tree_structure_final" --processor_type branching_edge_t \
    --num_layers 5 \
    --runs 3 \
    --train_steps 10000 \
    --save_name "branching_edge_t_layers_5"