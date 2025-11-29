CUDA_DEVICE=0

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python -m clrs.examples.run \
    --algorithms "bfs","dfs","topological_sort","articulation_points","bridges","strongly_connected_components","mst_kruskal","mst_prim","dijkstra","bellman_ford",'dag_shortest_paths',"floyd_warshall"\
    --use_branching_structure --branching_structure_dir "gat_tree" --processor_type branching_gatv2 \
    --num_layers 3 \
    --runs 3 \
    --train_steps 10000 \
    --save_name "branching_gatv2_layers_3"