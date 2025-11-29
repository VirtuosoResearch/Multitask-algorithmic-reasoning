# python -m clrs.examples.run --algorithms "bfs","dfs","topological_sort","articulation_points","bridges","strongly_connected_components","mst_kruskal","mst_prim","dijkstra","bellman_ford",'dag_shortest_paths',"floyd_warshall"\
#     --train_steps 50

# python -m clrs.examples.run --algorithms "bfs","dfs","topological_sort","articulation_points","bridges","strongly_connected_components","mst_kruskal","mst_prim","dijkstra","bellman_ford",'dag_shortest_paths',"floyd_warshall"\
#     --use_branching_structure --branching_structure_dir "tree_structure_1" --processor_type branching_edge_t --train_steps 10

# python -m clrs.examples.run --algorithms "bfs","dfs","topological_sort","articulation_points","bridges","strongly_connected_components","mst_kruskal","mst_prim","dijkstra","bellman_ford",'dag_shortest_paths',"floyd_warshall"\
#     --use_branching_structure --branching_structure_dir "tree_structure_2" --processor_type branching_edge_t --train_steps 10

# python -m clrs.examples.run --algorithms "bfs","dfs","topological_sort","articulation_points","bridges","strongly_connected_components","mst_kruskal","mst_prim","dijkstra","bellman_ford",'dag_shortest_paths',"floyd_warshall"\
#     --use_branching_structure --branching_structure_dir "tree_structure_3" --processor_type branching_edge_t --train_steps 10

# python -m clrs.examples.run --algorithms "bfs","dfs","topological_sort","articulation_points","bridges","strongly_connected_components","mst_kruskal","mst_prim","dijkstra","bellman_ford",'dag_shortest_paths',"floyd_warshall"\
#     --use_branching_structure --branching_structure_dir "grouping_structure_1" --processor_type branching_edge_t --train_steps 10

# python -m clrs.examples.run --algorithms "bfs","dfs","topological_sort","articulation_points","bridges","strongly_connected_components","mst_kruskal","mst_prim","dijkstra","bellman_ford",'dag_shortest_paths',"floyd_warshall"\
#     --use_branching_structure --branching_structure_dir "grouping_structure_2" --processor_type branching_edge_t --train_steps 10

# python -m clrs.examples.run --algorithms "bfs","dfs","topological_sort","articulation_points","bridges","strongly_connected_components","mst_kruskal","mst_prim","dijkstra","bellman_ford",'dag_shortest_paths',"floyd_warshall"\
#     --use_branching_structure --branching_structure_dir "grouping_structure_3" --processor_type branching_edge_t --train_steps 10

# python -m clrs.examples.run --algorithms "bfs","dfs","topological_sort","articulation_points","bridges","strongly_connected_components","mst_kruskal","mst_prim","dijkstra","bellman_ford",'dag_shortest_paths',"floyd_warshall"\
#     --use_branching_structure --branching_structure_dir "grouping_structure_4" --processor_type branching_edge_t --train_steps 10



# CUDA_VISIBLE_DEVICES=0 python -m clrs.examples.run --algorithms bfs --train_steps 10 --batch_size 4

# python -m clrs.examples.run --algorithms "bfs","dfs","topological_sort","articulation_points","bridges","strongly_connected_components","mst_kruskal","mst_prim","dijkstra","bellman_ford",'dag_shortest_paths',"floyd_warshall"\
#     --use_branching_structure --branching_structure_dir "grouping_structure_1" --processor_type branching_edge_t --train_steps 10 --batch_size 2 --num_layers 6 --projection_dim 16

python -m clrs.examples.run --algorithms "bfs","dfs","topological_sort","articulation_points","bridges","strongly_connected_components","mst_kruskal","mst_prim","dijkstra","bellman_ford",'dag_shortest_paths',"floyd_warshall"\
    --use_branching_structure --branching_structure_dir "tree_structure_1_new" --processor_type branching_edge_t --train_steps 10 --batch_size 2 --num_layers 6 --projection_dim 16

python -m clrs.examples.run --algorithms "bfs","dfs","topological_sort","articulation_points","bridges","strongly_connected_components","mst_kruskal","mst_prim","dijkstra","bellman_ford",'dag_shortest_paths',"floyd_warshall"\
    --train_steps 10 --batch_size 2 --num_layers 6 --projection_dim 16

for algorithm in 'bfs' "dfs" "topological_sort" "articulation_points" "bridges" "strongly_connected_components" "mst_kruskal" "mst_prim" "dijkstra" "bellman_ford" 'dag_shortest_paths' "floyd_warshall"
do
python -m clrs.examples.run --algorithms $algorithm --train_steps 10 --batch_size 2 --num_layers 6 --projection_dim 16
done

python -m clrs.examples.run --algorithms 'floyd_warshall' --train_steps 10 --batch_size 2 --num_layers 6 --projection_dim 16

