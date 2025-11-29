for algorithm in "bubble_sort" "insertion_sort" "heapsort" "quicksort" "bfs" "dfs" "topological_sort" "articulation_points" "bridges" "strongly_connected_components" "mst_kruskal" "mst_prim" "dijkstra" "bellman_ford" 'dag_shortest_paths'
do
for dim in 16
do
CUDA_VISIBLE_DEVICES=2 python -m clrs.examples.run --algorithms "floyd_warshall","${algorithm}"\
    --processor_type "edge_t" --num_layers 5
done
done