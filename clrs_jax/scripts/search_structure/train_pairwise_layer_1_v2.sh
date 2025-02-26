task_names=("bubble_sort" "insertion_sort" "heapsort" "quicksort" "bfs" "dfs" "topological_sort" "articulation_points" "bridges" "strongly_connected_components" "mst_kruskal" "mst_prim" "dijkstra" "bellman_ford" 'dag_shortest_paths' "floyd_warshall")

length=${#task_names[@]}

for ((i = 4; i < $length; i++)); do
  for ((j = i + 1; j < $length; j++)); do
CUDA_VISIBLE_DEVICES=0 python -m clrs.examples.run --algorithms "${task_names[$i]}","${task_names[$j]}"\
    --processor_type 'edge_t' --hidden_size 48 --num_layers 6 --use_projection --projection_dim 4 
done
done