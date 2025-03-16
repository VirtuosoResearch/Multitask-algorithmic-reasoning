for algorithm in "mst_kruskal" "mst_prim" "dijkstra" "bellman_ford" 'dag_shortest_paths' "floyd_warshall"
do
for dim in 16
do
CUDA_VISIBLE_DEVICES=1 python -m clrs.examples.run --algorithms $algorithm --use_projection --projection_dim $dim
done
done


