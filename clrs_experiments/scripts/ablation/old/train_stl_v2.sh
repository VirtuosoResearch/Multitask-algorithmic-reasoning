for algorithm in "mst_kruskal" "mst_prim" "dijkstra" "bellman_ford" 'dag_shortest_paths' "floyd_warshall"
do
for dim in 16
do
CUDA_VISIBLE_DEVICES=1 python -m clrs.examples.run --algorithms $algorithm\
    --processor_type "edge_t" --num_layers 5
done
done
