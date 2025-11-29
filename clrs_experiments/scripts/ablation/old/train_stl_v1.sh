for algorithm in 'bfs' "dfs" "topological_sort" "articulation_points" "bridges" "strongly_connected_components"
do
for dim in 16
do
CUDA_VISIBLE_DEVICES=0 python -m clrs.examples.run --algorithms $algorithm\
    --processor_type "edge_t" --num_layers 5
done
done


