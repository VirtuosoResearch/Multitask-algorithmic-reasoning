for algorithm in "dfs" "dijkstra" "mst_prim" "topological_sort" # 'bfs' 'bellman_ford'
do
for processor_type in 'mpnn' 'triplet_gmpnn' 'edge_t'
do
CUDA_VISIBLE_DEVICES=0 python -m clrs.examples.run --algorithms $algorithm --use_projection --projection_dim 16 --processor_type $processor_type
done
done




CUDA_VISIBLE_DEVICES=0 python -m clrs.examples.run --algorithms 'dfs' --use_projection --projection_dim 16 --processor_type 'edge_t' --train_steps 10
CUDA_VISIBLE_DEVICES=1 python -m clrs.examples.run --algorithms "bfs"\
    --processor_type "edge_t" --num_layers 5 --freeze_processor --freeze_layers 2 