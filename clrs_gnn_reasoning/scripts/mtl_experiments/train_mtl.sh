for dim in 128 256 512 1024
do
python train_mtl.py --algorithms 'bfs' "dfs" "topological_sort" "articulation_points" "bridges" "strongly_connected_components" "mst_kruskal" "mst_prim" "dijkstra" "bellman_ford" 'dag_shortest_paths' "floyd_warshall"\
  --cfg "./configs/SAGE.yml" --lr 5e-5 --hidden_dim $dim --gnn_layers 3 --enable_gru --enbale_gru_task_wise --devices 1 --batch_size 8 --epochs 100 --runs 2 --loss_weight_hint 2\
  --save_name "mtl_${dim}"
done

for expert in 2 4 8
do
python train_mtl.py --algorithms 'bfs' "dfs" "topological_sort" "articulation_points" "bridges" "strongly_connected_components" "mst_kruskal" "mst_prim" "dijkstra" "bellman_ford" 'dag_shortest_paths' "floyd_warshall" \
  --cfg "./configs/SAGE.yml" --lr 5e-5 --hidden_dim 128 --gnn_layers 3 --enable_gru --enbale_gru_task_wise --devices 1 --batch_size 8 --epochs 100 --runs 2 --loss_weight_hint 2\
  --save_name "moe_${expert}" --train_mmoe --num_experts $expert
done