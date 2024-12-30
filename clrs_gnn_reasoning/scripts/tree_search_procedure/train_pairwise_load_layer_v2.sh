
task_names=("articulation_points" "bridges" "mst_prim" "floyd_warshall")
length=${#task_names[@]}

for ((i = 0; i < $length; i++)); do
for ((j = i+1; j < $length; j++)); do
  python train_mtl.py --algorithms "${task_names[$i]}" "${task_names[$j]}" \
  --cfg "./configs/SAGE.yml" --lr 5e-5 --hidden_dim 128 --gnn_layers 3 --enable_gru --devices 0 --batch_size 8 --epochs 50 --runs 1 --loss_weight_hint 2\
  --save_name "pairwise_layer_1" \
  --load_checkpoint_dir "bfs_dfs_topological_sort_articulation_points_bridges_strongly_connected_components_mst_kruskal_mst_p/SAGE-hints-run0-mtl/seed892-epoch=19-step=30000.pt" \
  --load_layer_checkpoint_dirs "articulation_points_bridges_mst_prim_floyd_warshall/SAGE-hints-run0-mtl/seed892-epoch=2-step=1500.pt" \
  --load_layers 1 \
  --train_layer 2
done
done