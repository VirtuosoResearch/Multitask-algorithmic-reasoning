task_names=("bfs" "dfs" "bellman_ford" "dijkstra" "mst_prim")
length=${#task_names[@]}


for layer in 0 1 2
do
for ratio in 0.02 0.04 0.06 0.08 0.1
do
for ((i = 0; i < $length; i++)); do
CUDA_VISIBLE_DEVICES=0 python -m clrs.examples.fast_estimation_compute_outputs\
    --algorithms "${task_names[$i]}" --change_algo_index $i --processor_type "gat" --num_layers 5 --hidden_size 192\
    --use_projection --projection_dim 16 --batch_size 1 \
    --load_checkpoint_path "processor_gat_layers_5_dim_192_bfs_dfs_bel_dij_mst_mst" --gradient_projection_dim 400\
    --layer $layer --runs 1 --perturb_ratio $ratio --train_steps 20 --runs 5
done
done
done

for layer in 0 1 2
do
for ratio in 0.02 0.04 0.06 0.08 0.1
do
for ((i = 0; i < $length; i++)); do
CUDA_VISIBLE_DEVICES=0 python -m clrs.examples.fast_estimation_compute_outputs\
    --algorithms "${task_names[$i]}" --change_algo_index $i --processor_type "triplet_mpnn" --num_layers 5 --hidden_size 192\
    --use_projection --projection_dim 16 --batch_size 1 \
    --load_checkpoint_path "processor_triplet_mpnn_layers_5_dim_192_bfs_dfs_bel_dij_mst_mst" --gradient_projection_dim 400\
    --layer $layer --runs 1 --perturb_ratio $ratio --train_steps 20 --runs 5
done
done
done
