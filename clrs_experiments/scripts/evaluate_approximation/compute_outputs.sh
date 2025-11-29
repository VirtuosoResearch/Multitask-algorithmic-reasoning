for layer in 0 1 2
do
for ratio in 0.02 0.04 0.06 0.08 0.1
do
CUDA_VISIBLE_DEVICES=1 python -m clrs.examples.fast_estimation_compute_outputs --algorithms "bfs"\
    --use_projection --projection_dim 16 --processor_type "mpnn" --batch_size 1 --load_checkpoint_path "processor_mpnn_layers_3_dim_192_bfs_dfs_top_art_bri_str_mst_mst_dij_bel_dag_flo" --train_steps 20\
    --layer $layer --runs 1 --perturb_ratio $ratio --train_steps 20
done
done