algorithms=("bubble_sort" "bfs" "insertion_sort" "dfs" "heapsort" "bellman_ford" "quicksort")

# pair wise training with projection

for i in "${!algorithms[@]}"
do
    CUDA_VISIBLE_DEVICES=2 python -m clrs.examples.run --algorithms "${algorithms[$i]}"\
    --use_projection --projection_dim 16 --processor_type "edge_t" --runs 1 --train_steps 2000
done
