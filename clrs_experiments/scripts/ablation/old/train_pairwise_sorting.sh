algorithms=("bubble_sort" "bfs" "insertion_sort" "dfs" "heapsort" "bellman_ford" "quicksort")

# pair wise training with projection

for i in "${!algorithms[@]}"
do
for j in "${!algorithms[@]}"
do
if [ $i -lt $j ]
then
    CUDA_VISIBLE_DEVICES=0 python -m clrs.examples.run --algorithms "${algorithms[$i]}","${algorithms[$j]}"\
    --use_projection --projection_dim 16 --processor_type "edge_t" --runs 1 --train_steps 2000
else
    continue
fi
done
done
