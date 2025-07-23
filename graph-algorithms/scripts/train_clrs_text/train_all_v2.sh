python train_clrs_text.py --task_names "bfs" "dfs" "topological_sort" "articulation_points" "bridges" "strongly_connected_components" "mst_kruskal" "mst_prim" "dijkstra" "bellman_ford" 'dag_shortest_paths' "floyd_warshall" \
    --model_key "meta-llama/Llama-3.2-1B" \
    --devices 1 --batch_size 4 --inference_batch_size 4 --max_length 632 --max_output_length 480 --train_lengths 10 --test_lengths 10 --generate_output --runs 1 --lr 2e-5 \
    --save_name clrs_all_algorithms --epochs 10 --precision "bf16-true" --train_lora --lora_rank 16 --lora_alpha 128 \
    --few_shot_k 0 --downsample_ratio 0.01 --minimum_samples 200 --minimum_samples_validation 50