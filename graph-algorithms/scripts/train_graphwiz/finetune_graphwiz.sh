task_names=("shortest" "topology" "triangle" "flow" "hamilton" "substructure" "connectivity" "bipartite" "cycle") 

for task_name in "${task_names[@]}"
do
python train_graphwiz.py --task_names "$task_name" \
    --model_key "GraphWiz/LLaMA2-7B-DPO" \
    --devices 1 --batch_size 1 --accumulate 4 --inference_batch_size 2 --max_length 2800 --max_output_length 1024 --generate_output --runs 1 --lr 5e-5 --eval_split 0\
    --save_name graphwiz_finetuning --epochs 5 --train_lora --lora_rank 16 --lora_alpha 128 --precision "bf16-true" --write_results --only_evaluate_test_set
done