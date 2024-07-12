for d in 25 30 18 15 # 20
do
# for alpha in 0.02 0.05 0.1
# do
python train_multitask.py --config configs/config.json \
    --algorithms insertion_sort --data_dirs inter_results_length_5 \
    --batch_size 256 --max_length 32 --num_of_instances 1000000 --generate_length 5\
    --lr 5e-3 --device 0 --num_layers 2 --random_init --tokenizer_dir gpt2_sort_100\
    --train_size 10000 --eval_epoch 5 --max_steps 20000 \
    --load_model_dir gpt2_bubble_sort_inter_results_length_5_layers_2_train_size_50000\
    --load_target --target_algorithm bubble_sort --target_data_dir inter_results_length_5 --target_generate_length 5 --runs 2\
    --train_constraint_allocation --reg_total $d --early_stop 200 --allocation_alpha 0.01
done
# done