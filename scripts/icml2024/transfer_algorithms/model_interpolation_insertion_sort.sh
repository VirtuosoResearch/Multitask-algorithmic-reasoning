for alpha in 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 
do
python train_multitask.py --config configs/config.json \
    --algorithms insertion_sort --data_dirs inter_results_length_5 \
    --batch_size 256 --max_length 32 --num_of_instances 1000000 --generate_length 5\
    --lr 5e-3 --device 3 --num_layers 2 --random_init --tokenizer_dir gpt2_sort_100\
    --train_size 10000 --eval_epoch 5 --max_steps 0 \
    --model_interpolate \
    --model_interpolate_start gpt2_bubble_sort_inter_results_length_5_layers_2_train_size_50000\
    --model_interpolate_end gpt2_insertion_sort_inter_results_length_5_layers_2_train_size_50000\
    --model_interpolate_alpha $alpha\
    --load_target --target_algorithm bubble_sort --target_data_dir inter_results_length_5 --target_generate_length 5 --runs 1
done
