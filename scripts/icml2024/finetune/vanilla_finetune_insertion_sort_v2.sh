for size in 100000
do
python train_multitask.py --config configs/config.json \
    --algorithm insertion_sort --data_dir inter_results_length_10 \
    --batch_size 256 --max_length 32 --num_of_instances 1000000 --generate_length 10\
    --lr 1e-3 --device 1 --num_layers 2 --random_init --tokenizer_dir gpt2_sort_100\
    --train_size $size --max_steps 200000\
    --load_model_dir gpt2_insertion_sort_inter_results_length_5_layers_2_train_size_20000_load_model \
    --load_target --target_algorithm insertion_sort --target_data_dir inter_results_length_5 --target_generate_length 5 --runs 1
done
