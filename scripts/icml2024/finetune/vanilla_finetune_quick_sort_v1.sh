for size in 50000 100000
do
python train_multitask.py --config configs/config.json \
    --algorithm quick_sort --data_dir inter_results_length_10 \
    --batch_size 256 --max_length 32 --num_of_instances 1000000 --generate_length 10\
    --lr 5e-4 --device 3 --num_layers 2 --random_init --tokenizer_dir gpt2_sort_100\
    --train_size 20000 --max_steps 200000 \
    --load_model_dir gpt2_quick_sort_inter_results_length_5_layers_2_train_size_20000_load_model \
    --load_target --target_algorithm quick_sort --target_data_dir inter_results_length_5 --target_generate_length 5 --runs 1
done
