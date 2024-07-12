for alpha in 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 
do
python train.py --config configs/config.json \
    --algorithm sorting --data_dir length_10 \
    --batch_size 256 --max_length 32 --num_of_instances 1000000 --generate_length 10\
    --lr 5e-4 --device 2 --num_layers 2 --random_init --tokenizer_dir gpt2_sort_100\
    --train_size 5000 --max_steps 0\
    --model_interpolate \
    --model_interpolate_start gpt2_sorting_length_5_layers_2_train_size_20000\
    --model_interpolate_end gpt2_sorting_length_10_layers_2_train_size_50000\
    --model_interpolate_alpha $alpha\
    --load_target --target_algorithm sorting --target_data_dir length_5 --target_generate_length 5 --runs 2
done