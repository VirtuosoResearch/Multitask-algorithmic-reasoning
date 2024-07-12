for d in 1 2 4 8 12
do
python train.py --config configs/config.json \
    --algorithm sorting --data_dir length_10 \
    --batch_size 256 --max_length 32 --num_of_instances 1000000 --generate_length 10\
    --lr 5e-4 --device 1 --num_layers 2 --random_init --tokenizer_dir gpt2_sort_100\
    --train_size 20000 --max_steps 10000\
    --load_model_dir gpt2_sorting_length_5_layers_2_train_size_20000\
    --train_constraint_allocation --reg_total $d\
    --load_target --target_algorithm sorting --target_data_dir length_5 --target_generate_length 5 --runs 2
done