for d in 3.0 2.6 2.3 2.0 1.6 1.3 1.0
do
python train.py --config configs/config.json \
    --algorithm sorting --data_dir length_10 \
    --batch_size 256 --max_length 32 --num_of_instances 1000000 --generate_length 10\
    --lr 5e-4 --device 2 --num_layers 2 --random_init --tokenizer_dir gpt2_sort_100\
    --train_size 20000 --max_steps 10000\
    --load_model_dir gpt2_sorting_length_5_layers_2_train_size_20000\
    --train_constraint --reg_method constraint --reg_attention $d --reg_linear $d --reg_predictor $d\
    --load_target --target_algorithm sorting --target_data_dir length_5 --target_generate_length 5 --runs 2
done