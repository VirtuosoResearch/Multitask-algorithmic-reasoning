for size in 50000 100000 200000 500000 1000000
do
python train.py --config configs/config.json \
    --algorithm quick_select --data_dir length_10 \
    --batch_size 256 --max_length 16 --num_of_instances 1000000 --generate_length 1\
    --lr 5e-4 --device 0 --num_layers 2 --random_init --tokenizer_dir gpt2_quick_select\
    --train_size 50000 \
    --load_model_dir gpt2_quick_select_length_5_layers_2_train_size_50000\
    --load_target --target_algorithm quick_select --target_data_dir length_5 --target_generate_length 1 --runs 1
done