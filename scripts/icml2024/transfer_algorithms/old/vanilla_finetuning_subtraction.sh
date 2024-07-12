for size in 1000 2000 5000 10000 20000 50000
do
python train_v2.py --config configs/config.json \
    --algorithm subtraction --data_dir digit_5_borrow_True \
    --batch_size 256 --max_length 32 --num_of_instances 1000000 --generate_length 5\
    --lr 5e-4 --device 0 --num_layers 2 --random_init --tokenizer_dir gpt2_arithmetic\
    --train_size $size \
    --load_model_dir gpt2_addition_digit_5_carry_True_layers_2_train_size_5000 --reset_lm_head \
    --load_target --target_algorithm addition --target_data_dir digit_5_carry_True --target_generate_length 6 --runs 1
done