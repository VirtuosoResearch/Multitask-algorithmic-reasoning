for d in 2.0 2.5 3.0
do
python train_v2.py --config configs/config.json \
    --algorithm subtraction --data_dir digit_5_borrow_True \
    --batch_size 256 --max_length 32 --num_of_instances 1000000 --generate_length 5\
    --lr 5e-4 --device 1 --num_layers 2 --random_init --tokenizer_dir gpt2_arithmetic\
    --train_size 20000 --max_steps 20000\
    --load_model_dir gpt2_addition_digit_5_carry_True_layers_2_train_size_5000 --reset_lm_head \
    --train_constraint --reg_method constraint --reg_attention $d --reg_linear $d \
    --load_target --target_algorithm addition --target_data_dir digit_5_carry_True --target_generate_length 6 --runs 2
done