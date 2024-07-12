for alpha in 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 
do
python train_v2.py --config configs/config.json \
    --algorithm subtraction --data_dir digit_5_borrow_True \
    --batch_size 256 --max_length 32 --num_of_instances 1000000 --generate_length 5\
    --lr 5e-4 --device 0 --num_layers 2 --random_init --tokenizer_dir gpt2_arithmetic\
    --train_size 5000 --max_steps 0\
    --model_interpolate \
    --model_interpolate_start gpt2_addition_digit_5_carry_True_layers_2_train_size_5000\
    --model_interpolate_end gpt2_subtraction_digit_5_borrow_True_layers_2_train_size_5000_load_model\
    --model_interpolate_alpha $alpha\
    --load_target --target_algorithm addition --target_data_dir digit_5_carry_True --target_generate_length 6 --runs 1
done