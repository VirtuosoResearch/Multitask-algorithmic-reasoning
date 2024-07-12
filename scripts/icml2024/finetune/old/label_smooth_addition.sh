for alpha in 0.1 0.2 0.3 0.4
do
python train.py --config configs/config.json \
    --algorithm addition --data_dir digit_10_carry_True \
    --batch_size 256 --max_length 32 --num_of_instances 1000000 --generate_length 11\
    --lr 5e-4 --device 3 --num_layers 2 --random_init --tokenizer_dir gpt2_addition\
    --train_size 20000 --max_steps 20000\
    --load_model_dir gpt2_addition_digit_5_carry_True_layers_2_train_size_5000 \
    --load_target --target_algorithm addition --target_data_dir digit_5_carry_True --target_generate_length 6 --runs 2\
    --train_ls --ls_alpha $alpha
done
