# python train_multitask.py --config configs/config.json \
#     --algorithms multiplication --data_dirs digit_10_carry_True \
#     --batch_size 256 --max_length 64 --num_of_instances 1000000 --generate_length 11\
#     --lr 5e-3 --device 3 --num_layers 2 --random_init --tokenizer_dir gpt2_arithmetic\
#     --train_size 2000 --eval_epoch 20 \
#     --load_model_dir gpt2_multiplication_digit_5_carry_True_layers_2_train_size_10000\
#     --load_target --target_algorithm multiplication --target_data_dir digit_5_carry_True --target_generate_length 6 --runs 1

# python train_multitask.py --config configs/config.json \
#     --algorithms multiplication --data_dirs digit_10_carry_True \
#     --batch_size 256 --max_length 64 --num_of_instances 1000000 --generate_length 11\
#     --lr 5e-3 --device 3 --num_layers 2 --random_init --tokenizer_dir gpt2_arithmetic\
#     --train_size 5000 --eval_epoch 10 \
#     --load_model_dir gpt2_multiplication_digit_5_carry_True_layers_2_train_size_10000\
#     --load_target --target_algorithm multiplication --target_data_dir digit_5_carry_True --target_generate_length 6 --runs 1

# python train_multitask.py --config configs/config.json \
#     --algorithms multiplication --data_dirs digit_10_carry_True \
#     --batch_size 256 --max_length 64 --num_of_instances 1000000 --generate_length 11\
#     --lr 5e-3 --device 3 --num_layers 2 --random_init --tokenizer_dir gpt2_arithmetic\
#     --train_size 10000 --eval_epoch 5 \
#     --load_model_dir gpt2_multiplication_digit_5_carry_True_layers_2_train_size_10000\
#     --load_target --target_algorithm multiplication --target_data_dir digit_5_carry_True --target_generate_length 6 --runs 1

python train_multitask.py --config configs/config.json \
    --algorithms multiplication --data_dirs digit_10_carry_True \
    --batch_size 256 --max_length 64 --num_of_instances 1000000 --generate_length 11\
    --lr 5e-3 --device 2 --num_layers 2 --random_init --tokenizer_dir gpt2_arithmetic\
    --train_size 20000 --max_steps 20000\
    --load_model_dir gpt2_multiplication_digit_5_carry_True_layers_2_train_size_10000\
    --load_target --target_algorithm multiplication --target_data_dir digit_5_carry_True --target_generate_length 6 --runs 1