# for size in 1000 2000 5000 10000
# do
# python train_multitask.py --config configs/config.json \
#     --algorithms addition --data_dirs digit_10_carry_True \
#     --batch_size 256 --max_length 64 --num_of_instances 1000000 --generate_length 11\
#     --lr 5e-3 --device 2 --num_layers 2 --random_init --tokenizer_dir gpt2_arithmetic\
#     --train_size $size \
#     --load_model_dir gpt2_addition_digit_5_carry_True_layers_2_train_size_5000\
#     --load_target --target_algorithm addition --target_data_dir digit_5_carry_True --target_generate_length 5 --runs 1
# done

# python train_multitask.py --config configs/config.json \
#     --algorithms addition --data_dirs digit_10_carry_True \
#     --batch_size 256 --max_length 64 --num_of_instances 1000000 --generate_length 11\
#     --lr 5e-3 --device 2 --num_layers 2 --random_init --tokenizer_dir gpt2_arithmetic\
#     --train_size 1000 --eval_epoch 50 \
#     --load_model_dir gpt2_addition_digit_5_carry_True_layers_2_train_size_5000\
#     --load_target --target_algorithm addition --target_data_dir digit_5_carry_True --target_generate_length 5 --runs 1

# python train_multitask.py --config configs/config.json \
#     --algorithms addition --data_dirs digit_10_carry_True \
#     --batch_size 256 --max_length 64 --num_of_instances 1000000 --generate_length 11\
#     --lr 5e-3 --device 2 --num_layers 2 --random_init --tokenizer_dir gpt2_arithmetic\
#     --train_size 2000 --eval_epoch 20 \
#     --load_model_dir gpt2_addition_digit_5_carry_True_layers_2_train_size_5000\
#     --load_target --target_algorithm addition --target_data_dir digit_5_carry_True --target_generate_length 5 --runs 1

python train_multitask.py --config configs/config.json \
    --algorithms addition --data_dirs digit_10_carry_True \
    --batch_size 256 --max_length 64 --num_of_instances 1000000 --generate_length 11\
    --lr 5e-3 --device 1 --num_layers 2 --random_init --tokenizer_dir gpt2_arithmetic\
    --train_size 5000 --eval_epoch 10 \
    --load_model_dir gpt2_addition_digit_5_carry_True_layers_2_train_size_5000\
    --load_target --target_algorithm addition --target_data_dir digit_5_carry_True --target_generate_length 5 --runs 1

# python train_multitask.py --config configs/config.json \
#     --algorithms addition --data_dirs digit_10_carry_True \
#     --batch_size 256 --max_length 64 --num_of_instances 1000000 --generate_length 11\
#     --lr 5e-3 --device 2 --num_layers 2 --random_init --tokenizer_dir gpt2_arithmetic\
#     --train_size 10000 --eval_epoch 1 \
#     --load_model_dir gpt2_addition_digit_5_carry_True_layers_2_train_size_5000\
#     --load_target --target_algorithm addition --target_data_dir digit_5_carry_True --target_generate_length 5 --runs 1