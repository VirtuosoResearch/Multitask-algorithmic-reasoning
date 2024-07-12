python train.py --config configs/config.json \
    --algorithm addition --data_dir digit_5_carry_True \
    --batch_size 256 --max_length 32 --num_of_instances 1000000 --generate_length 6\
    --lr 5e-4 --device 1 --num_layers 2 --random_init --tokenizer_dir gpt2_addition\
    --train_size 5000 \
    --load_model_dir gpt2_addition_digit_3_carry_True_layers_2_train_size_5000

python train.py --config configs/config.json \
    --algorithm addition --data_dir digit_3_carry_True \
    --batch_size 256 --max_length 16 --num_of_instances 1000000 --generate_length 4\
    --lr 5e-4 --device 1 --num_layers 2 --random_init --tokenizer_dir gpt2_addition\
    --train_size 5000 --max_steps 0\
    --load_model_dir gpt2_addition_digit_5_carry_True_layers_2_train_size_5000_load_model
