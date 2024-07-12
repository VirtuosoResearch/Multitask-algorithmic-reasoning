python train.py --config configs/config.json \
    --algorithm addition --data_dir digit_3_carry_True \
    --batch_size 256 --max_length 16 --num_of_instances 1000000 --generate_length 4\
    --lr 5e-4 --device 0 --num_layers 2 --random_init --tokenizer_dir gpt2_addition\
    --train_size 2000 --max_steps 100000\
    --load_model_dir gpt2_addition_digit_3_carry_False_layers_2_train_size_1000

python train.py --config configs/config.json \
    --algorithm addition --data_dir digit_3_carry_False \
    --batch_size 256 --max_length 16 --num_of_instances 1000000 --generate_length 3\
    --lr 5e-4 --device 0 --num_layers 2 --random_init --tokenizer_dir gpt2_addition\
    --train_size 1000 --max_steps 0\
    --load_model_dir gpt2_addition_digit_3_carry_True_layers_2_train_size_2000_load_model