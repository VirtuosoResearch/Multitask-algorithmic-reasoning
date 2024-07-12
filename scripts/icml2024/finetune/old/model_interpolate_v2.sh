for alpha in 0.95 0.90 0.85 0.80 0.75 0.70 0.65 0.60 0.55 0.50 0.45 0.40 0.35 0.30 0.25 0.20 0.15 0.10 0.05
do
python train.py --config configs/config.json \
    --algorithm addition --data_dir digit_5_carry_True \
    --batch_size 256 --max_length 32 --num_of_instances 1000000 --generate_length 6\
    --lr 5e-4 --device 0 --num_layers 2 --random_init --tokenizer_dir gpt2_addition\
    --train_size 5000 --max_steps 0\
    --model_interpolate \
    --model_interpolate_start gpt2_addition_digit_5_carry_True_layers_2_train_size_10000\
    --model_interpolate_end gpt2_addition_digit_10_carry_True_layers_2_train_size_20000\
    --model_interpolate_alpha $alpha

python train.py --config configs/config.json \
    --algorithm addition --data_dir digit_10_carry_True \
    --batch_size 256 --max_length 32 --num_of_instances 1000000 --generate_length 11\
    --lr 5e-4 --device 0 --num_layers 2 --random_init --tokenizer_dir gpt2_addition\
    --train_size 5000 --max_steps 0\
    --model_interpolate \
    --model_interpolate_start gpt2_addition_digit_5_carry_True_layers_2_train_size_10000\
    --model_interpolate_end gpt2_addition_digit_10_carry_True_layers_2_train_size_20000\
    --model_interpolate_alpha $alpha
done