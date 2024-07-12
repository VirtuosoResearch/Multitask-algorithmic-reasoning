for size in 5000 # 20000 50000 100000
do
python train_v2.py --config configs/config.json \
    --algorithm addition --data_dir digit_5_carry_True \
    --batch_size 256 --max_length 32 --num_of_instances 1000000 --generate_length 6\
    --lr 5e-4 --device 1 --num_layers 2 --random_init --tokenizer_dir gpt2_arithmetic\
    --train_size $size --runs 1

python train_v2.py --config configs/config.json \
    --algorithm subtraction --data_dir digit_5_borrow_True \
    --batch_size 256 --max_length 32 --num_of_instances 1000000 --generate_length 5\
    --lr 5e-4 --device 1 --num_layers 2 --random_init --tokenizer_dir gpt2_arithmetic\
    --train_size $size --runs 1
done

# --load_model_dir gpt2_addition_digit_3_carry_False_layers_2_lr_0.0005
