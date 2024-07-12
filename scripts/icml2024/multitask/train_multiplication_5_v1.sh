
for size in 1000 2000 5000 10000 20000 50000 #  100000 200000 500000 1000000
do
python train_multitask.py --config configs/config.json \
    --algorithms multiplication --data_dirs digit_5_carry_True \
    --batch_size 256 --max_length 16 --num_of_instances 1000000 --generate_length 6\
    --lr 5e-4 --random_init --device 0 --num_layers 2 --random_init --tokenizer_dir gpt2_arithmetic\
    --train_sizes $size --runs 1
done

