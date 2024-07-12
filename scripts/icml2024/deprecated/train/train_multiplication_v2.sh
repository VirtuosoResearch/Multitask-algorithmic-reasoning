for size in 2000 5000 10000 20000 50000 100000 200000 500000 1000000
do
python train.py --config configs/config.json \
    --algorithm multiplication --data_dir digit_10_carry_True \
    --batch_size 256 --max_length 64 --num_of_instances 1000000 --generate_length 20\
    --lr 5e-4 --device 1 --num_layers 2 --random_init --tokenizer_dir gpt2_multiplication\
    --train_size $size
done

for size in 2000 5000 10000 20000 50000 100000 200000 500000 1000000
do
python train.py --config configs/config.json \
    --algorithm multiplication --data_dir digit_20_carry_True \
    --batch_size 256 --max_length 128 --num_of_instances 1000000 --generate_length 40\
    --lr 5e-4 --device 1 --num_layers 2 --random_init --tokenizer_dir gpt2_multiplication\
    --train_size $size
done