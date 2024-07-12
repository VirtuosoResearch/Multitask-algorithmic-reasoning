for size in 1000 2000 5000 10000 20000 50000 100000 200000 500000 1000000
do
python train_v2.py --config configs/config.json \
    --algorithm addition --data_dir digit_20_carry_True \
    --batch_size 256 --max_length 64 --num_of_instances 1000000 --generate_length 21\
    --lr 5e-4 --device 2 --num_layers 2 --random_init --tokenizer_dir gpt2_arithmetic\
    --train_size $size --runs 1
done

for size in 1000 2000 5000 10000 20000 50000 100000 200000 500000 1000000
do
python train_v2.py --config configs/config.json \
    --algorithm addition --data_dir digit_40_carry_True \
    --batch_size 256 --max_length 128 --num_of_instances 1000000 --generate_length 128\
    --lr 5e-4 --device 2 --num_layers 2 --random_init --tokenizer_dir gpt2_arithmetic\
    --train_size $size --runs 1
done