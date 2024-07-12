for size in 1000 2000 5000 # 10000 # 20000 50000 100000
do
python train.py --config configs/config.json \
    --algorithm addition --data_dir digit_10_carry_False \
    --batch_size 256 --max_length 32 --num_of_instances 1000000 --generate_length 10\
    --lr 5e-4 --device 3 --num_layers 2 --random_init --tokenizer_dir gpt2_addition\
    --train_size $size
done

for size in 1000 2000 5000 
do
python train.py --config configs/config.json \
    --algorithm addition --data_dir digit_20_carry_False \
    --batch_size 256 --max_length 64 --num_of_instances 1000000 --generate_length 20\
    --lr 5e-4 --device 3 --num_layers 2 --random_init --tokenizer_dir gpt2_addition\
    --train_size $size
done
