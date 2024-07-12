# for size in 2000 5000 10000 20000 50000 100000 200000 500000 1000000
# do
# python train.py --config configs/config.json \
#     --algorithm multiplication --data_dir digit_3_carry_True \
#     --batch_size 256 --max_length 16 --num_of_instances 1000000 --generate_length 6\
#     --lr 5e-4 --device 0 --num_layers 2 --random_init --tokenizer_dir gpt2_multiplication\
#     --train_size $size
# done

# for size in 2000 5000 10000 20000 50000 100000 200000 500000 1000000
# do
# python train.py --config configs/config.json \
#     --algorithm multiplication --data_dir digit_5_carry_True \
#     --batch_size 256 --max_length 32 --num_of_instances 1000000 --generate_length 10\
#     --lr 5e-4 --device 0 --num_layers 2 --random_init --tokenizer_dir gpt2_multiplication\
#     --train_size $size
# done

python train.py --config configs/config.json \
    --algorithm multiplication --data_dir digit_3_carry_True \
    --batch_size 256 --max_length 16 --num_of_instances 1000000 --generate_length 6\
    --lr 5e-4 --device 2 --num_layers 2 --random_init --tokenizer_dir gpt2_multiplication\
    --train_size 1000000 --max_steps 1000000 

for l in 4 8 12
do
python train.py --config configs/config.json \
    --algorithm multiplication --data_dir digit_3_carry_True \
    --batch_size 256 --max_length 16 --num_of_instances 1000000 --generate_length 6\
    --lr 5e-4 --device 2 --num_layers $l --random_init --tokenizer_dir gpt2_multiplication\
    --train_size 1000000
done