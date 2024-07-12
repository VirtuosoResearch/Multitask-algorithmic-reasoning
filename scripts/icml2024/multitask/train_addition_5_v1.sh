
for size in 1000 2000 5000 10000 20000 50000 #  100000 200000 500000 1000000
do
python train_multitask.py --config configs/config.json \
    --algorithms addition --data_dirs digit_5_carry_True \
    --batch_size 256 --max_length 32 --num_of_instances 1000000 --generate_length 6\
    --lr 5e-3 --random_init --device 2 --num_layers 6 --random_init --tokenizer_dir gpt2_arithmetic\
    --train_sizes $size --eval_epoch 5 --early_stop 200 --runs 1 
done


# for size in 1000 2000 # 5000 10000 20000 50000 # 1000 100000 200000 500000 1000000
# do
# python train_multitask.py --config configs/config.json \
#     --algorithms subtraction --data_dirs digit_5_borrow_True \
#     --batch_size 256 --max_length 32 --num_of_instances 1000000 --generate_length 5\
#     --lr 5e-4 --random_init --device 3 --num_layers 2 --random_init --tokenizer_dir gpt2_arithmetic\
#     --train_sizes $size --runs 1 --only_output --early_stop 200 --runs 3
# done
