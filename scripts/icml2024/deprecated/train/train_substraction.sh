for size in 1000 2000 5000 10000 20000 50000 100000
do
python train.py --config configs/config.json \
    --algorithm subtraction --data_dir digit_3_borrow_True \
    --batch_size 256 --max_length 16 --num_of_instances 1000000 --generate_length 3\
    --lr 5e-4 --device 1 --num_layers 2 --random_init --tokenizer_dir gpt2_subtraction\
    --train_size $size

python train.py --config configs/config.json \
    --algorithm subtraction --data_dir digit_5_borrow_True \
    --batch_size 256 --max_length 32 --num_of_instances 1000000 --generate_length 5\
    --lr 5e-4 --device 1 --num_layers 2 --random_init --tokenizer_dir gpt2_subtraction\
    --train_size $size

python train.py --config configs/config.json \
    --algorithm subtraction --data_dir digit_10_borrow_True \
    --batch_size 256 --max_length 32 --num_of_instances 1000000 --generate_length 10\
    --lr 5e-4 --device 1 --num_layers 2 --random_init --tokenizer_dir gpt2_subtraction\
    --train_size $size

python train.py --config configs/config.json \
    --algorithm subtraction --data_dir digit_20_borrow_True \
    --batch_size 256 --max_length 64 --num_of_instances 1000000 --generate_length 20\
    --lr 5e-4 --device 1 --num_layers 2 --random_init --tokenizer_dir gpt2_subtraction\
    --train_size $size
done