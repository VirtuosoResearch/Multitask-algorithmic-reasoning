for size in 20000 50000
do
python train.py --config configs/config.json \
    --algorithm division --data_dir digit_10 \
    --batch_size 256 --max_length 32 --num_of_instances 1000000 --generate_length 10\
    --lr 5e-4 --device 3 --num_layers 2 --random_init --tokenizer_dir gpt2_division\
    --train_size $size
done

for size in 100000 200000 500000 1000000
do
python train.py --config configs/config.json \
    --algorithm division --data_dir digit_3 \
    --batch_size 256 --max_length 16 --num_of_instances 1000000 --generate_length 3\
    --lr 5e-4 --device 3 --num_layers 2 --random_init --tokenizer_dir gpt2_division\
    --train_size $size
done

for size in 100000 200000 500000 1000000
do
python train.py --config configs/config.json \
    --algorithm division --data_dir digit_5 \
    --batch_size 256 --max_length 32 --num_of_instances 1000000 --generate_length 5\
    --lr 5e-4 --device 3 --num_layers 2 --random_init --tokenizer_dir gpt2_division\
    --train_size $size
done

for size in 100000 200000 500000 1000000
do
python train.py --config configs/config.json \
    --algorithm division --data_dir digit_10 \
    --batch_size 256 --max_length 32 --num_of_instances 1000000 --generate_length 10\
    --lr 5e-4 --device 3 --num_layers 2 --random_init --tokenizer_dir gpt2_division\
    --train_size $size
done
