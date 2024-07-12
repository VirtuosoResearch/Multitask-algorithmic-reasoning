for size in 1000 2000 5000 10000 20000 50000
do
python train.py --config configs/config.json \
    --algorithm insertion_sort --data_dir inter_results_length_10 \
    --batch_size 256 --max_length 32 --num_of_instances 1000000 --generate_length 10\
    --lr 5e-4 --random_init --device 2 --num_layers 2 --tokenizer_dir gpt2_sort_100\
    --train_size $size --max_steps 20000 --runs 1
done

for size in 100000 200000 500000 1000000
do
python train.py --config configs/config.json \
    --algorithm insertion_sort --data_dir inter_results_length_10 \
    --batch_size 256 --max_length 32 --num_of_instances 1000000 --generate_length 10\
    --lr 5e-4 --random_init --device 2 --num_layers 2 --tokenizer_dir gpt2_sort_100\
    --train_size $size --max_steps 100000 --runs 1
done