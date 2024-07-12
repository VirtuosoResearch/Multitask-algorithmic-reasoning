for size in 2000 5000 # 10000 20000 50000 100000 200000 500000 1000000
do
python train.py --config configs/config.json \
    --algorithm bubble_sort --data_dir inter_results_length_5\
    --batch_size 256 --max_length 16 --num_of_instances 1000000 \
    --lr 5e-4 --random_init --device 1 --num_layers 2 --tokenizer_dir gpt2_sort_100\
    --train_size $size

python train.py --config configs/config.json \
    --algorithm selection_sort --data_dir inter_results_length_5 \
    --batch_size 256 --max_length 16 --num_of_instances 1000000 \
    --lr 5e-4 --random_init --device 1 --num_layers 2 --tokenizer_dir gpt2_sort_100\
     --train_size $size

python train.py --config configs/config.json \
    --algorithm quick_sort --data_dir inter_results_length_5\
    --batch_size 256 --max_length 16 --num_of_instances 1000000 \
    --lr 5e-4 --random_init --device 1 --num_layers 2 --tokenizer_dir gpt2_sort_100\
    --train_size $size
done


python train.py --config configs/config.json \
    --algorithm bubble_sort --data_dir inter_results_length_5 \
    --batch_size 256 --max_length 16 --num_of_instances 1000000 --generate_length 5\
    --lr 5e-4 --random_init --device 0 --num_layers 2 --tokenizer_dir gpt2_sort_100\
    --train_size 1000 --max_steps 20 --runs 1