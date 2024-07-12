python train.py --config configs/config.json \
    --algorithm sorting --data_dir length_5 \
    --batch_size 256 --max_length 16 --num_of_instances 1000000 --generate_length 5\
    --lr 5e-4 --device 1 --num_layers 2 --random_init --tokenizer_dir gpt2_sort_100\
    --train_size 20000

python train.py --config configs/config.json \
    --algorithm sorting --data_dir length_10 \
    --batch_size 256 --max_length 32 --num_of_instances 1000000 --generate_length 10\
    --lr 5e-4 --device 1 --num_layers 2 --random_init --tokenizer_dir gpt2_sort_100\
    --train_size 2000 \
    --load_model_dir gpt2_sorting_length_5_layers_2_train_size_20000

python train.py --config configs/config.json \
    --algorithm sorting --data_dir length_5 \
    --batch_size 256 --max_length 16 --num_of_instances 1000000 --generate_length 5\
    --lr 5e-4 --device 1 --num_layers 2 --random_init --tokenizer_dir gpt2_sort_100\
    --train_size 5000 --max_steps 0\
    --load_model_dir gpt2_sorting_length_10_layers_2_train_size_2000_load_model

python train.py --config configs/config.json \
    --algorithm sorting --data_dir length_10 \
    --batch_size 256 --max_length 32 --num_of_instances 1000000 --generate_length 10\
    --lr 5e-4 --device 1 --num_layers 2 --random_init --tokenizer_dir gpt2_sort_100\
    --train_size 5000 \
    --load_model_dir gpt2_sorting_length_5_layers_2_train_size_20000

python train.py --config configs/config.json \
    --algorithm sorting --data_dir length_5 \
    --batch_size 256 --max_length 16 --num_of_instances 1000000 --generate_length 5\
    --lr 5e-4 --device 1 --num_layers 2 --random_init --tokenizer_dir gpt2_sort_100\
    --train_size 5000 --max_steps 0\
    --load_model_dir gpt2_sorting_length_10_layers_2_train_size_5000_load_model

python train.py --config configs/config.json \
    --algorithm sorting --data_dir length_10 \
    --batch_size 256 --max_length 32 --num_of_instances 1000000 --generate_length 10\
    --lr 5e-4 --device 1 --num_layers 2 --random_init --tokenizer_dir gpt2_sort_100\
    --train_size 10000 \
    --load_model_dir gpt2_sorting_length_5_layers_2_train_size_20000

python train.py --config configs/config.json \
    --algorithm sorting --data_dir length_5 \
    --batch_size 256 --max_length 16 --num_of_instances 1000000 --generate_length 5\
    --lr 5e-4 --device 1 --num_layers 2 --random_init --tokenizer_dir gpt2_sort_100\
    --train_size 5000 --max_steps 0\
    --load_model_dir gpt2_sorting_length_10_layers_2_train_size_10000_load_model

python train.py --config configs/config.json \
    --algorithm sorting --data_dir length_10 \
    --batch_size 256 --max_length 32 --num_of_instances 1000000 --generate_length 10\
    --lr 5e-4 --device 1 --num_layers 2 --random_init --tokenizer_dir gpt2_sort_100\
    --train_size 20000 \
    --load_model_dir gpt2_sorting_length_5_layers_2_train_size_20000

python train.py --config configs/config.json \
    --algorithm sorting --data_dir length_5 \
    --batch_size 256 --max_length 16 --num_of_instances 1000000 --generate_length 5\
    --lr 5e-4 --device 1 --num_layers 2 --random_init --tokenizer_dir gpt2_sort_100\
    --train_size 5000 --max_steps 0\
    --load_model_dir gpt2_sorting_length_10_layers_2_train_size_20000_load_model

python train.py --config configs/config.json \
    --algorithm sorting --data_dir length_10 \
    --batch_size 256 --max_length 32 --num_of_instances 1000000 --generate_length 10\
    --lr 5e-4 --device 1 --num_layers 2 --random_init --tokenizer_dir gpt2_sort_100\
    --train_size 50000 \
    --load_model_dir gpt2_sorting_length_5_layers_2_train_size_20000

python train.py --config configs/config.json \
    --algorithm sorting --data_dir length_5 \
    --batch_size 256 --max_length 16 --num_of_instances 1000000 --generate_length 10\
    --lr 5e-4 --device 1 --num_layers 2 --random_init --tokenizer_dir gpt2_sort_100\
    --train_size 5000 --max_steps 0\
    --load_model_dir gpt2_sorting_length_10_layers_2_train_size_50000_load_model
