for d in 18 15 12
do
python train_multitask.py --config configs/config.json \
    --algorithms division --data_dirs digit_10 \
    --batch_size 256 --max_length 64 --num_of_instances 1000000 --generate_length 10\
    --lr 5e-3 --device 2 --num_layers 2 --random_init --tokenizer_dir gpt2_arithmetic\
    --train_size 20000 --eval_epoch 10 --max_steps 20000 \
    --load_model_dir gpt2_division_digit_5_layers_2_train_size_10000\
    --load_target --target_algorithm division --target_data_dir digit_5 --target_generate_length 5 --runs 2\
    --train_constraint_allocation --reg_total $d --early_stop 200 --allocation_strategy topk --allocation_alpha 0.01
done

for d in 20 25 30
do
python train_multitask.py --config configs/config.json \
    --algorithms division --data_dirs digit_10 \
    --batch_size 256 --max_length 64 --num_of_instances 1000000 --generate_length 10\
    --lr 5e-3 --device 2 --num_layers 2 --random_init --tokenizer_dir gpt2_arithmetic\
    --train_size 20000 --eval_epoch 10 --max_steps 20000 \
    --load_model_dir gpt2_division_digit_5_layers_2_train_size_10000\
    --load_target --target_algorithm division --target_data_dir digit_5 --target_generate_length 5 --runs 2\
    --train_constraint_allocation --reg_total $d --early_stop 200 --allocation_strategy old_topk --allocation_alpha 0.01
done
