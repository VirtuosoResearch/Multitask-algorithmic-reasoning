for d in 23 20 18 15
do
python train_multitask.py --config configs/config.json \
    --algorithms division --data_dirs digit_5 \
    --batch_size 256 --max_length 32 --num_of_instances 1000000 --generate_length 5\
    --lr 5e-3 --device 2 --num_layers 2 --random_init --tokenizer_dir gpt2_arithmetic\
    --train_size 10000 --eval_epoch 5 --max_steps 20000\
    --load_model_dir gpt2_multiplication_digit_5_carry_True_layers_2_train_size_10000\
    --load_target --target_algorithm multiplication --target_data_dir digit_5_carry_True --target_generate_length 6 --runs 2\
    --train_constraint_allocation --reg_total $d --early_stop 200 --allocation_strategy old_topk
done