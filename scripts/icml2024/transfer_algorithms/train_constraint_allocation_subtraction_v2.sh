# python train_multitask.py --config configs/config.json \
#     --algorithms subtraction --data_dirs digit_5_borrow_True \
#     --batch_size 256 --max_length 32 --num_of_instances 1000000 --generate_length 5\
#     --lr 5e-3 --device 1 --num_layers 2 --random_init --tokenizer_dir gpt2_arithmetic\
#     --train_size 5000 --max_steps 20000 --eval_epoch 10 \
#     --load_model_dir gpt2_addition_digit_5_carry_True_layers_2_train_size_5000\
#     --load_target --target_algorithm addition --target_data_dir digit_5_carry_True --target_generate_length 6 --runs 2


# for d in 25 30 35 40
# do
# python train_multitask.py --config configs/config.json \
#     --algorithms subtraction --data_dirs digit_5_borrow_True \
#     --batch_size 256 --max_length 32 --num_of_instances 1000000 --generate_length 5\
#     --lr 5e-3 --device 1 --num_layers 2 --random_init --tokenizer_dir gpt2_arithmetic\
#     --train_size 5000 --max_steps 20000 --eval_epoch 10 \
#     --load_model_dir gpt2_addition_digit_5_carry_True_layers_2_train_size_5000\
#     --load_target --target_algorithm addition --target_data_dir digit_5_carry_True --target_generate_length 6 --runs 2\
#     --train_constraint_allocation --reg_total $d --early_stop 200
# done



python train_multitask.py --config configs/config.json \
    --algorithms subtraction --data_dirs digit_5_borrow_True \
    --batch_size 256 --max_length 32 --num_of_instances 1000000 --generate_length 5\
    --lr 5e-3 --device 2 --num_layers 2 --random_init --tokenizer_dir gpt2_arithmetic\
    --train_size 5000 --max_steps 20000 --eval_epoch 10 \
    --load_model_dir gpt2_addition_digit_5_carry_True_layers_2_train_size_5000\
    --load_target --target_algorithm addition --target_data_dir digit_5_carry_True --target_generate_length 6 --runs 2\
    --train_constraint_allocation --reg_total 25 --early_stop 200 --use_topk --allocation_alpha 0.001

python train_multitask.py --config configs/config.json \
    --algorithms subtraction --data_dirs digit_5_borrow_True \
    --batch_size 256 --max_length 32 --num_of_instances 1000000 --generate_length 5\
    --lr 5e-3 --device 2 --num_layers 2 --random_init --tokenizer_dir gpt2_arithmetic\
    --train_size 5000 --max_steps 20000 --eval_epoch 10 \
    --load_model_dir gpt2_addition_digit_5_carry_True_layers_2_train_size_5000\
    --load_target --target_algorithm addition --target_data_dir digit_5_carry_True --target_generate_length 6 --runs 2\
    --train_constraint_allocation --reg_total 30 --early_stop 200 --use_topk --allocation_alpha 0.001
