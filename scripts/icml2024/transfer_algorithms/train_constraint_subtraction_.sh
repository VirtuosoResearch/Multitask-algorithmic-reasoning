for d in 1.4 1.6 1.8 2.0
do
python train_multitask.py --config configs/config.json \
    --algorithms subtraction --data_dirs digit_5_borrow_True \
    --batch_size 256 --max_length 32 --num_of_instances 1000000 --generate_length 5\
    --lr 5e-3 --device 0 --num_layers 2 --random_init --tokenizer_dir gpt2_arithmetic\
    --train_size 5000 --max_steps 20000 --eval_epoch 10 \
    --load_model_dir gpt2_addition_digit_5_carry_True_layers_2_train_size_5000\
    --load_target --target_algorithm addition --target_data_dir digit_5_carry_True --target_generate_length 6 --runs 2\
    --train_constraint --reg_method constraint --reg_attention $d --reg_linear $d
done

