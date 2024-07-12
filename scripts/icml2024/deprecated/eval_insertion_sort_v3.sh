for k in 0 1 2 3 4 5 6 7 8 9 10
do
python eval.py --model_name gpt2 --algorithm insertion_sort --data_dir length_2_test \
    --num_examples 100 --num_train $k --length 2 --checkpoint_dir saved/gpt2_insertion_sort_length_2_7 --device 2
done

for k in 0 1 2 3 4 5 6 7 8 9 10
do
python eval.py --model_name gpt2 --algorithm insertion_sort --data_dir length_2_test \
    --num_examples 100 --num_train $k --length 2 --checkpoint_dir saved/gpt2_insertion_sort_length_2_8 --device 2
done