for k in 1 2 3 4 5 6 7 8 9 10
do
python train.py --config configs/config.json \
    --algorithm insertion_sort --data_dir length_2\
    --lr 1e-5 --incontext_k $k
done