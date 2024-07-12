for l in 5 10 20
do
python notebooks/generate_subtraction.py --length $l --data_size 1000000 --allow_carry
done