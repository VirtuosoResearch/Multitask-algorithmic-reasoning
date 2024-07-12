from algorithms.utils import add_result_to_csv
from algorithms import insertion_sort, bubble_sort, selection_sort
from copy import deepcopy
import numpy as np
import os
import pandas as pd


# save results into .csv
length = 5
data_size = 1000000
file_dir = "./data/insertion_sort/"
if not os.path.exists(file_dir):
    os.makedirs(file_dir)
file_name = os.path.join(file_dir, f"inter_results_length_{length}.csv")

df = None
for _ in range(data_size):
    arr = np.random.randint(0, 100, length)
    records = [deepcopy(arr)]
    output_arr, intermediate_steps, intermediate_variables, comparisons, swaps = insertion_sort(arr)
    records += intermediate_steps
    
    for k, record in enumerate(records):
        if k == len(records) - 1:
            continue
        instance = {
            "input": " ".join([str(int(i)) for i in record]),
        }
        instance.update({
            "output": " ".join([str(int(i)) for i in records[k+1]]),
        })
        for key, val in instance.items():
            instance[key] = [val, ]
        tmp_df = pd.DataFrame(instance)
        df = pd.concat([df, tmp_df], ignore_index=True) if df is not None else tmp_df

        if len(df) == 10000:
            result_df = pd.read_csv(file_name, index_col=0)
            result_df = pd.concat([result_df, df], ignore_index = True)
            result_df.to_csv(file_name)        
            df = None

            if result_df.shape[0] > data_size:
                exit()