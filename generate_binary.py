import numpy as np
import pandas as pd
import os
import argparse

def generate_sorted_array(size, max_val=100):
    array = np.random.randint(0, max_val, size)
    array.sort()
    return array

def binary_search(arr, target):
    low, high = 0, len(arr) - 1
    steps = 0
    while low <= high:
        steps += 1
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid, steps
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1, steps

def main(args):
    data_size = args.data_size
    array_size = args.array_size
    file_dir = "./data/binary/"
    
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    file_name = os.path.join(file_dir, f"binary_search_data_{array_size}.csv")
    
    df = None
    for _ in range(data_size):
        array = generate_sorted_array(array_size)
        target = np.random.choice(array)
        index, steps = binary_search(array, target)
        
        input_data = "array: " + " ".join(map(str, array)) + " target: " + str(target)
        output_data = "index: "+str(index)+ " steps: "+str(steps)
        
        instance = {
            "node": array_size,
            "input": input_data,
            "output": output_data
        }
        
        for key, val in instance.items():
            instance[key] = [val, ]
        tmp_df = pd.DataFrame(instance)
        df = pd.concat([df, tmp_df], ignore_index=True) if df is not None else tmp_df

        if len(df) >= 1000:
            if not os.path.exists(file_name):
                df.to_csv(file_name, index=False)
            else:
                result_df = pd.read_csv(file_name)
                result_df = pd.concat([result_df, df], ignore_index=True)
                result_df.to_csv(file_name, index=False)
                if result_df.shape[0] >= data_size:
                    exit()
            df = None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_size", type=int, default=100000)
    parser.add_argument("--array_size", type=int, default=10)
    args = parser.parse_args()
    
    main(args)
