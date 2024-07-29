from copy import deepcopy
import numpy as np
import os
import pandas as pd
import argparse

def quick_select(array, k):
    length = len(array)
    intermediate_steps = []
    
    def partition(array, left, right):
        pivot = array[right]
        i = left
        for j in range(left, right):
            if array[j] <= pivot:
                array[i], array[j] = array[j], array[i]

                if i != j: intermediate_steps.append(list(deepcopy(array)))
                i += 1

        array[i], array[right] = array[right], array[i]

        if i != right: intermediate_steps.append(list(deepcopy(array)))
        return i

    def select(array, left, right, k):
        if k == 0:
            return array[left]
        assert (k > 0 and k <= right - left + 1) 

        pivot_index = partition(array, left, right)
        if (k - 1) == (pivot_index - left):
            return array[pivot_index]
        elif (k-1) < (pivot_index-left):
            return select(array, left, pivot_index - 1, k)
        else:
            return select(array, pivot_index + 1, right, k - pivot_index + left - 1)

    return select(array, 0, length - 1, k), intermediate_steps

def main(args):
    # save results into .csv
    length = args.length
    data_size = args.data_size
    file_dir = f"./data/{args.save_dir}/"
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    file_name = os.path.join(file_dir, f"length_{length}.csv") if not args.save_inter_results else \
                os.path.join(file_dir, f"inter_results_length_{length}.csv")

    df = None
    for _ in range(data_size):
        arr = np.random.randint(0, 100, length)
        k = np.random.randint(1, length+1)
        records = [deepcopy(arr)]
        output, intermediate_steps = \
            quick_select(arr, k)

        if args.save_inter_results:
            # save intermediate steps
            records += intermediate_steps
            for k, record in enumerate(records):
                if k == len(records) - 1:
                    continue
                instance = {
                    "input": f"{k}" + ", " + " ".join([str(int(i)) for i in record]),
                }
                instance.update({
                    "output": " ".join([str(int(i)) for i in records[k+1]]),
                })
                for key, val in instance.items():
                    instance[key] = [val, ]
                tmp_df = pd.DataFrame(instance)
                df = pd.concat([df, tmp_df], ignore_index=True) if df is not None else tmp_df

                if len(df) == 10000:
                    if not os.path.exists(file_name):
                        df.to_csv(file_name)
                    else:
                        result_df = pd.read_csv(file_name, index_col=0)
                        result_df = pd.concat([result_df, df], ignore_index = True)
                        result_df.to_csv(file_name)        
                        
                        if result_df.shape[0] > data_size:
                            exit()
                    df = None
        else:
            instance = {
                "input": f"{k}" + ", " + " ".join([str(int(i)) for i in records[0]]),
            }
            instance.update({
                "output": f"{output}",
            })
            for key, val in instance.items():
                instance[key] = [val, ]
            tmp_df = pd.DataFrame(instance)
            df = pd.concat([df, tmp_df], ignore_index=True) if df is not None else tmp_df

            if len(df) == 10000:
                if not os.path.exists(file_name):
                    df.to_csv(file_name)
                else:
                    result_df = pd.read_csv(file_name, index_col=0)
                    result_df = pd.concat([result_df, df], ignore_index = True)
                    result_df.to_csv(file_name)        
                    
                    if result_df.shape[0] > data_size:
                        exit()
                df = None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--length", type=int, default=5)
    parser.add_argument("--data_size", type=int, default=1000000)
    parser.add_argument("--save_dir", type=str, default="quick_select")
    parser.add_argument("--save_inter_results", action="store_true")
    args = parser.parse_args()
    
    main(args)