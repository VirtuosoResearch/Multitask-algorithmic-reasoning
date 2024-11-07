from copy import deepcopy
import numpy as np
import os
import pandas as pd
import argparse


def substraction(digits1, digits2):
    intermediate_steps = []

    digits1 = digits1[::-1]
    digits2 = digits2[::-1]
    digits = digits1 - digits2
    borrow = False
    for i in range(len(digits)):
        tmp_borrow = False
        if digits[i] < 0:
            assert i != len(digits) - 1
            digits[i+1] -= 1
            digits[i] += 10
            borrow = True
            tmp_borrow = True
        intermediate_steps.append(
            [1 if tmp_borrow else 0] + list(deepcopy(digits[:i+1][::-1]))
        )
    digits1 = digits1[::-1]
    digits2 = digits2[::-1]
    return digits[::-1], borrow, intermediate_steps

# save results into .csv
def main(args):
    length = args.length
    allow_carry = args.allow_carry
    data_size = args.data_size
    file_dir = "./data/subtraction/"
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    file_name = os.path.join(file_dir, f"digit_{length}_borrow_{allow_carry}.csv")

    df = None
    for _ in range(data_size):
        # sample digits
        digits1 = np.random.randint(0, 10, length)
        while digits1[0] == 0:
            digits1 = np.random.randint(0, 10, length)

        digits2 = np.random.randint(0, 10, length)
        while digits2[0] == 0:
            digits2 = np.random.randint(0, 10, length)

        if int("".join([str(int(i)) for i in digits1])) < int("".join([str(int(i)) for i in digits2])):
            digits1, digits2 = digits2, digits1


        # add
        output_digits, carry, intermediate_steps = substraction(digits1, digits2)
        if (not allow_carry ) and carry:
            continue

        # save
        input = " ".join([str(int(i)) for i in digits1]) + " - " + " ".join([str(int(i)) for i in digits2])
        instance = {
            "input": input,
        }
        output = " ".join([str(int(i)) for i in output_digits])
        instance.update({
            "output": output,
        })
        records = intermediate_steps
        for k, record in enumerate(records):
            if k == len(records) - 1: continue # skip the last one
            instance.update({
                f"step_{k}": " ".join([str(int(i)) for i in record]), # count it backwards
            })
        
        for key, val in instance.items():
            instance[key] = [val, ]
        tmp_df = pd.DataFrame(instance)
        df = pd.concat([df, tmp_df], ignore_index=True) if df is not None else tmp_df

        if len(df) == 1000:
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
    
    parser.add_argument("--length", type=int, default=3)
    parser.add_argument("--allow_carry", action="store_true")
    parser.add_argument("--data_size", type=int, default=1000000)
    args = parser.parse_args()
    
    main(args)