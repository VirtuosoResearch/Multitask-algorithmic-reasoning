from copy import deepcopy
import numpy as np
import os
import pandas as pd
import argparse
import math

def long_division(number, divisor):
    number = number.replace(" ", "")
    intermediate_steps = []
    # As result can be very large 
    # store it in string 
    ans = "" 
     
    # Find prefix of number that 
    # is larger than divisor. 
    idx = 0 
    temp = int(number[idx])
    while (temp < divisor):
        temp = (temp * 10 + int(number[idx+1]))
        idx += 1
     
    idx += 1
 
    # Repeatedly divide divisor with temp. 
    # After every division, update temp to 
    # include one more digit. 
    while ((len(number)) > idx): 
         
        # Store result in answer i.e. temp / divisor 
        ans += str(math.floor(temp // divisor)) # + ord('0'))
        intermediate_steps.append(list(deepcopy([int(i) for i in ans])))
         
        # Take next digit of number
        temp = (temp % divisor) * 10 + int(number[idx]) # - ord('0'))
        idx += 1
 
    ans += str(math.floor(temp // divisor)) # + ord('0'))
    intermediate_steps.append(list(deepcopy([int(i) for i in ans])))
        
    # If divisor is greater than number 
    if (len(ans) == 0): 
        return "0"; 
     
    ans = " ".join(list(ans))
    return ans, intermediate_steps

# save results into .csv
def main(args):
    length = args.length
    data_size = args.data_size
    file_dir = "./data/division/"
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    file_name = os.path.join(file_dir, f"digit_{length}.csv")

    df = None
    for _ in range(data_size):
        # sample digits
        digits1 = np.random.randint(0, 10, length)
        while digits1[0] == 0:
            digits1 = np.random.randint(0, 10, length)
        num1 = int("".join([str(i) for i in digits1]))
        digits1 = " ".join(list(str(num1)))

        num2 = np.random.randint(1, 10)
        digits2 = " ".join(list(str(num2)))

        if num1 < num2:
            num2, num1 = num1, num2
            digits1, digits2 = digits2, digits1

        # add
        # output_digits = num1 // num2
        # output_digits = " ".join(list(str(output_digits)))
        output_digits, intermediate_steps = long_division(digits1, num2)
        
        # save
        input = digits1 + " / " + digits2
        instance = {
            "input": input,
        }
        output = output_digits
        instance.update({
            "output": output,
        })
        records = intermediate_steps
        for k, record in enumerate(records):
            if k == len(records) - 1: continue # skip the last one
            instance.update({
                f"step_{k}": " ".join([str(i) for i in record]), # count it backwards
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
    parser.add_argument("--data_size", type=int, default=1000000)
    args = parser.parse_args()
    
    main(args)