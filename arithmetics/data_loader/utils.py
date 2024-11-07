import numpy as np
import pandas as pd

def process_intermediate_steps(intermediate_steps, is_array=True, is_graph=False):
    '''
    Break intermediate steps into a list
    '''
    if is_array:
        intermediate_steps =  intermediate_steps.strip("[]").split("], [")
        intermediate_steps = [f"Array: [" + step + "]" for i, step in enumerate(intermediate_steps)]
        return intermediate_steps
    elif is_graph:
        # TODO: process graph
        pass

def process_intermediate_variables(intermediate_variables):
    intermediate_variables = eval(intermediate_variables)
    intermediate_variables = ["Compare index {} and {}, Swap: {},".format(step["i"], step["j"], step["swap"]) for i, step in enumerate(intermediate_variables)]
    return intermediate_variables

def format_algorithm_process(df, idx):
    process_prompt = ""
    intermediate_variables = df["intermediate_variables"].iloc[idx]
    intermediate_variables = process_intermediate_variables(intermediate_variables)

    intermediate_steps = df["intermediate_steps"].iloc[idx]
    intermediate_steps = process_intermediate_steps(intermediate_steps, is_array=True, is_graph=False)

    for i, step in enumerate(intermediate_steps):
        intermediate_steps[i] = str(intermediate_variables[i]) + " " + step 
    
    for i, step in enumerate(intermediate_steps):
        process_prompt += f"\n" + step
    return process_prompt

def format_algorithm_example(df, idx, 
        include_input=True,
        include_answer=True, 
        include_inter_results=True,
        init_prompt="Sort the array with insertion sort"):
    prompt = {}
    if include_input:
        prompt['input'] = init_prompt + df["input"].iloc[idx]

    if include_inter_results:
        process_prompt = format_algorithm_process(df, idx)
        prompt["inter_results"] = process_prompt

    if include_answer:
        output = "\nAnswer:" 
        output += "{}\n\n".format(df["output"].iloc[idx])
        prompt["output"] = output
    return prompt

def generate_algorithm_prompt(df, idx, k=0,
        include_input=True,
        include_answer=True, 
        include_inter_results=True,
        algorithm_name="insertion sort", 
        init_prompt="Sort the array with insertion sort",
        input_prompt = f"The following are examples of excuting insertion sort algorithm.\n\n"):

    if k == 0:
        return format_algorithm_example(df, idx, init_prompt=init_prompt, include_input=include_input, include_inter_results=include_inter_results, include_answer=include_answer)
    
    indexes = np.random.choice(df.shape[0], size=k, replace=False)
    for i in indexes:
        tmp_prompt = format_algorithm_example(df, i, init_prompt=init_prompt, include_input=True, include_inter_results=True, include_answer=True)
        input_prompt += tmp_prompt["input"] + tmp_prompt["inter_results"] + tmp_prompt["output"]
    
    prompt = {}
    if include_input:
        prompt['input'] = input_prompt + init_prompt + df["input"].iloc[idx]

    if include_inter_results:
        process_prompt = format_algorithm_process(df, idx)
        prompt["inter_results"] = process_prompt

    if include_answer:
        output = "\nAnswer:" 
        output += "{}\n\n".format(df["output"].iloc[idx])
        prompt["output"] = output
    return prompt

'''
Simple input-output format
'''
def format_simple_example(df, idx, add_newline=True, input_columns=["input"], output_columns=["output"]):
    prompt = {}    
    
    # print("***********************check1")
    # print("df")
    # print(df)
    # print("***********************check2")
    # print(df["input"])
    # print("***********************check3")
    prompt['input'] = df["input"].iloc[idx] + "\n"
    prompt["output"] = ""
    
    for column in input_columns: # deprecated for training from scratch
        if "index_i" in column:
            prompt["index_i"] = df[column].iloc[idx]
        elif "index_j" in column:
            prompt["index_j"] = df[column].iloc[idx]
        if "step" in column and (not pd.isna(df[column].iloc[idx])):
            prompt["input"] += (str(df[column].iloc[idx]) + "\n")
    for i, column in enumerate(output_columns):
        if pd.isna(df[column].iloc[idx]): 
            prompt[f"column_{column}"] = ""
        else:
            # add output for each column
            prompt[f"column_{column}"] = str(df[column].iloc[idx])
        prompt["output"] += (str(df[column].iloc[idx]) + "\n") if i < len(output_columns) - 1 else (str(df[column].iloc[idx]))
    prompt["output"] += ("\n\n" if add_newline else "")
    return prompt

def generate_simple_algorithm_example(df, idx, k=0, input_columns=["input"], output_columns=["output"]):

    if k == 0:
        return format_simple_example(df, idx, add_newline=False, input_columns=input_columns, output_columns=output_columns)
    rng = np.random.default_rng(42)
    indexes = rng.choice(df.shape[0], size=k, replace=False)
    input_prompt = ""
    
    
    for i in indexes:
        tmp_prompt = format_simple_example(df, i, input_columns=input_columns, output_columns=output_columns)
        input_prompt = tmp_prompt["input"] + tmp_prompt["output"]
    
    prompt = {}
    prompt['input'] = input_prompt + df["input"].iloc[idx] + "\n"
    prompt["output"] = str(df["output"].iloc[idx])
    return prompt