# %%
import clrs

# %%
# 'insertion_sort'
# 'bubble_sort'
# 'heapsort'
# 'quicksort'
train_ds, num_samples, spec = clrs.create_dataset(
      folder='./data/CLRS', algorithm='bubble_sort',
      split='train', batch_size=32)

# %%
for i, feedback in enumerate(train_ds.as_numpy_iterator()):
    print(feedback)
    break

# %%
features = feedback.features
outputs = feedback.outputs
inputs = features.inputs # inputs "key" and "pos"
hints = features.hints
lengths = features.lengths

# Hints in insertion sort
#       i: correct place for the  j-th element at the start of each iteration
#       j: the index of the current element
#       pred_h: predecessor pointers

# Hints in bubble sort
#       i: the index of the outer loop
#       j: the index of the inner loop, the element to be compared
#       pred_h: predecessor pointers 

# Hints in quick sort
#       p: the index of pivot
#       i: the index for the next element smaller than pivot
#       j: the index that scans the array
#       pred_h: predecessor pointers
#       r: the rightmost index of the current sort array

# %%
# best case: the array is already sorted
# worst case: the array is sorted in reverse order
#       insertion sort is one of the fastest algorithms for sorting very small arrays, 
#       even faster than quicksort;

# 5-->3-->4-->7-->8
# labels for j: 0 1 0 0 0 
# labels for i: 1 0 0 0 0

# j = 1; i = 0; 3 5 4 7 8 
# j = 2; i = 1; 3 4 5 7 8

def insertion_sort(array):
    comparisons = 0
    swaps = 0
    intermediate_steps = []
    for j in range(1, len(array)):
        key = array[j]
        i = j - 1
        while i >= 0 and array[i] > key:
            array[i + 1] = array[i]
            i = i - 1
            comparisons += 1
            swaps += 1
        array[i + 1] = key
        intermediate_steps.append(list(array.copy()))

        if i!=-1: comparisons += 1
        if i+1!=j: swaps += 1
    return array, intermediate_steps, comparisons, swaps


def bubble_sort(array):
    comparisons = 0
    swaps = 0
    length = len(array)
    intermediate_steps = []
    for _ in range(length):
        swapped = False
        for i in range(1, length):
            comparisons += 1
            if array[i - 1] > array[i]:
                array[i - 1], array[i] = array[i], array[i - 1]
                swapped = True
                swaps += 1
        intermediate_steps.append(list(array.copy()))
        if not swapped:
            break
    return array, intermediate_steps, comparisons, swaps

def quick_sort_(array, lo, hi, comparisons, swaps):
    if lo>=hi or lo<0:
        return array, comparisons, swaps
    pivot = array[hi-1]

    i = lo
    for j in range(lo, hi):
        comparisons += 1
        if array[j]<pivot:
            array[i], array[j] = array[j], array[i]
            if i!=j: 
                swaps += 1
            i += 1

    array[i], array[hi-1] = array[hi-1], array[i]
    if i!=hi-1: 
        swaps += 1
    array, comparisons, swaps = quick_sort_(array, lo, i, comparisons, swaps)
    array, comparisons, swaps = quick_sort_(array, i+1, hi, comparisons, swaps)
    return array, comparisons, swaps

def quick_sort(array):
    array, comprisons, swaps = quick_sort_(array, 0, len(array), 0, 0)
    return array, comprisons, swaps

# %%
import os
import numpy as np
import pandas as pd

def add_result_to_csv(result_datapoint, file_name):
    for key, val in result_datapoint.items():
        result_datapoint[key] = [val, ]
    
    if os.path.exists(file_name):
        result_df = pd.read_csv(file_name, index_col=0)
        tmp_df = pd.DataFrame(result_datapoint)
        result_df = pd.concat([result_df, tmp_df], ignore_index = True)
        result_df.to_csv(file_name)
    else:
        result_df = pd.DataFrame(result_datapoint)  
        result_df.to_csv(file_name) 

# save results into .csv
file_dir = "./sorting"
if not os.path.exists(file_dir):
    os.makedirs(file_dir)
file_name = os.path.join(file_dir, "insertion_sort.csv")

for _ in range(100):
    arr = np.random.randint(0, 100, 5)
    instance = {
        "input": list(arr.copy()),
    }
    output_arr, intermediate_steps, comparisons, swaps = insertion_sort(arr)
    instance.update({
        "output": list(output_arr.copy()),
        "intermediate_steps": intermediate_steps,
    })
    add_result_to_csv(instance, file_name)

# %%

instance_df = pd.read_csv(file_name, index_col=0)

# %%
import numpy as np

arr = np.random.randint(0, 100, 5)

print(arr.copy())
print("Insertion sort:")
insertion_sort(arr.copy())
print("Bubble sort:")
bubble_sort(arr.copy())

