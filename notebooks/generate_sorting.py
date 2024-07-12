from copy import deepcopy
import numpy as np
import os
import pandas as pd
import argparse

def bubble_sort(array):
    comparisons = 0
    swaps = 0
    length = len(array)
    intermediate_steps = []; intermediate_variables = []
    for _ in range(length):
        swapped = False
        for i in range(1, length):
            comparisons += 1
            if array[i - 1] > array[i]:
                array[i - 1], array[i] = array[i], array[i - 1]
                swapped = True
                swaps += 1

                intermediate_variables.append({"i": deepcopy(i-1), "j": deepcopy(i), "swap": True})
                intermediate_steps.append(list(deepcopy(array)))
            # else: 
            #     intermediate_variables.append({"i": deepcopy(i-1), "j": deepcopy(i), "swap": False})
            #     intermediate_steps.append(list(deepcopy(array)))
        if not swapped:
            break
    return array, intermediate_steps, intermediate_variables, comparisons, swaps

def insertion_sort(array):
    comparisons = 0
    swaps = 0
    intermediate_steps = []; intermediate_variables = []
    for j in range(1, len(array)):
        key = array[j]
        i = j - 1; tmp_j = j
        while i >= 0 and array[i] > key:
            array[i + 1] = array[i]
            array[i] = key
            intermediate_variables.append({"i": deepcopy(i), "j": deepcopy(tmp_j), "swap": True})
            intermediate_steps.append(list(deepcopy(array)))

            i = i - 1; tmp_j = tmp_j - 1
            comparisons += 1
            swaps += 1

        if i!=-1: 
            comparisons += 1
            # intermediate_variables.append({"i": deepcopy(i), "j": deepcopy(tmp_j), "swap": False})
            # intermediate_steps.append(list(deepcopy(array)))
            
        if i+1!=j: 
            swaps += 1
        
    return array, intermediate_steps, intermediate_variables, comparisons, swaps

def selection_sort(arrary):
    n = len(arrary)

    comparisons, swaps = 0, 0
    intermediate_steps, intermediate_variables = [], []
    # Traverse through all array elements
    for i in range(n):
        # Find the minimum element in the remaining unsorted array
        min_index = i
        tmp_array = deepcopy(arrary)
        for j in range(i + 1, n):
            if arrary[j] < arrary[min_index]:
                min_index = j

                swaps += 1
                intermediate_variables.append({"i": deepcopy(i), "j": deepcopy(j), "swap": True})
                tmp_array[i], tmp_array[j] = tmp_array[j], tmp_array[i]
                intermediate_steps.append(list(deepcopy(tmp_array)))
            # else:
            #     intermediate_variables.append({"i": deepcopy(i), "j": deepcopy(j), "swap": False})
            #     intermediate_steps.append(list(deepcopy(tmp_array)))

            comparisons += 1

        # Swap the found minimum element with the first element
        arrary[i], arrary[min_index] = arrary[min_index], arrary[i]
    return arrary, intermediate_steps, intermediate_variables, comparisons, swaps


def quick_sort(array):
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
    
    def sort(array, left, right):
        if left >= right:
            return array
        
        pivot_index = partition(array, left, right)
        sort(array, left, pivot_index - 1)
        sort(array, pivot_index + 1, right)
        return array

    return sort(array, 0, length - 1), intermediate_steps, None, None, None

def merge_sort(array):
    intermediate_steps = []

    def merge_sort(array, left, right):
        if left < right:
            mid = (left + right) // 2  # Find the middle index
            merge_sort(array, left, mid)  # Recursively sort the left half
            merge_sort(array, mid + 1, right)  # Recursively sort the right half
            merge(array, left, mid, right)  # Merge the sorted halves

    def merge(array, left, mid, right):
        swaped = False
        left_half = deepcopy(array[left:mid + 1])  # Copy the left half
        right_half = deepcopy(array[mid + 1:right + 1])  # Copy the right half

        i, j, k = 0, 0, left  # i for left_half, j for right_half, k for array

        while i < len(left_half) and j < len(right_half):
            if left_half[i] < right_half[j]:
                array[k] = left_half[i]
                i += 1
            else:
                swaped = True
                array[k] = right_half[j]
                j += 1
            k += 1

        # Check for any remaining elements in left_half
        while i < len(left_half):
            array[k] = left_half[i]
            i += 1
            k += 1

        # Check for any remaining elements in right_half
        while j < len(right_half):
            array[k] = right_half[j]
            j += 1
            k += 1

        if swaped: intermediate_steps.append(list(deepcopy(array)))

    merge_sort(array, 0, len(array) - 1)
    return array, intermediate_steps, None, None, None

def heap_sort(arr):
    n = len(arr)
    intermediate_steps = []

    def heapify(arr, n, i):
        largest = i  # Initialize the root as the largest
        left_child = 2 * i + 1
        right_child = 2 * i + 2

        # If the left child is larger than the root
        if left_child < n and arr[left_child] > arr[largest]:
            largest = left_child

        # If the right child is larger than the root
        if right_child < n and arr[right_child] > arr[largest]:
            largest = right_child

        # Swap the root if needed
        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            intermediate_steps.append(list(deepcopy(arr)))

            # Recursively heapify the affected sub-tree
            heapify(arr, n, largest)

    # Build a max heap
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)

    # Extract elements one by one
    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]  # Swap
        intermediate_steps.append(list(deepcopy(arr)))
        heapify(arr, i, 0)

    return arr, intermediate_steps, None, None, None


sorting_algorithms = {
    "insertion_sort": insertion_sort,
    "bubble_sort": bubble_sort,
    "selection_sort": selection_sort,
    "quick_sort": quick_sort,
    "merge_sort": merge_sort,
    "heap_sort": heap_sort,
}

def main(args):
    # save results into .csv
    length = args.length
    data_size = args.data_size
    file_dir = f"./data/sorting/"
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    file_name = os.path.join(file_dir, f"length_{length}.csv")

    df = None
    for _ in range(data_size):
        arr = np.random.randint(0, 100, length)
        records = [deepcopy(arr)]
        output_arr, intermediate_steps, intermediate_variables, comparisons, swaps = \
            insertion_sort(arr)
        
        # save intermediate steps
        records += [deepcopy(output_arr)]
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
    parser.add_argument("--save_dir", type=str, default="sorting")
    args = parser.parse_args()
    
    main(args)