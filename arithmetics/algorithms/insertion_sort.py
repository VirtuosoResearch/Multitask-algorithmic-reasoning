from copy import deepcopy

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
