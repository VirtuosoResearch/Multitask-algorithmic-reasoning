from copy import deepcopy

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