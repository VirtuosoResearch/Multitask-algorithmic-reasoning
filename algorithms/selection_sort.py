from copy import deepcopy

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