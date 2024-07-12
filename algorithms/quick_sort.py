from copy import deepcopy

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