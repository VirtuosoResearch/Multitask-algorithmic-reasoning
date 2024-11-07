def heap_sort(arr):
    n = len(arr)

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

            # Recursively heapify the affected sub-tree
            heapify(arr, n, largest)

    # Build a max heap
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)

    # Extract elements one by one
    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]  # Swap
        heapify(arr, i, 0)
