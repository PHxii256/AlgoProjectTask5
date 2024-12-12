def quickSort(names):
    if len(names) <= 1:
        return names
    pivot = names[len(names) // 2]  # Choose the middle element as the pivot
    left = [name for name in names if name < pivot]  # Elements less than pivot
    middle = [name for name in names if name == pivot]  # Elements equal to pivot
    right = [name for name in names if name > pivot]  # Elements greater than pivot
    return quickSort(left) + middle + quickSort(right)