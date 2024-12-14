**We got two of the best algorithms**

#### Merge Sort

- **Algorithm Type**:
    - Merge Sort is a **divide-and-conquer** algorithm that divides the array into halves, recursively sorts them, and merges the results.
- **Time Complexity**:
    - **Best Case**: $O(nlog⁡n)$
    - **Average Case**: $O(nlog⁡n)$
    - **Worst Case**: $O(nlog⁡n)$
- **Space Complexity**:
    - Requires $O(n)$ additional space for merging.
- **Advantages**:
    - Merge Sort is **stable**, <u>meaning duplicate entries retain their relative order</u>.
    - Performs well on **linked lists** (avoids extra memory for rearranging).
    - Guaranteed $O(nlog⁡n)$ performance regardless of data distribution.
- **Disadvantages**:
    - Can be slower for small datasets.

```python
def merge(arr, l, m, r):

    n1 = m - l + 1
    n2 = r - m
    # create temp arrays
    L = [0] * (n1)
    R = [0] * (n2)

    # Copy data to temp arrays L[] and R[]

    for i in range(0, n1):
        L[i] = arr[l + i]
    for j in range(0, n2):
        R[j] = arr[m + 1 + j]

    # Merge the temp arrays back into arr[l..r]

    i = 0     # Initial index of first subarray
    j = 0     # Initial index of second subarray
    k = l     # Initial index of merged subarray

    while i < n1 and j < n2:

        if L[i]["first_name"]+L[i]["last_name"] <= R[j]["first_name"]+ R[j]["last_name"]:
            arr[k] = L[i]
            i += 1
        else:
            arr[k] = R[j]
            j += 1
        k += 1

    # Copy the remaining elements of L[], if there are any

    while i < n1:

        arr[k] = L[i]

        i += 1

        k += 1

    # Copy the remaining elements of R[], if there are any

    while j < n2:
        arr[k] = R[j]
        j += 1
        k += 1

# l is for left index and r is right index of the
# sub-array of arr to be sorted

def mergeSort(arr, l, r):

    if l < r:
        # Same as (l+r)//2, but avoids overflow for
        # large l and h
        m = l+(r-l)//2
        # Sort first and second halves
        mergeSort(arr, l, m)
        mergeSort(arr, m+1, r)
        merge(arr, l, m, r)

```

#### Quick Sort:

1. **Algorithm Type**:
    - Quick Sort is also a **divide-and-conquer** algorithm but partitions the array around a pivot and recursively sorts the partitions.
2. **Time Complexity**:
    - **Best Case**: $O(nlog⁡n)$
    - **Average Case**: $O(nlog⁡n)$
    - **Worst Case**: $O(n^2)$ (occurs when pivot selection is poor, e.g., sorted or nearly sorted data).
3. **Space Complexity**:
    - Requires $O(log⁡n)$ space for recursion (in-place algorithm).
4. **Stability**:
    - Quick Sort is typically **not stable**, but it can be made stable with modifications.
5. **Advantages**:
    - Often faster than Merge Sort for **in-place sorting** due to lower constant factors.
    - Minimal additional memory usage.
6. **Disadvantages**:
    - Worst-case time complexity can degrade to $O(n2)$.

```python 
def quickSort(names):

    if len(names) <= 1:
        return names
        
    pivot = names[len(names) // 2]  # Choose the middle element as the pivot
    left = [name for name in names if name < pivot]  # Elements less than pivot
    middle = [name for name in names if name == pivot]  # Elements equal to pivot
    right = [name for name in names if name > pivot]  # Elements greater than pivot
    return quickSort(left) + middle + quickSort(right)

```


| **Criteria**                | **Merge Sort**                                    | **Quick Sort**                              |
| --------------------------- | ------------------------------------------------- | ------------------------------------------- |
| **Time Complexity**         | $O(nlog⁡n)$ in all cases                          | $O(nlog⁡n)$ on average, $O(n^2)$ worst-case |
| **Space Complexity**        | $O(n)$                                            | $O(log⁡n)$                                  |
| **Stability**               | Stable                                            | Not stable                                  |
| **Practical Speed**         | Slower than Quick Sort for small datasets         | Faster for small datasets                   |
| **Dataset Characteristics** | Handles duplicate and multi-column sorting better | Works better with random data distribution  |

#### Which is better for my system?

**Merge sort is better for our case cuz it is more stable and better in worse case scenarios.**

while quick sort is more memory-efficient, it can degrade to $n^2$.
