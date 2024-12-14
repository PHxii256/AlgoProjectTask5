# Phone Book System Overview

This system serves as a tool to store and manage contact information. The key fields for each contact include:

- **Phone Number** (Primary Key, e.g., 040-2731021)
- **First Name**
- **Last Name**
- **Address**
- **City**
- **Email**

### Tasks and Algorithm Assignments

Each task corresponds to a specific operation on the phone book, with different algorithms to be implemented and evaluated:

1. **Add New Contact**
    - Algorithms: Linear Search, Hash Map
2. **Retrieve Contact by Phone Number**
    - Algorithms: Binary Search, Jump Search
3. **Search Contact by City**
    - Algorithms: Linear Filtering, B-Tree
4. **Delete Contact by Phone Number**
    - Algorithms: Array-based Deletion, Linked-List Based Deletion
5. **Sort Contacts by Name**
    - Algorithms: Merge Sort, Quick Sort

For my part, I am focusing on **Task 2**: **Retrieve Contact by Phone Number**, where I will compare two different insertion algorithms: **Linear Search** and **Hash Map**. The test will be conducted using a CSV file in Python.

---

# Task 1

**Assigned to:** Assem Mohamed -> 200030013

### **Linear Search Insertion**

In the linear search approach, we check if the phone number exists by iterating through the entire database. If the number is found, no insertion is performed, ensuring no duplicates. Here's an implementation:

```python
import random
import sys
import time

# Initialize a large array with 100 million elements
arr = [i for i in range(100_000_000)]  # 100 million elements

def insert(num):
    if num in arr:
        return -1  # Number already exists
    arr.append(num)  # Append the new number
    return len(arr) - 1  # Return the index

# Exotic test cases
test_cases = [
    0,                        # Boundary: smallest existing number
    99_999_999,               # Boundary: largest existing number
    -1,                       # Negative number: doesn't exist
    100_000_000,              # Next consecutive number: doesn't exist
    random.randint(-100, 100),# Random small number (may or may not exist)
    sys.maxsize,              # Largest integer supported by Python
    -sys.maxsize - 1,         # Smallest negative integer
    random.randint(100_000_001, 200_000_000),  # Random large non-existent number
    random.choice(arr),       # Already existing random number
    random.choice(arr) + 1    # Near-existing number (may not exist)
]

start_time = time.time()

# Running the test cases
for i, num in enumerate(test_cases, 1):
    result = insert(num)
    print(f"Test {i}: insert({num}) => {result}")
    
end_time = time.time()
print(f"Took {end_time - start_time:.4f} seconds")
```

**Output**: The linear insertion took 4.3408 seconds, demonstrating that the search for an existing number can be slow, especially with large datasets.

---

### **Hash Map Insertion**

In contrast, the hash map implementation leverages a dictionary to store the phone numbers as keys, enabling O(1) average time complexity for lookups and insertions. Here’s the implementation:

```python
import random
import sys
import time

# Initialize a hash map with 100 million elements
hash_map = {i: True for i in range(100_000_000)}  # Key: number, Value: True

def insert(num):
    if num in hash_map:
        return -1  # Number already exists
    hash_map[num] = True  # Insert into the hash map
    return len(hash_map)  # Return the total count of elements in the hash map

# Exotic test cases
test_cases = [
    0,                        # Boundary: smallest existing number
    99_999_999,               # Boundary: largest existing number
    -1,                       # Negative number: doesn't exist
    100_000_000,              # Next consecutive number: doesn't exist
    random.randint(-100, 100),# Random small number (may or may not exist)
    sys.maxsize,              # Largest integer supported by Python
    -sys.maxsize - 1,         # Smallest negative integer
    random.randint(100_000_001, 200_000_000),  # Random large non-existent number
    random.choice(range(100_000_000)),  # Already existing random number
    random.choice(range(100_000_000)) + 1  # Near-existing number (may not exist)
]

start_time = time.time()

# Running the test cases
for i, num in enumerate(test_cases, 1):
    result = insert(num)
    print(f"Test {i}: insert({num}) => {result}")
    
end_time = time.time()
print(f"Took {end_time - start_time:.4f} seconds")
```

**Output**: The hash map insertion took 0.0001 seconds, demonstrating a significant performance improvement compared to the linear search method.

---

### **Key Adjustments in the Hash Map Approach**

1. **Hash Map Initialization**:
    - A dictionary is used to store phone numbers as keys, with their values set to `True`.
    - This enables fast lookups and insertions with O(1) average time complexity.
2. **Modified Insert Function**:
    - Instead of searching through an array, the insert function checks the hash map for the phone number.
    - If the number doesn't exist, it is added to the hash map.
3. **Test Cases**:
    - The same set of test cases were used to ensure comparability between the list-based and hash map-based approaches.

---

### **Performance Comparison**

- **List-Based Approach**: With a time complexity of O(n) for searching and inserting, the list-based approach becomes slower as the dataset grows. In this case, with 100 million elements, the search took several seconds.
    
- **Hash Map-Based Approach**: With an average time complexity of O(1) for both lookup and insertion, the hash map approach is much faster and more efficient, even with large datasets.
    

---

### **Conclusion: Is Hash Map Useful in This Scenario?**

Although the hash map approach demonstrates superior performance for large datasets, in this particular scenario where we're using CSV files, the difference between both methods is relatively small and falls within the margin of error. This is due to the inherent limitations of working with CSV files, which are not optimized for high-performance searching or insertion operations.

Thus, for this use case, the additional complexity of using a hash map may not be necessary, especially when working with CSVs, which do not provide the same level of performance benefits as a proper database. Consequently, while hash maps are typically faster, the performance improvement may not be significant enough to justify their use in this scenario.

---

### **Linear Search Insertion Implementation**

This method performs the insertion using a linear search approach by checking if the phone number already exists in the CSV file.

```python
import pandas as pd
import time

def insert_linear(csv_file, new_data_str):
    # Split the input string by commas to extract the components
    components = new_data_str.split(',')
    new_data = {
        "phone_number": components[0],  # Assuming phone number is the first component
        "first_name": components[1],
        "last_name": components[2],
        "email": components[3],
        "city": components[4],
        "address": components[5]
    }
    
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)
    
    start_time = time.time()
    
    # Set of phone numbers already present in the DataFrame
    existing_numbers = df['phone_number'].tolist()
    
    # Check if the phone number already exists in the DataFrame
    if new_data['phone_number'] not in existing_numbers:
        # Use pd.concat instead of append to avoid modifying the original DataFrame
        df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
    
    # Write the updated DataFrame back to the CSV file
    df.to_csv(csv_file, index=False)
    
    end_time = time.time()
    return f"Inserted 1 entry in {end_time - start_time:.4f} seconds"

# Example new data to insert (single entry string)
new_entry_str = "+86-882-726-2942,Assem,Clute,eclute0@fc2.com,Jiading,8 Graedel Center"

# Call the function to insert a single entry
csv_file = 'contacts.csv'  # Make sure to update with the correct path to your CSV file
print(insert_linear(csv_file, new_entry_str))
```

### **Hash Map Insertion Implementation**

This method uses a hash map (Python dictionary) for faster lookups and insertion by converting the phone numbers into dictionary keys.

```python
import pandas as pd
import time

# Function to insert a single entry using Hash Map (Python Dictionary)
def insert_hash_map(csv_file, new_data_str):
    # Split the input string by commas to extract the components
    components = new_data_str.split(',')
    new_data = {
        "phone_number": components[0],  # Assuming phone number is the first component
        "first_name": components[1],
        "last_name": components[2],
        "email": components[3],
        "city": components[4],
        "address": components[5]
    }
    
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)
    
    # Convert the DataFrame column to a list of phone numbers
    phone_list = df['phone_number'].tolist()

    # Create the dictionary with phone numbers as keys and their indices as values
    hash_map = dict(zip(phone_list, range(len(phone_list))))

    start_time = time.time()

    # Get the phone number from the new data
    phone_number = new_data['phone_number']
    
    # Insert the new data into the hash map if the phone number doesn't exist
    if phone_number not in hash_map:
        # Use pd.concat instead of append to avoid modifying the original DataFrame
        df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
        hash_map[phone_number] = new_data
    
    # Write the updated DataFrame back to the CSV file
    df.to_csv(csv_file, index=False)
    
    end_time = time.time()
    return f"Inserted 1 entry in {end_time - start_time:.4f} seconds"

# Example new data to insert (single entry string)
new_entry_str = "+86-882-726-2942,Assem,Clute,eclute0@fc2.com,Jiading,8 Graedel Center"

# Call the function to insert a single entry using hash map
csv_file = 'contacts.csv'  # Make sure to update with the correct path to your CSV file
print(insert_hash_map(csv_file, new_entry_str))
```

### **Key Considerations**

1. **Data Input Format**: Both methods expect the data to be passed as a comma-separated string, which is split and used to form a dictionary representing a contact.
2. **Performance**: For small datasets, the performance difference between the two methods is negligible. For larger datasets, the hash map offers better performance by reducing the time complexity for lookups.
3. **Efficiency**: The CSV file is read once, and the changes are written back after the insertion, ensuring no modifications are done during the iteration.


# Task 2

**Assigned to:** Omar Sherif -> 200027721
Not done yet

# Task 3

**Assigned to:** Omar Asaad -> 200027710

Our phonebook can be quite large, potentially containing hundreds or even thousands of people from different cities. Therefore, we need a fast and efficient algorithm to quickly look people up by their city names. Two common approaches for this are B-Trees and Linear Filtering.

### B-Trees

A **B-tree** is a self-balancing **search tree** that maintains sorted data, supporting efficient insertion, deletion, and search operations. It is widely used in database and file systems for managing large data sets that cannot entirely fit into memory.

#### Key Features:

1. **Structure**:
    - A B-tree node contains multiple keys and child pointers.
    - Each node has a minimum (⌈m/2⌉-1) and maximum (m-1) number of keys, where **m** represents the tree's order (the maximum number of child pointers per node).
    - All leaves are at the same level, ensuring the tree remains balanced.
2. **Properties**:
    - Keys within a node are kept in sorted order.
    - A node with `n` keys has `n+1` children, and the keys in the children's subtrees fall between the keys in the parent node.
    - The B-tree dynamically grows and shrinks, minimizing unnecessary rebalancing.

#### Time Complexity:

- Searching takes **O(log n)** time.

#### Space Complexity:

- **Space per node**: O(m), where **m** is the tree's order (maximum number of child pointers).
- **Total space**: O(n), where **n** is the number of keys stored in the tree.

```python
class BTreeNode:

    def __init__(self, t, leaf=False):
        self.t = t
        self.leaf = leaf
        self.keys = []  # List of keys (cities)
        self.values = []  # List of values (rows)
        self.children = []  # List of children nodes

class BTree:

    def __init__(self, t):
        self.t = t  # Minimum degree (t)
        self.root = BTreeNode(t, True)

    def search(self, node, key):
        i = 0
        while i < len(node.keys) and key > node.keys[i]:
            i += 1
        if i < len(node.keys) and key == node.keys[i]:
            return node.values[i]  # Return the corresponding row (value)
        if node.leaf:
            return None
        return self.search(node.children[i], key)

    def insert(self, key, value):
        if len(self.root.keys) == 2 * self.t - 1:
            new_root = BTreeNode(self.t, False)
            new_root.children.append(self.root)
            self.split_child(new_root, 0)
            self.root = new_root
        self.insert_non_full(self.root, key, value)

    def insert_non_full(self, node, key, value):
        i = len(node.keys) - 1
        if node.leaf:
            while i >= 0 and key < node.keys[i]:
                i -= 1
            node.keys.insert(i + 1, key)
            node.values.insert(i + 1, value)
        else:
            while i >= 0 and key < node.keys[i]:
                i -= 1
            i += 1
            if len(node.children[i].keys) == 2 * self.t - 1:
                self.split_child(node, i)
                if key > node.keys[i]:
                    i += 1
            self.insert_non_full(node.children[i], key, value)

    def split_child(self, parent, index):
        t = self.t
        node = parent.children[index]
        new_node = BTreeNode(t, node.leaf)
        parent.keys.insert(index, node.keys[t - 1])
        parent.values.insert(index, node.values[t - 1])
        new_node.keys = node.keys[t:]
        new_node.values = node.values[t:]
        node.keys = node.keys[:t - 1]
        node.values = node.values[:t - 1]
        if not node.leaf:
            new_node.children = node.children[t:]
            node.children = node.children[:t]
        parent.children.insert(index + 1, new_node)

    def print_tree(self, node, level=0):
        print('Level', level, 'Keys:', node.keys)
        if not node.leaf:
            for child in node.children:
                self.print_tree(child, level + 1)
```

### Linear Filtering

**Linear Filtering** refers to applying a linear operation, such as a convolution or spatial filter, to a dataset (e.g., a signal or image). The goal is often to smooth, enhance, or modify the data based on the applied filter.

#### Key Features:

- **Definition**: Linear filtering applies a filter (kernel or set of weights) to an input dataset to produce an output. In the case of a CSV file, it could mean processing rows based on certain criteria (e.g., summing values or applying a function across columns).
- **Operations**:
    - **Time Complexity**: For a CSV file with **n** rows and **m** columns, the time complexity is **O(n × m)**. This accounts for the operations required to read and process each row and column.
    - **Space Complexity**: The space complexity is also **O(n × m)**, representing the memory required to store the data.

#### Searching through a CSV File:

- **Definition**: Searching for a specific key in a CSV file to retrieve corresponding values.
- **Time Complexity**:
    - **For 100 rows**: **O(100)**.
    - **For 500 rows**: **O(500)**.
    - **For 1000 rows**: **O(1000)**.
    - Time complexity increases linearly with the number of rows.
- **Space Complexity**: The space complexity is **O(n)**, as we may need to store intermediate search results or copies of data.

### Summary:

- **Linear Filtering** involves applying operations across all rows and columns of the CSV file, with both time and space complexities proportional to the number of rows and columns **O(n × m)**.
- **Searching in a CSV file** for a key is a linear search operation, where time complexity grows linearly with the number of rows **O(n)** and space complexity is also **O(n)**.
- Both methods are considered linear algorithms due to their direct relationship with the data size.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

def get_city_data(city, csv_file_path):
    data = pd.read_csv(csv_file_path)
    if city in data.iloc[:, 0].values:
        row = data[data.iloc[:, 0] == city].iloc[0, 1:].values
        return row
    else:
        return None

# Example usage with the uploaded file
csv_file_path = '../data/data.csv'
city = input("Enter the city name: ")
city_row = get_city_data(city, csv_file_path)
if city_row is not None:
    print(f"Data for {city}: {city_row}")
else:
    print(f"City {city} not found in the data.")

# Input arrays (example inputs, replace with actual data)
X = np.array([[0, 0], [0, 0]])  # Replace with the actual image matrix
corr = np.array([[1, 1], [1, 1]])  # Replace with the actual filter kernel

# Padding sizes
pad1 = corr.shape[0] - 1
pad2 = corr.shape[1] - 1

# Initialize output
output = np.zeros_like(X, dtype=np.uint8)

if corr.shape[0] == 1:
    Y = np.zeros((X.shape[0], X.shape[1] + pad2))
    n = 0
    m = corr.shape[1] // 2
    sz1 = Y.shape[0]
    sz2 = Y.shape[1] - pad2
elif corr.shape[1] == 1:
    Y = np.zeros((X.shape[0] + pad1, X.shape[1]))
    n = corr.shape[0] // 2
    m = 0
    sz1 = Y.shape[0] - pad1
    sz2 = Y.shape[1]
else:
    Y = np.zeros((X.shape[0] + pad1, X.shape[1] + pad2))
    n = corr.shape[0] // 2
    m = corr.shape[1] // 2
    sz1 = Y.shape[0] - pad1
    sz2 = Y.shape[1] - pad2

# Copy X into padded Y
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Y[i + n, j + m] = X[i, j]

# Perform the filtering
szcorr1 = corr.shape[0]
szcorr2 = corr.shape[1]
for i in range(sz1):
    for j in range(sz2):
        sum_val = 0
        n_i = i
        m_j = j
        for a in range(szcorr1):
            for b in range(szcorr2):
                sum_val += Y[n_i, m_j] * corr[a, b]
                m_j += 1
            m_j = j
            n_i += 1
        output[i, j] = sum_val

# Display the result
plt.imshow(output, cmap='gray')
plt.title('After linear filtering')
plt.show()
```

| |100 row|500 row|1000 row|
|---|---|---|---|
|B-Tree|0.8844|0.7633|0.6630|
|Linear filtering|600|3000|6000|


# Task 4

**Assigned to:** Mohamed Badawy -> 200021612
Not done yet

# Task 5

**Assigned to:** Mohamed Ayman -> 200026500

**Two of the Best Sorting Algorithms**

### Merge Sort

- **Algorithm Type**:  
    Merge Sort is a **divide-and-conquer** algorithm that splits the array into halves, recursively sorts them, and merges the sorted halves back together.
    
- **Time Complexity**:
    
    - **Best Case**: O(nlog⁡n)
    - **Average Case**: O(nlog⁡n)
    - **Worst Case**: O(nlog⁡n)
- **Space Complexity**:  
    Requires **O(n)** extra space for merging.
    
- **Advantages**:
    
    - Merge Sort is **stable**, meaning that duplicate elements retain their relative order.
    - It works well with **linked lists**, as it doesn't require extra memory for rearranging.
    - It guarantees O(nlog⁡n) performance regardless of the input data distribution.
- **Disadvantages**:
    
    - Merge Sort can be slower for smaller datasets.

```python
def merge(arr, l, m, r):
    n1 = m - l + 1
    n2 = r - m
    # Create temporary arrays
    L = [0] * n1
    R = [0] * n2

    # Copy data to temporary arrays L[] and R[]
    for i in range(n1):
        L[i] = arr[l + i]
    for j in range(n2):
        R[j] = arr[m + 1 + j]

    # Merge the temporary arrays back into arr[l..r]
    i = j = k = 0
    while i < n1 and j < n2:
        if L[i]["first_name"] + L[i]["last_name"] <= R[j]["first_name"] + R[j]["last_name"]:
            arr[k] = L[i]
            i += 1
        else:
            arr[k] = R[j]
            j += 1
        k += 1

    # Copy remaining elements of L[] and R[] if any
    while i < n1:
        arr[k] = L[i]
        i += 1
        k += 1
    while j < n2:
        arr[k] = R[j]
        j += 1
        k += 1

def mergeSort(arr, l, r):
    if l < r:
        m = l + (r - l) // 2
        mergeSort(arr, l, m)
        mergeSort(arr, m + 1, r)
        merge(arr, l, m, r)
```

### Quick Sort

- **Algorithm Type**:  
    Quick Sort is also a **divide-and-conquer** algorithm, but it partitions the array around a pivot element and recursively sorts the subarrays.
    
- **Time Complexity**:
    
    - **Best Case**: O(nlog⁡n)
    - **Average Case**: O(nlog⁡n)
    - **Worst Case**: O(n^2) (occurs when the pivot selection is poor, e.g., for already sorted or nearly sorted data).
- **Space Complexity**:  
    Quick Sort is an **in-place** algorithm, requiring only **O(log⁡n)** space for recursion.
    
- **Stability**:  
    Quick Sort is **not stable**, but it can be modified to be stable.
    
- **Advantages**:
    
    - Quick Sort tends to be faster than Merge Sort for **in-place sorting** due to smaller constant factors.
    - It has minimal additional memory requirements.
- **Disadvantages**:
    
    - Quick Sort’s worst-case time complexity is O(n^2), making it potentially inefficient for specific data distributions.

```python
def quickSort(names):
    if len(names) <= 1:
        return names
    
    pivot = names[len(names) // 2]  # Choose the middle element as pivot
    left = [name for name in names if name < pivot]  # Elements less than pivot
    middle = [name for name in names if name == pivot]  # Elements equal to pivot
    right = [name for name in names if name > pivot]  # Elements greater than pivot
    return quickSort(left) + middle + quickSort(right)
```

---

|**Criteria**|**Merge Sort**|**Quick Sort**|
|---|---|---|
|**Time Complexity**|O(nlog⁡n) in all cases|O(nlog⁡n) on average, O(n^2) in the worst-case|
|**Space Complexity**|O(n)|O(log⁡n)|
|**Stability**|Stable|Not stable|
|**Practical Speed**|Slower for small datasets|Faster for small datasets|
|**Dataset Characteristics**|Better for handling duplicates and multi-column sorting|More efficient with random data distribution|

#### Which is better for my system?

Which is better for my system? Quick Sort is better in our case, We suspect it to be due to existence of duplicates in large datasets

|       | 1k     | 10k    | 100k   |
| ----- | ------ | ------ | ------ |
| Merge | 0.0043 | 0.0579 | 0.9484 |
| Quick | 0.0025 | 0.0186 | 0.2663 |

