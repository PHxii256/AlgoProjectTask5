You need obsidian charts plugin to see all the charts!

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

```python
import csv
import math
import time

def load_csv(file_path):
    """Load CSV data into a sorted list of dictionaries by phone_number."""
    data = []
    with open(file_path, "r") as csvfile:
        reader = csv.DictReader(csvfile)
        data.extend(row for row in reader)
    data.sort(key=lambda x: x["phone_number"])
    return data

# Binary Search Implementation
def binary_search(data, phone_number):
    """Perform binary search to find a phone number in sorted data."""
    low, high = 0, len(data) - 1
    while low <= high:
        mid = (low + high) // 2
        if data[mid]["phone_number"] == phone_number:
            return data[mid]
        elif data[mid]["phone_number"] < phone_number:
            low = mid + 1
        else:
            high = mid - 1
    return None

# Binary Search Usage and Timing
def binary_search_main():
    file_path = "data1000rows.csv"  # Path to the CSV file
    data = load_csv(file_path)
    search_number = "+351-656-361-0451"  # Replace with the phone number to search

    # Measure and display Binary Search results
    start_time = time.time()
    result_binary = binary_search(data, search_number)
    elapsed_time = time.time() - start_time

    if result_binary:
        print("Binary Search Result:", result_binary)
    else:
        print("Phone number not found (Binary Search).")
    print(f"Binary Search took {elapsed_time:.6f} seconds.")

# Jump Search Implementation
def jump_search(data, phone_number):
    """Perform jump search to find a phone number in sorted data."""
    n = len(data)
    step = int(math.sqrt(n))
    prev = 0

    while prev < n and data[min(prev + step, n) - 1]["phone_number"] < phone_number:
        prev += step
        if prev >= n:
            return None

    for i in range(prev, min(prev + step, n)):
        if data[i]["phone_number"] == phone_number:
            return data[i]
    return None

# Jump Search Usage and Timing
def jump_search_main():
    file_path = "data1000rows.csv"  # Path to the CSV file
    data = load_csv(file_path)
    search_number = "+351-656-361-0451"  # Replace with the phone number to search

    # Measure and display Jump Search results
    start_time = time.time()
    result_jump = jump_search(data, search_number)
    elapsed_time = time.time() - start_time

    if result_jump:
        print("Jump Search Result:", result_jump)
    else:
        print("Phone number not found (Jump Search).")
    print(f"Jump Search took {elapsed_time:.6f} seconds.")

if __name__ == "__main__":
    print("Running Binary Search:")
    binary_search_main()
    print("\nRunning Jump Search:")
    jump_search_main()
```

# Task 3

**Assigned to:** Omar Asaad -> 200027710
Our phonebook can be large, you could have hundreds of people maybe even thousands, all from different cities, so we need a fast and efficient algorithm to look people up quickly from their city names, 2 ways are B-Trees and Linear filtering.

#### B-Trees

A **B-tree** is a self-balancing **search tree** that maintains sorted data and allows efficient insertion, deletion, and search operations. It is widely used in database systems and file systems to organize large amounts of data that cannot fit entirely in memory.

##### Key Features:

1. **Structure**:
    - A B-tree node can contain multiple keys and child pointers.
    - Each node has a minimum ( ⌈m/2⌉-1) and maximum (m-1) keys, where **m** is the tree's order (maximum number of child pointers per node).
    - All leaves are at the same level, ensuring balance.
2. **Properties**:
    - Keys in a node are kept in sorted order.
    - For a node with `n` keys, it has `n+1` children, and keys in the children’s subtrees fall between the keys in the parent node.
    - A B-tree grows and shrinks dynamically, avoiding excessive rebalancing.

##### Time Complexity:
The search will take O(log n)
##### Space Complexity:
- **Space per node**: O(m), where m is the order of the tree.
- **Total space**: Depends on the total number of keys stored, generally O(n), where  is the number of keys.

##### Technique
 **B-Tree** belongs to **balanced search trees**


```PYTHON
import time
start_time = time.time()

class BTreeNode:
    def __init__(self, t, leaf=False)
        self.t = t
        self.leaf = leaf
        self.keys = []
        self.values = []
        self.children = []

class BTree:

    def __init__(self, t):
        self.t = t
        self.root = BTreeNode(t, True)

    def search(self, node, key):
        i = 0
        while i < len(node.keys) and key > node.keys[i]:
            i += 1
        if i < len(node.keys) and key == node.keys[i]:
            return f"{node.values[i]}, \ntime: {execution_time:.7f}"
        if node.leaf:
            return None
        return self.search(node.children[i], key)

    def insert(self, key, value):
        if len(self.root.keys) == 2 * self.t - 1:
            new_root = BTreeNode(self.t, False)
            new_root.children.append(self.root)
            self.split_child(new_root, 0)
            self.root = new_root
            
        current_node = self.root

        while not current_node.leaf:
            i = len(current_node.keys) - 1
            while i >= 0 and key < current_node.keys[i]:
                i -= 1
            i += 1
            if len(current_node.children[i].keys) == 2 * self.t - 1:
                self.split_child(current_node, i)
                if key > current_node.keys[i]:
                    i += 1

            current_node = current_node.children[i]
        i = len(current_node.keys) - 1
        while i >= 0 and key < current_node.keys[i]
            i -= 1
        current_node.keys.insert(i + 1, key)
        current_node.values.insert(i + 1, value)

  
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

  
end_time = time.time()
execution_time = end_time - start_time
```

#### Linear Filtering
**Linear Filtering** is a simple and straightforward searching algorithm. It involves checking each element in a list (or file) one by one to find a target value. Here's a summary of its key aspects:
##### Technique:
- **Linear Search** belongs to a category of **comparison-based search algorithms**.
- It is considered an **unoptimized search algorithm** due to its sequential nature.
##### Time Complexity:
- **Best Case**: O(1) – The target value is found in the first comparison.
- **Worst Case**: O(n) – Every element in the list must be checked before finding the target (if the list is not sorted, n comparisons are necessary).
- **Average Case**: O(n) – Similar to the worst case, as on average, the search requires checking half of the elements.
##### Space Complexity:
- O(1) – The amount of extra space required does not depend on the size of the input. The algorithm only requires a few variables for comparisons and indexes, irrespective of the list's size.
```python

import csv
import time

start_time = time.time()


def linear_filter(city):
    with open("../10k.csv") as file:
        reader = csv.DictReader(file)
        found = False
        for row in reader:
            if row['city'] == city:
                return f"{row}, \ntime: {execution_time:.9f}"  
        if not found:
            return "No matching city found."
            
end_time = time.time()
execution_time = end_time - start_time


if __name__ == "__main__":

    linear_filter()

```

## which is better?

**B-Tree** wins, simply because of its time complexity and speed, even though linear filter is close, that is due to the small dataset 
and B-Tree will perform even better for bigger datasets with millions and billions of rows.

|                  | 1000 row    | 10000 row   | 100000 row  |
| ---------------- | ----------- | ----------- | ----------- |
| B-Tree           | 0.000000167 | 0.000000145 | 0.000000132 |
| Linear filtering | 0.000000477 | 0.000000715 | 0.000000954 |
^searching

```chart
type: line
id: searching
layout: rows
width: 80%
beginAtZero: true
```

# Task 4

**Assigned to:** Mohamed Badawy -> 200021612

**1. Array-based Deletion**  
In an array-based approach, contacts are stored in a list, and we iterate through the list to locate and remove a contact based on its phone number.

### Array-based Implementation

```python
class Contact:
    def __init__(self, name, phone_number):
        self.name = name
        self.phone_number = phone_number

    def __str__(self):
        return f"{self.name}: {self.phone_number}"


def delete_contact_array(contacts, phone_number):
    for i, contact in enumerate(contacts):
        if contact.phone_number == phone_number:
            del contacts[i]
            return f"Contact with phone number {phone_number} deleted."
    return f"Contact with phone number {phone_number} not found."


# Example Usage
contacts_array = [
    Contact("Alice", "123-456-7890"),
    Contact("Bob", "234-567-8901"),
    Contact("Charlie", "345-678-9012")
]

print("Before Deletion:")
for contact in contacts_array:
    print(contact)

result = delete_contact_array(contacts_array, "234-567-8901")
print(result)

print("\nAfter Deletion:")
for contact in contacts_array:
    print(contact)
```

---

**2. Linked-List-based Deletion**  
The linked-list-based approach stores each contact in a node, linked sequentially. Deletion involves traversing the list to find the node with the specified phone number and updating pointers to unlink it.

### Linked-List Implementation

```python
class Contact:
    def __init__(self, name, phone_number):
        self.name = name
        self.phone_number = phone_number

    def __str__(self):
        return f"{self.name}: {self.phone_number}"


class Node:
    def __init__(self, contact=None):
        self.contact = contact
        self.next = None


class LinkedList:
    def __init__(self):
        self.head = None

    def append(self, contact):
        new_node = Node(contact)
        if not self.head:
            self.head = new_node
            return
        last_node = self.head
        while last_node.next:
            last_node = last_node.next
        last_node.next = new_node

    def delete_contact_linked_list(self, phone_number):
        current = self.head
        previous = None

        # If the head node holds the phone number
        if current and current.contact.phone_number == phone_number:
            self.head = current.next
            return f"Contact with phone number {phone_number} deleted."

        # Traverse the list to find the contact
        while current and current.contact.phone_number != phone_number:
            previous = current
            current = current.next

        if not current:  # Contact not found
            return f"Contact with phone number {phone_number} not found."

        # Unlink the node
        previous.next = current.next
        return f"Contact with phone number {phone_number} deleted."

    def display(self):
        current = self.head
        while current:
            print(current.contact)
            current = current.next


# Example Usage
linked_list = LinkedList()
linked_list.append(Contact("Alice", "123-456-7890"))
linked_list.append(Contact("Bob", "234-567-8901"))
linked_list.append(Contact("Charlie", "345-678-9012"))

print("Before Deletion:")
linked_list.display()

result = linked_list.delete_contact_linked_list("234-567-8901")
print(result)

print("\nAfter Deletion:")
linked_list.display()
```

---

**Comparison of Array-based and Linked-List-based Deletion**

|**Aspect**|**Array-based Deletion**|**Linked-List-based Deletion**|
|---|---|---|
|**Search Complexity**|O(n)|O(n)|
|**Deletion Complexity**|O(n) (due to shifting)|O(1) (adjusting pointers)|
|**Space Complexity**|O(n)|O(n) (plus extra memory for pointers)|
|**Memory Management**|Contiguous memory|Non-contiguous memory|
|**Random Access**|O(1)|O(n) (must traverse the list)|
|**Ease of Implementation**|Simple|Slightly more complex|

---

**When to Use Each Approach**

1. **Array-based Approach**
    
    - Ideal for small or relatively static contact lists.
    - Provides random access to elements and simplicity in implementation.
2. **Linked-List-based Approach**
    
    - Suitable for large or frequently changing contact lists.
    - Efficient for dynamic size and frequent deletions without shifting elements.

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
^sorting

```chart
type: line
id: sorting
layout: rows
width: 80%
beginAtZero: true
```


