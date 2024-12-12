import csv
import time 
from B_Tree import BTree

start_time = time.time()

def main():
    btree = BTree(7) # for sepcfiying the degree of the tree 
    with open("../data.csv", "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            btree.insert(row["city"], row)

    
    city = input("Enter city name: ")
    result = btree.search(btree.root, city)
    if result:
        print("City found:", result)
    else:
        print("City not found.")
            
            

if __name__ == "__main__":
    main()

end_time = time.time()
execution_time = end_time - start_time

print(f"Execution time: {execution_time:.4f} seconds") 