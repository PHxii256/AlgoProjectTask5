import csv 
from B_Tree import BTree
from linear_filter import linear_filter

def main():
    print("B-Tree[1] or Linear Search[2]")
    choice = int(input("choose: "))
    if choice == 1:
        B_tree()
    elif choice == 2:
        linear_filtering()
    else:
        print("Invalid choice.")

def B_tree():
    btree = BTree(100) # for sepcfiying the degree of the tree 
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
        
def linear_filtering():
    print("City name: ", end="")
    city = input()
    print(linear_filter(city))                               

if __name__ == "__main__":
    main() 