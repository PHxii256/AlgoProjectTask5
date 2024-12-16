import csv
import pandas as pd
import time
from insertion.hash_map_insertion import insert_hash_map
from insertion.linear_search_insertion import insert_linear
from search.B_Tree import BTree
from search.linear_filter import linear_filter
from sort.mergesort import mergeSort
from sort.quicksort import quickSort

def main():
    print("Welcome to the Database Management App")
    csv_file = input("Enter the path to your CSV file: ").strip()

    try:
        with open(csv_file, "r") as file:
            print(f"CSV file '{csv_file}' loaded successfully.")
    except FileNotFoundError:
        print(f"File '{csv_file}' not found. Please check the path.")
        return

    while True:
        print("\nChoose an option:")
        print("1. Insert Data")
        print("2. Search Data")
        print("3. Sort Data")
        print("4. Exit")

        choice = input("Enter your choice (1/2/3/4): ").strip()

        if choice == '1':
            handle_insertion(csv_file)
        elif choice == '2':
            handle_search(csv_file)
        elif choice == '3':
            handle_sorting(csv_file)
        elif choice == '4':
            print("Exiting the application. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

def handle_insertion(csv_file):
    print("\nInsertion Options:")
    print("1. Insert using Hash Map")
    print("2. Insert using Linear Search")
    choice = input("Enter your choice (1/2): ").strip()

    if choice in ['1', '2']:
        new_entry_str = input("Enter the new entry string (comma-separated): ").strip()
        try:
            df = pd.read_csv(csv_file)
            print(f"Loaded {len(df)} entries from '{csv_file}'.")

            start_time = time.time()
            if choice == '1':
                df, message = insert_hash_map(df, new_entry_str)
            elif choice == '2':
                df, message = insert_linear(df, new_entry_str)
            end_time = time.time()

            df.to_csv(csv_file, index=False)
            print(message)
            print(f"Operation completed in {end_time - start_time:.4f} seconds.")
        except Exception as e:
            print(f"An error occurred: {e}")
    else:
        print("Invalid choice for insertion.")

def handle_search(csv_file):
    print("\nSearch Options:")
    print("1. B-Tree Search")
    print("2. Linear Filter Search")
    choice = input("Enter your choice (1/2): ").strip()

    if choice == '1':
        btree = BTree(100)
        with open(csv_file, "r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                btree.insert(row["city"], row)
        city = input("Enter city name: ").strip()
        result = btree.search(btree.root, city)
        if result:
            print("City found:", result)
        else:
            print("City not found.")
    elif choice == '2':
        city = input("Enter city name: ").strip()
        print(linear_filter(city))
    else:
        print("Invalid choice for search.")

def handle_sorting(csv_file):
    try:
        DictList = []
        firstLastNamesList = []

        with open(csv_file, "r", encoding="utf8") as file:
            reader = csv.DictReader(file)
            for line in reader:
                dictEntry = {
                    "phone_number": line['phone_number'],
                    "first_name": line["first_name"],
                    "last_name": line["last_name"],
                    "email": line["email"],
                    "city": line["city"],
                    "address": line['address']
                }
                firstLastNamesList.append(line["first_name"] + " " + line["last_name"])
                DictList.append(dictEntry)

        print("\nSorted names (quick sort):\n", quickSort(firstLastNamesList))
        mergeSort(DictList, 0, len(DictList) - 1)

        with open("sorted_data.csv", "w", encoding="utf8", newline='') as sorted_file:
            writer = csv.DictWriter(sorted_file, DictList[0].keys())
            writer.writeheader()
            writer.writerows(DictList)

        print("\nSorted records by names (merge sort):\n")
        with open("sorted_data.csv", "r", encoding="utf8") as sorted_file:
            reader = csv.DictReader(sorted_file)
            for line in reader:
                print(line)
    except Exception as e:
        print(f"An error occurred during sorting: {e}")

if __name__ == "__main__":
    main()
