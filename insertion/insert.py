import pandas as pd
import time
from hash_map_insertion import insert_hash_map
from linear_search_insertion import insert_linear

def main():
    print("Welcome to the Database Insertion App")
    
    # Prompt user to provide the path to the CSV file
    csv_file = input("Enter the path to your CSV file: ").strip()
    
    # Load the CSV file into a DataFrame
    try:
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} entries from {csv_file}.")
    except FileNotFoundError:
        print(f"File not found: {csv_file}. Please ensure the file exists.")
        return
    
    print("Please choose an option:")
    print("1. Insert using Hash Map")
    print("2. Insert using Linear Search")
    print("3. Exit")
    
    choice = input("Enter your choice (1/2/3): ").strip()
    
    if choice in ['1', '2']:
        # Prompt for the new entry string
        new_entry_str = input("Enter the new entry string (comma-separated): ").strip()
        
        # Measure execution time for the selected insertion method
        start_time = time.time()
        
        if choice == '1':
            df, message = insert_hash_map(df, new_entry_str)
        elif choice == '2':
            df, message = insert_linear(df, new_entry_str)
        
        end_time = time.time()
        
        # Save the updated DataFrame back to the CSV file
        df.to_csv(csv_file, index=False)
        
        # Display the message and timing information
        print(message)
        print(f"Operation completed in {end_time - start_time:.4f} seconds.")
    elif choice == '3':
        print("Exiting the application. Goodbye!")
    else:
        print("Invalid choice. Please try again.")
        main()

if __name__ == "__main__":
    main()
