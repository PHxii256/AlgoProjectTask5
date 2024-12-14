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
        # Use pd.concat instead of append
        df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
        hash_map[phone_number] = new_data
    
    # Write the updated DataFrame back to the CSV file once
    df.to_csv(csv_file, index=False)
    
    end_time = time.time()
    return f"Inserted 1 entry in {end_time - start_time:.4f} seconds"

# Example new data to insert (single entry string)
new_entry_str = "+86-882-726-2942,Assem,Clute,eclute0@fc2.com,Jiading,8 Graedel Center"

# Call the function to insert a single entry using hash map
csv_file = '../data.csv'  # Make sure to update with the correct path to your CSV file
print(insert_hash_map(csv_file, new_entry_str))
