import time, pandas as pd

# hash_map_insertion.py
def insert_hash_map(df, new_data_str):
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
    
    # Convert the DataFrame column to a list of phone numbers
    phone_list = df['phone_number'].tolist()

    # Create the dictionary with phone numbers as keys and their indices as values
    hash_map = dict(zip(phone_list, range(len(phone_list))))

    # Get the phone number from the new data
    phone_number = new_data['phone_number']
    
    # Insert the new data into the hash map if the phone number doesn't exist
    if phone_number not in hash_map:
        # Use pd.concat instead of append
        df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
    
    return df, "Inserted 1 entry using Hash Map."
