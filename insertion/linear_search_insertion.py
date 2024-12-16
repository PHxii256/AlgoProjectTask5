import time, pandas as pd

# linear_search_insertion.py
def insert_linear(df, new_data_str):
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
    
    # Set of phone numbers already present in the DataFrame
    existing_numbers = df['phone_number'].tolist()
    
    # Check if the phone number already exists in the DataFrame
    if new_data['phone_number'] not in existing_numbers:
        # Use pd.concat instead of append
        df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
    
    return df, "Inserted 1 entry using Linear Search."
