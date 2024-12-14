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
        # Use pd.concat instead of append
        df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
    
    # Write the updated DataFrame back to the CSV file
    df.to_csv(csv_file, index=False)
    
    end_time = time.time()
    return f"Inserted 1 entry in {end_time - start_time:.4f} seconds"

# Example new data to insert (single entry string)
new_entry_str = "+86-882-726-2942,Assem,Clute,eclute0@fc2.com,Jiading,8 Graedel Center"

# Call the function to insert a single entry
csv_file = '../data.csv'  # Make sure to update with the correct path to your CSV file
print(insert_linear(csv_file, new_entry_str))
