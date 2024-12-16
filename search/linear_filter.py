import csv
import time

start_time = time.time()

def linear_filter(city):
    with open("../data.csv") as file:
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