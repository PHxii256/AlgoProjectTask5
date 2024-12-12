import csv 
from mergesort import mergeSort
from quicksort import quickSort

DictList = [] #Dict for a single csv record
firstLastNamesList = []
with open(r'../data.csv', 'r', encoding="utf8") as csv_file:
    reader = csv.DictReader(csv_file)

    for line in reader:
          dictEntry = {
              "phone_number": line['phone_number'],
              "first_name" : line["first_name"],
              "last_name" : line["last_name"],
              "email" : line["email"],
              "city" : line["city"],
              "address": line['address']
          }
          firstLastNamesList.append(line["first_name"] + " " + line["last_name"])
          DictList.append(dictEntry) 

    print("\n sorted names (quick sort): \n\n" , quickSort(firstLastNamesList) , "\n")
    mergeSort(DictList,0, len(DictList) -1)

with open(r'sorted_data.csv', "w", encoding="utf8",newline='') as sorted_csv_file:
    sort = csv.DictWriter(sorted_csv_file, DictList[0].keys())
    sort.writeheader()
    sort.writerows(DictList)
    sorted_csv_file.close()

print("\n sorted records by names (merge sort): \n")

with open(r'sorted_data.csv', 'r', encoding="utf8") as sorted_csv_file:
    reader = csv.DictReader(sorted_csv_file)
    for line in reader:
        print(line)