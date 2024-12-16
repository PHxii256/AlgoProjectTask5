def B_tree(csv_file):
    btree = BTree(100)
    with open(csv_file, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            btree.insert(row["city"], row)
    return btree

def linear_filtering(csv_file, city):
    print("City name: ", end="")
    return linear_filter(city)
