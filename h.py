my_list = []  # Initialize an empty list

# Insert elements based on user input
while True:
    value = input("Enter a value (or press Enter to stop): ")
    if value == "":
        break
    my_list.append(value)

print(my_list)
