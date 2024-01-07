def classify_number(num):
    if num < 0:
        return "The number is negative."
    elif num == 0:
        return "The number is zero."
    else:
        return "The number is positive."

number = int(input("Enter a number: "))
print(classify_number(number))
