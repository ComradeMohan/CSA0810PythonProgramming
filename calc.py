# Get input from the user
expression = input("Enter an expression (e.g., 5+5): ")
try:
    # Use eval() to evaluate the expression
    result = eval(expression)
    print("Result:", result)
except Exception as e:
    print("Invalid input or error:", e)
