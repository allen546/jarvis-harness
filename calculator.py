#!/usr/bin/env python3
"""
Simple Python Calculator
Supports basic arithmetic: addition, subtraction, multiplication, division.
"""

def add(a, b):
    """Add two numbers."""
    return a + b

def subtract(a, b):
    """Subtract b from a."""
    return a - b

def multiply(a, b):
    """Multiply two numbers."""
    return a * b

def divide(a, b):
    """Divide a by b with division by zero handling."""
    if b == 0:
        raise ZeroDivisionError("Cannot divide by zero")
    return a / b

def get_number(prompt):
    """Get a valid number from user input."""
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Invalid input. Please enter a valid number.")

def get_operator():
    """Get a valid operator from user input."""
    operators = {'+': add, '-': subtract, '*': multiply, '/': divide}
    while True:
        op = input("Enter operator (+, -, *, /): ").strip()
        if op in operators:
            return op, operators[op]
        print("Invalid operator. Please enter +, -, *, or /.")

def main():
    """Main calculator function."""
    print("=== Simple Python Calculator ===")
    print("Operations: + (add), - (subtract), * (multiply), / (divide)")
    print("Type 'quit' to exit\n")
    
    while True:
        # Get first number
        try:
            num1_input = input("Enter first number (or 'quit'): ").strip()
            if num1_input.lower() == 'quit':
                print("Goodbye!")
                break
            num1 = float(num1_input)
        except ValueError:
            print("Invalid input. Please enter a valid number or 'quit'.")
            continue
        
        # Get operator
        try:
            op_symbol, operation = get_operator()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break
        
        # Get second number
        try:
            num2_input = input("Enter second number (or 'quit'): ").strip()
            if num2_input.lower() == 'quit':
                print("Goodbye!")
                break
            num2 = float(num2_input)
        except ValueError:
            print("Invalid input. Please enter a valid number or 'quit'.")
            continue
        
        # Perform calculation
        try:
            result = operation(num1, num2)
            print(f"{num1} {op_symbol} {num2} = {result}\n")
        except ZeroDivisionError as e:
            print(f"Error: {e}\n")
        except Exception as e:
            print(f"An unexpected error occurred: {e}\n")

if __name__ == "__main__":
    main()
