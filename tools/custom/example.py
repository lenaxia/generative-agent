"""Example Custom Tool

This is a template for creating custom tools.
Copy this file and modify it to create your own tools.

Delete or modify this file as needed - it's just an example!
"""

from strands import tool


@tool
def example_calculator(num1: float, num2: float, operation: str = "add") -> dict:
    """Example calculator tool demonstrating custom tool creation.

    This is a simple calculator that performs basic arithmetic operations.
    Use this as a template for your own custom tools.

    Args:
        num1: First number
        num2: Second number
        operation: Operation to perform - "add", "subtract", "multiply", "divide" (default: "add")

    Returns:
        Dictionary with:
        - success: Boolean indicating if operation succeeded
        - result: Calculated result
        - operation: Operation performed
        - error: Error message if operation failed
    """
    try:
        if operation == "add":
            result = num1 + num2
        elif operation == "subtract":
            result = num1 - num2
        elif operation == "multiply":
            result = num1 * num2
        elif operation == "divide":
            if num2 == 0:
                return {"success": False, "error": "Cannot divide by zero"}
            result = num1 / num2
        else:
            return {
                "success": False,
                "error": f"Unknown operation: {operation}. Use 'add', 'subtract', 'multiply', or 'divide'",
            }

        return {
            "success": True,
            "result": result,
            "operation": operation,
            "inputs": {"num1": num1, "num2": num2},
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


@tool
def example_text_processor(text: str, operation: str = "uppercase") -> dict:
    """Example text processing tool.

    Demonstrates string manipulation in a custom tool.

    Args:
        text: Input text to process
        operation: Operation - "uppercase", "lowercase", "reverse", "word_count" (default: "uppercase")

    Returns:
        Dictionary with:
        - success: Boolean indicating success
        - result: Processed text or count
        - original_length: Length of original text
        - error: Error message if failed
    """
    try:
        original_length = len(text)

        if operation == "uppercase":
            result = text.upper()
        elif operation == "lowercase":
            result = text.lower()
        elif operation == "reverse":
            result = text[::-1]
        elif operation == "word_count":
            result = len(text.split())
        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}

        return {
            "success": True,
            "result": result,
            "operation": operation,
            "original_length": original_length,
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


# You can add multiple tools in the same file!
# Each function with @tool decorator will be discovered automatically.
