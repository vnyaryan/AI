"""
This script demonstrates the 9 different types of arguments in Python functions.

Types of Arguments:
-------------------
1. **Positional Arguments**:
   - Passed based on their position in the function call.
   - Must be provided in the exact order defined in the function.

2. **Keyword Arguments**:
   - Passed with explicit parameter names (e.g., arg=value).
   - Can be provided in any order during the function call.

3. **Default Arguments**:
   - Have a default value defined in the function signature.
   - Become optional when calling the function, as the default value is used if no value is provided.

4. **Variable-Length Positional Arguments (`*args`)**:
   - Collects additional positional arguments into a tuple.
   - Allows functions to accept an arbitrary number of positional arguments.

5. **Variable-Length Keyword Arguments (`**kwargs`)**:
   - Collects additional keyword arguments into a dictionary.
   - Allows functions to accept an arbitrary number of keyword arguments.

6. **Required Arguments**:
   - Must be passed during the function call.
   - Typically positional or explicitly required keyword arguments.

7. **Positional-Only Arguments (`/`)**:
   - Must be passed positionally, without using parameter names.
   - Defined by placing `/` in the function signature.

8. **Keyword-Only Arguments (`*`)**:
   - Must be passed using parameter names.
   - Defined by placing `*` in the function signature.

9. **Mixed Arguments**:
   - Combines positional, default, variable-length, and keyword-only arguments.
   - Demonstrates how different types can work together in one function.

Key Differences:
----------------
- **Positional vs Keyword**: Positional arguments depend on their order, while keyword arguments explicitly use parameter names.
- **Default vs Required**: Default arguments provide a fallback value, while required arguments must always be provided.
- **`*args` vs `**kwargs`**: `*args` collects positional arguments into a tuple, and `**kwargs` collects keyword arguments into a dictionary.
- **Positional-Only vs Keyword-Only**: Positional-only arguments must not use names, while keyword-only arguments must use names.

Each type is illustrated with a separate function and example calls.
"""

# 1. Positional Arguments
def positional_arguments(arg1, arg2):
    """Function demonstrating positional arguments."""
    print(f"Positional Arguments: arg1={arg1}, arg2={arg2}")

# 2. Keyword Arguments
def keyword_arguments(arg1=None, arg2=None):
    """Function demonstrating keyword arguments."""
    print(f"Keyword Arguments: arg1={arg1}, arg2={arg2}")

# 3. Default Arguments
def default_arguments(arg1, arg2="Default Value"):
    """Function demonstrating default arguments."""
    print(f"Default Arguments: arg1={arg1}, arg2={arg2}")

# 4. Variable-Length Positional Arguments (*args)
def variable_positional_arguments(*args):
    """Function demonstrating variable-length positional arguments."""
    print(f"Variable-Length Positional Arguments: {args}")

# 5. Variable-Length Keyword Arguments (**kwargs)
def variable_keyword_arguments(**kwargs):
    """Function demonstrating variable-length keyword arguments."""
    print(f"Variable-Length Keyword Arguments: {kwargs}")

# 6. Required Arguments
def required_arguments(arg1, arg2):
    """Function demonstrating required arguments."""
    print(f"Required Arguments: arg1={arg1}, arg2={arg2}")

# 7. Positional-Only Arguments
def positional_only_arguments(arg1, arg2, /):
    """Function demonstrating positional-only arguments."""
    print(f"Positional-Only Arguments: arg1={arg1}, arg2={arg2}")

# 8. Keyword-Only Arguments
def keyword_only_arguments(*, arg1, arg2):
    """Function demonstrating keyword-only arguments."""
    print(f"Keyword-Only Arguments: arg1={arg1}, arg2={arg2}")

# 9. Mixed Arguments
def mixed_arguments(arg1, /, arg2, *args, kw_only1, **kwargs):
    """Function demonstrating mixed arguments."""
    print(f"Positional-Only Argument: arg1={arg1}")
    print(f"Keyword or Positional Argument: arg2={arg2}")
    print(f"Variable-Length Positional Arguments: {args}")
    print(f"Keyword-Only Argument: kw_only1={kw_only1}")
    print(f"Variable-Length Keyword Arguments: {kwargs}")


# Calling each function to demonstrate the argument types
print("1. Positional Arguments:")
positional_arguments(1, 2)

print("\n2. Keyword Arguments:")
keyword_arguments(arg1="Value1", arg2="Value2")

print("\n3. Default Arguments:")
default_arguments("Provided Value")

print("\n4. Variable-Length Positional Arguments:")
variable_positional_arguments(1, 2, 3, 4, 5)

print("\n5. Variable-Length Keyword Arguments:")
variable_keyword_arguments(key1="Value1", key2="Value2", key3="Value3")

print("\n6. Required Arguments:")
required_arguments("Required1", "Required2")

print("\n7. Positional-Only Arguments:")
positional_only_arguments(1, 2)

print("\n8. Keyword-Only Arguments:")
keyword_only_arguments(arg1="Value1", arg2="Value2")

print("\n9. Mixed Arguments:")
mixed_arguments(1, 2, 3, 4, kw_only1="Value1", kw_only2="Value2", extra1="Extra1", extra2="Extra2")
