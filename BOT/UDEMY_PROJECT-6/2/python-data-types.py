"""
Python Data Types Demonstration Script

This script explains and demonstrates different types of data types in Python.
1. Numeric Types:
    - int: Integer numbers (e.g., 42, -5)
    - float: Floating-point numbers (e.g., 3.14, -0.001)
    - complex: Complex numbers with real and imaginary parts (e.g., 3+4j)

2. Sequence Types:
    - str: Strings, sequences of Unicode characters
    - list: Mutable, ordered collection of items
    - tuple: Immutable, ordered collection of items
    - range: Immutable sequence of numbers (used in loops)

3. Set Types:
    - set: Unordered collection of unique items
    - frozenset: Immutable version of a set

4. Mapping Types:
    - dict: Key-value pairs (unordered and mutable)

5. Boolean Type:
    - bool: Represents `True` or `False`

6. Binary Types:
    - bytes: Immutable sequence of bytes
    - bytearray: Mutable sequence of bytes
    - memoryview: View of a memory buffer

7. None Type:
    - NoneType: Represents the absence of a value (`None`)

This script uses individual functions to explain and demonstrate each data type.
"""

# Function to demonstrate numeric types
def numeric_types_demo():
    """
    Demonstrates numeric types in Python: int, float, and complex.
    """
    print("Numeric Types:")
    int_num = 42
    float_num = 3.14
    complex_num = 2 + 3j
    print(f"Integer: {int_num}, Type: {type(int_num)}")
    print(f"Float: {float_num}, Type: {type(float_num)}")
    print(f"Complex: {complex_num}, Type: {type(complex_num)}")
    print()

# Function to demonstrate sequence types
def sequence_types_demo():
    """
    Demonstrates sequence types in Python: str, list, and tuple.
    """
    print("Sequence Types:")
    string = "Hello, Python!"
    list_example = [1, 2, 3, "Python", True]
    tuple_example = (1, 2, 3, "Immutable", False)
    print(f"String: {string}, Type: {type(string)}")
    print(f"List: {list_example}, Type: {type(list_example)}")
    print(f"Tuple: {tuple_example}, Type: {type(tuple_example)}")
    print()

# Function to demonstrate set types
def set_types_demo():
    """
    Demonstrates set types in Python: set and frozenset.
    """
    print("Set Types:")
    set_example = {1, 2, 3, "Unique", True}
    frozenset_example = frozenset([1, 2, 3, "Immutable"])
    print(f"Set: {set_example}, Type: {type(set_example)}")
    print(f"Frozenset: {frozenset_example}, Type: {type(frozenset_example)}")
    print()

# Function to demonstrate mapping types
def mapping_types_demo():
    """
    Demonstrates mapping types in Python: dict.
    """
    print("Mapping Types:")
    dict_example = {"key1": "value1", "key2": 42, "key3": True}
    print(f"Dictionary: {dict_example}, Type: {type(dict_example)}")
    print()

# Function to demonstrate boolean type
def boolean_type_demo():
    """
    Demonstrates boolean type in Python: bool.
    """
    print("Boolean Type:")
    true_value = True
    false_value = False
    print(f"True: {true_value}, Type: {type(true_value)}")
    print(f"False: {false_value}, Type: {type(false_value)}")
    print()

# Function to demonstrate binary types
def binary_types_demo():
    """
    Demonstrates binary types in Python: bytes, bytearray, and memoryview.
    """
    print("Binary Types:")
    bytes_example = b"Binary data"
    bytearray_example = bytearray([65, 66, 67])
    memoryview_example = memoryview(bytes_example)
    print(f"Bytes: {bytes_example}, Type: {type(bytes_example)}")
    print(f"Bytearray: {bytearray_example}, Type: {type(bytearray_example)}")
    print(f"Memoryview: {memoryview_example}, Type: {type(memoryview_example)}")
    print()

# Function to demonstrate None type
def none_type_demo():
    """
    Demonstrates None type in Python: NoneType.
    """
    print("None Type:")
    none_value = None
    print(f"None: {none_value}, Type: {type(none_value)}")
    print()

# Main function to call all the demos
def main():
    """
    Main function to execute all data type demonstrations.
    """
    numeric_types_demo()
    sequence_types_demo()
    set_types_demo()
    mapping_types_demo()
    boolean_type_demo()
    binary_types_demo()
    none_type_demo()

# Call the main function
if __name__ == "__main__":
    main()
