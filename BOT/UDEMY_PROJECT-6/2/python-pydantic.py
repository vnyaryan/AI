"""
Script demonstrating Pydantic features for beginners.

This script explains the following key concepts of Pydantic:
1. Core Concepts: Using BaseModel and type validation.
2. Optional Fields: Defining fields that may or may not be provided.
3. Default Values: Setting default values for model fields.
4. Nested Data Validation: Validating nested objects within models.
5. Handling Validation Errors: Catching and handling invalid input.
6. Practical Example: Validating user registration data for an API.
7. Complex Models: Working with lists and nested structures.
8. Serialization: Converting Pydantic models to dictionaries or JSON.

Each feature is demonstrated in a separate function for clarity.
"""

from typing import Optional, List
from pydantic import BaseModel, ValidationError

# 1. Core Concept: BaseModel and Type Validation
def core_concept_example():
    """
    Demonstrates the core concept of using Pydantic's BaseModel for type validation.
    """
    print("\n--- Core Concept Example ---")
    class User(BaseModel):
        id: int
        name: str
        email: str
        is_active: bool = True

    data = {"id": 1, "name": "Alice", "email": "alice@example.com"}
    user = User(**data)
    print(user)


# 2. Optional Fields
def optional_fields_example():
    """
    Demonstrates the use of optional fields in Pydantic models.
    """
    print("\n--- Optional Fields Example ---")
    class User(BaseModel):
        id: int
        name: str
        age: Optional[int] = None  # Optional field with a default of None

    data_with_age = {"id": 2, "name": "Bob", "age": 30}
    user_with_age = User(**data_with_age)
    print(user_with_age)

    data_without_age = {"id": 3, "name": "Charlie"}
    user_without_age = User(**data_without_age)
    print(user_without_age)


# 3. Default Values
def default_values_example():
    """
    Demonstrates setting default values for fields in Pydantic models.
    """
    print("\n--- Default Values Example ---")
    class Product(BaseModel):
        name: str
        price: float
        in_stock: bool = True  # Default value

    product = Product(name="Laptop", price=999.99)
    print(product)


# 4. Nested Data Validation
def nested_data_example():
    """
    Demonstrates validation of nested objects using Pydantic.
    """
    print("\n--- Nested Data Example ---")
    class Address(BaseModel):
        city: str
        zip_code: str

    class User(BaseModel):
        id: int
        name: str
        address: Address

    data = {
        "id": 4,
        "name": "Dana",
        "address": {"city": "Wonderland", "zip_code": "12345"}
    }
    user = User(**data)
    print(user)


# 5. Handling Validation Errors
def validation_error_example():
    """
    Demonstrates how Pydantic handles validation errors and how to catch them.
    """
    print("\n--- Validation Error Example ---")
    class User(BaseModel):
        id: int
        name: str
        email: str

    try:
        invalid_data = {"id": "not_an_int", "name": "Eve", "email": "eve@example.com"}
        user = User(**invalid_data)
    except ValidationError as e:
        print("Validation Error:", e.json())


# 6. Practical Example: Validating API Data
def api_data_example():
    """
    Demonstrates validating API data using Pydantic models.
    """
    print("\n--- API Data Example ---")
    class RegisterUser(BaseModel):
        username: str
        password: str
        email: str

    data = {"username": "john_doe", "password": "secure123", "email": "john@example.com"}
    user = RegisterUser(**data)
    print(user.dict())  # Convert model to a dictionary


# 7. Complex Model with Lists
def complex_model_example():
    """
    Demonstrates working with lists and nested structures in Pydantic models.
    """
    print("\n--- Complex Model Example ---")
    class Item(BaseModel):
        name: str
        price: float

    class Cart(BaseModel):
        items: List[Item]

    cart_data = {
        "items": [
            {"name": "Book", "price": 12.99},
            {"name": "Pen", "price": 1.99}
        ]
    }
    cart = Cart(**cart_data)
    print(cart)


# 8. Serialization and Dict Conversion
def serialization_example():
    """
    Demonstrates how to serialize Pydantic models into dictionaries or JSON strings.
    """
    print("\n--- Serialization Example ---")
    class User(BaseModel):
        id: int
        name: str
        email: str

    user = User(id=5, name="Fiona", email="fiona@example.com")
    print("As Dictionary:", user.dict())  # Serialize to dictionary
    print("As JSON:", user.json())  # Serialize to JSON


# Main function to call all examples
def main():
    """
    Calls all the example functions to demonstrate Pydantic features.
    """
    core_concept_example()
    optional_fields_example()
    default_values_example()
    nested_data_example()
    validation_error_example()
    api_data_example()
    complex_model_example()
    serialization_example()


if __name__ == "__main__":
    main()
