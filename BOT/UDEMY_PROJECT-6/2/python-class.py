# Python Script to Demonstrate Classes in Python
# ------------------------------------------------
# This script demonstrates the key concepts of classes in Python with examples:
#
# 1. Defining a Class:
#    A class is a blueprint for creating objects. It defines attributes and methods
#    that the objects created from it will have.
#
# 2. Creating an Object:
#    Objects are instances of a class. They represent individual entities that
#    can store data (attributes) and perform actions (methods) defined in the class.
#
# 3. Understanding the __init__ Method:
#    The __init__ method is a constructor used to initialize attributes of an object
#    at the time of creation. It is automatically called when an object is created.
#
# 4. Adding Methods to a Class:
#    Methods are functions defined inside a class. They allow objects to perform
#    specific actions, often using or modifying their attributes.
#
# 5. Instance Variables vs. Class Variables:
#    Instance variables are specific to an object, while class variables are shared
#    by all objects of a class. This distinction determines how data is managed in a class.
#
# 6. Inheritance:
#    Inheritance allows one class (child class) to inherit attributes and methods
#    from another class (parent class). This promotes code reuse and modularity.
#
# 7. Encapsulation:
#    Encapsulation restricts direct access to some attributes and methods,
#    usually to protect the integrity of an object's data. This is done by marking
#    attributes or methods as private using underscores.
#
# 8. Polymorphism:
#    Polymorphism allows different classes to define methods with the same name,
#    but potentially with different behaviors. This promotes flexibility and generality in code.
#
# 9. Practical Example:
#    Combines the above concepts in a practical scenario, demonstrating how a class can be
#    used to model a real-world object, like a car, with attributes and methods.
# ------------------------------------------------

# 1. Defining a Class
class Person:
    """
    Represents a person with a name and age.

    Attributes:
        name (str): The name of the person.
        age (int): The age of the person.
    """

    def __init__(self, name, age):
        """
        Initializes a Person object with a name and age.
        """
        self.name = name
        self.age = age

    def display_info(self):
        """
        Displays the name and age of the person.
        """
        print(f"Name: {self.name}, Age: {self.age}")


# 2. Creating an Object
print("Point 2: Creating an Object")
person1 = Person("Alice", 25)
person1.display_info()
print("-" * 50)


# 3. Understanding the __init__ Method
class Animal:
    """
    Represents an animal with a species and sound.

    Attributes:
        species (str): The species of the animal.
        sound (str): The sound the animal makes.
    """

    def __init__(self, species, sound):
        """
        Initializes an Animal object with a species and sound.
        """
        self.species = species
        self.sound = sound

    def speak(self):
        """
        Prints the sound the animal makes.
        """
        print(f"The {self.species} says {self.sound}!")


print("Point 3: Understanding the __init__ Method")
cat = Animal("Cat", "Meow")
cat.speak()
print("-" * 50)


# 4. Adding Methods to a Class
class Calculator:
    """
    A simple calculator with addition and subtraction methods.
    """

    def add(self, a, b):
        """
        Adds two numbers.

        Args:
            a (int): The first number.
            b (int): The second number.

        Returns:
            int: The sum of the two numbers.
        """
        return a + b

    def subtract(self, a, b):
        """
        Subtracts the second number from the first number.

        Args:
            a (int): The first number.
            b (int): The second number.

        Returns:
            int: The result of the subtraction.
        """
        return a - b


print("Point 4: Adding Methods to a Class")
calc = Calculator()
print(f"Addition: {calc.add(10, 5)}")
print(f"Subtraction: {calc.subtract(10, 5)}")
print("-" * 50)


# 5. Instance Variables vs. Class Variables
class School:
    """
    Represents a school and its students.

    Attributes:
        school_name (str): The name of the school (class variable).
        student_name (str): The name of the student (instance variable).
    """
    school_name = "Green Valley High"  # Class variable

    def __init__(self, student_name):
        """
        Initializes a School object with a student's name.
        """
        self.student_name = student_name  # Instance variable


print("Point 5: Instance Variables vs. Class Variables")
student1 = School("John")
student2 = School("Emma")
print(f"Student 1: {student1.student_name}, School: {student1.school_name}")
print(f"Student 2: {student2.student_name}, School: {student2.school_name}")
print("-" * 50)


# 6. Inheritance
class Animal:
    """
    Represents a generic animal.
    """

    def speak(self):
        """
        Prints a generic sound for the animal.
        """
        print("Animal speaks")


class Dog(Animal):
    """
    Represents a dog, inheriting from Animal.
    """

    def speak(self):
        """
        Prints the sound specific to a dog.
        """
        print("Dog barks")


print("Point 6: Inheritance")
animal = Animal()
dog = Dog()
animal.speak()
dog.speak()
print("-" * 50)


# 7. Encapsulation
class BankAccount:
    """
    Represents a bank account with a balance.

    Attributes:
        __balance (float): The account balance (private attribute).
    """

    def __init__(self, balance):
        """
        Initializes a BankAccount object with a balance.
        """
        self.__balance = balance  # Private variable

    def deposit(self, amount):
        """
        Deposits an amount into the account.

        Args:
            amount (float): The amount to deposit.
        """
        self.__balance += amount

    def get_balance(self):
        """
        Returns the account balance.

        Returns:
            float: The account balance.
        """
        return self.__balance


print("Point 7: Encapsulation")
account = BankAccount(100)
account.deposit(50)
print(f"Balance: {account.get_balance()}")
print("-" * 50)


# 8. Polymorphism
class Bird:
    """
    Represents a bird.
    """

    def fly(self):
        """
        Prints a generic flying message for a bird.
        """
        print("Bird flies")


class Penguin(Bird):
    """
    Represents a penguin, inheriting from Bird.
    """

    def fly(self):
        """
        Prints a message indicating a penguin cannot fly.
        """
        print("Penguin cannot fly")


print("Point 8: Polymorphism")
bird = Bird()
penguin = Penguin()
bird.fly()
penguin.fly()
print("-" * 50)


# 9. Key Points Recap with a Practical Example
class Car:
    """
    Represents a car with a brand and model.

    Attributes:
        brand (str): The brand of the car.
        model (str): The model of the car.
    """

    def __init__(self, brand, model):
        """
        Initializes a Car object with a brand and model.
        """
        self.brand = brand
        self.model = model

    def start(self):
        """
        Prints a message indicating the car is starting.
        """
        print(f"The {self.brand} {self.model} is starting...")

    def display_info(self):
        """
        Displays the brand and model of the car.
        """
        print(f"Brand: {self.brand}, Model: {self.model}")


print("Point 9: Practical Example with a Car Class")
my_car = Car("Toyota", "Camry")
my_car.start()
my_car.display_info()
print("-" * 50)
