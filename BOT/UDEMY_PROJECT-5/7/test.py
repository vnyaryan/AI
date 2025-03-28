# Import the MessagesState class (assuming it behaves like a dictionary)
from langgraph.graph import MessagesState

# Initialize the MessagesState object
messages_state = MessagesState()

# 1. clear(): Remove all elements from the MessagesState
messages_state["key1"] = "value1"
messages_state["key2"] = "value2"
print("Before clear:", messages_state)
messages_state.clear()
print("After clear:", messages_state)

# 2. copy(): Create a shallow copy
messages_state["key1"] = "value1"
messages_state["key2"] = "value2"
messages_copy = messages_state.copy()
print("Original MessagesState:", messages_state)
print("Copied MessagesState:", messages_copy)

# 3. fromkeys(): Create a new MessagesState from an iterable
new_state = MessagesState.fromkeys(["key3", "key4"], "default_value")
print("New state from keys:", new_state)

# 4. get(): Retrieve a value by key
print("Value of 'key1':", messages_state.get("key1"))
print("Value of 'non_existing_key':", messages_state.get("non_existing_key", "default"))

# 5. items(): Get all key-value pairs as a view
print("Items in MessagesState:", messages_state.items())

# 6. keys(): Get all keys as a view
print("Keys in MessagesState:", messages_state.keys())

# 7. pop(): Remove a key and return its value
removed_value = messages_state.pop("key1", "default_value")
print("Removed value of 'key1':", removed_value)
print("MessagesState after pop:", messages_state)

# 8. popitem(): Remove and return an arbitrary key-value pair
arbitrary_item = messages_state.popitem()
print("Arbitrary item removed:", arbitrary_item)
print("MessagesState after popitem:", messages_state)

# 9. setdefault(): Get value of a key, or set it to a default if it doesn't exist
default_value = messages_state.setdefault("key5", "new_default_value")
print("Value of 'key5':", default_value)
print("MessagesState after setdefault:", messages_state)

# 10. update(): Update MessagesState with a new dictionary
messages_state.update({"key6": "value6", "key7": "value7"})
print("MessagesState after update:", messages_state)

# 11. values(): Get all values as a view
print("Values in MessagesState:", messages_state.values())
