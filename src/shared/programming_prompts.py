BASE_PROGRAMMING_PROMPTS = [
    "Write a Python function that takes a list of integers and returns the sum of all even numbers.",
    "Complete this code snippet: def is_palindrome(s):",
    "Explain in one sentence what this line does: result = [x*2 for x in numbers if x > 10]",
    "Fix the bug in this code: for i in range(len(lst)): lst[i] = lst[i] + 1",
    "What does the following Python code print? print(2 ** 3 ** 2)",
    "Write a one-line list comprehension that squares all numbers from 1 to 20.",
    "Convert this loop into a list comprehension: result = []; for x in data: if x % 2 == 0: result.append(x)",
    "What is the output of: print('Hello' + 5)",
    "Complete the function: def factorial(n):",
    "Write a simple Python script that reads a number from input and prints whether it is prime.",
]

INTERMEDIATE_PROGRAMMING_PROMPTS = [
    "Implement a binary search function that returns the index of the target or -1 if not found.",
    "Write a Python class for a Stack with push, pop, and is_empty methods.",
    "Explain how this recursive function works: def fib(n): return n if n <= 1 else fib(n-1) + fib(n-2)",
    "Optimize this O(n^2) function to run in better time complexity: def find_pairs(arr, target):",
    "What will this code output and why? def outer(): x=10; def inner(): print(x); inner(); outer()",
    "Implement a decorator that measures and prints the execution time of any function.",
    "Write a function that flattens a nested list of arbitrary depth.",
    "Complete this merge sort implementation: def merge_sort(arr):",
    "Explain the difference between shallow and deep copy with code examples.",
    "Write Python code using itertools to generate all permutations of [1,2,3].",
]

ADVANCED_PROGRAMMING_PROMPTS = [
    "Implement a lock-free thread-safe singleton pattern in Python using metaclasses.",
    "Write an async function with asyncio that fetches data from 5 URLs concurrently with a 2-second timeout.",
    "Implement a custom memory-efficient generator that yields prime numbers up to n using the Sieve of Eratosthenes.",
    "Explain and implement the difference between __new__ and __init__ with a metaclass example.",
    "Write a Python metaclass that automatically registers all subclasses in a global registry.",
    "Implement a context manager that acquires a database connection and handles rollback on exception.",
    "Write code that uses weakref to implement a cache that automatically evicts least-recently-used items.",
    "Implement a Python descriptor that validates attribute types at runtime.",
    "Write a complete implementation of a Trie (prefix tree) with insert, search, and starts_with methods.",
    "Explain how Python's Global Interpreter Lock affects multi-threading and show a workaround using multiprocessing.",
]