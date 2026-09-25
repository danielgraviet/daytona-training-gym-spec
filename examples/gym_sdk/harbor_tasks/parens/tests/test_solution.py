from solution import is_valid

assert is_valid("()") is True
assert is_valid("()[]{}") is True
assert is_valid("(]") is False
print("OK")
