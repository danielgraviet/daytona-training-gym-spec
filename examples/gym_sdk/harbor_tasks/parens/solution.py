def is_valid(s: str) -> bool:
    # BUG: never pops
    stack = []
    pairs = {")": "(", "]": "[", "}": "{"}
    for ch in s:
        if ch in "([{":
            stack.append(ch)
        elif ch in pairs:
            stack.append(ch)
    return not stack
