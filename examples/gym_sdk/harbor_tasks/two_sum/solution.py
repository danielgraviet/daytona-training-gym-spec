def two_sum(nums, target):
    # BUG: returns first index twice
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, i]
    return []
