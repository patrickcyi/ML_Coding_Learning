class Solution: random choice max number in index. 

    def __init__(self, nums: List[int]):
        self. nums =nums

    def pick(self, target: int) -> int:
        max_num = float("-inf") 
        max_count= 0
        max_index =-1 
        for i, num in enumerate(self.nums):
            if num > max_num:
                max_count=1
                max_index = i 
            elif num == max_num:
                max_count+=1 
                if random.randint(1, max_count) == max_count:
                    max_index =i 
        return max_index
-----

import random

def reservoir_sampling(stream, k):
    # Initialize the reservoir with the first k elements of the stream
    reservoir = []
    for i, item in enumerate(stream):
        if i < k:
            reservoir.append(item)
        else:
            # Generate a random index from 0 to i (inclusive)
            r = random.randint(0, i)
            if r < k:
                reservoir[r] = item  # Replace an element in the reservoir
    
    return reservoir

# Example stream of data (e.g., large dataset)
stream = range(1, 101)  # Stream of numbers 1 to 100

# Select 5 random samples from the stream
k = 5
samples = reservoir_sampling(stream, k)
print(f"Random {k} samples from the stream: {samples}")


# Your Solution object will be instantiated and called as such:
# obj = Solution(nums)
# param_1 = obj.pick(target)


结论：当你遇到第 i 个元素时，应该有 1/i 的概率选择该元素，1 - 1/i 的概率保持原有的选择。数学证明请看详细题解。

------
from random import choice
from typing import List

class Solution:

    def __init__(self, nums: List[int]):
        self.nums = nums
        self.indices = {}
        for index, num in enumerate(nums):
            if num not in self.indices:
                self.indices[num] = []
            self.indices[num].append(index)

    def pick(self, target: int) -> int:
        # Randomly select an index from the list of indices for the target
        return choice(self.indices[target])