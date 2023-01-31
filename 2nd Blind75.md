# Array & Hashing
[Two Sum](https://leetcode.com/problems/two-sum/description/)
- 01/17：基本知道但是不能bug free写出来 
```Python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        ind_map = {}
        for ind, num in enumerate(nums):
            if target - num in ind_map: #can't use the same index
                return [ind, ind_map[target - num]] #先ind再加就会避免使用同一个数在同一个位置的情况 还可以避免遗漏使用同一个数但是不同位置的情况
            ind_map[num] = ind
```

[Contains Duplicate](https://leetcode.com/problems/contains-duplicate/)
- 01/19: 一遍过
```Python
class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        num_map = {}
        for i in range(len(nums)):
            if nums[i] not in num_map:
                num_map[nums[i]] = i
            else:
                return True
        return False
```

[Valid Anagram](https://leetcode.com/problems/valid-anagram/)
- 01/19: 可以bug free写出来
```Python
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        from collections import Counter
        s_freq = Counter(s)
        for i in range(len(t)):
            if t[i] not in s_freq:
                return False
            if s_freq[t[i]] == 1:
                del s_freq[t[i]]
            else:
                s_freq[t[i]] -= 1
        if len(s_freq) == 0:
            return True
        else:
            return False
```

[Group Anagrams](https://leetcode.com/problems/group-anagrams/description/)
- 01/19: 没想法
```Python
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        # solution 1: hash map, tuple: List
        # t: o(nklogk) n: length of string, k:maximum length of a string, s: o(nk)
        # from collections import defaultdict
        # ana_map = defaultdict(list)
        # for s in strs:
        #     ana_map[tuple(sorted(s))].append(s)
        # return ana_map.values()
        # solution 2: hash map: count of each character, tuple: List 
        # t: o(nk), s:(nk)
        from collections import defaultdict 
        ana_map = defaultdict(list)
        for s in strs:
            count = [0] * 26
            for i in s:
                count[ord(i) - ord('a')] += 1
            ana_map[tuple(count)].append(s)
        return ana_map.values()
```

[Top K Frequent Elements](https://leetcode.com/problems/top-k-frequent-elements/description/)
- 01/23: 基础写法一遍过 但对于counter的most_common function陌生，时间空间复杂度都算错了; 第一次学quick select
```Python
当前的code有误啊
class Codec:
    # diffcult: any possible character
    def encode(self, strs: List[str]) -> str:
        """Encodes a list of strings to a single string.
        """
        res = ""
        for s in strs:
            res = res + str(len(s)) + '#'+ s
        return res

    def decode(self, s):
        res, i = [], 0

        while i < len(s):
            j = i
            while s[j] != "#":
                j += 1
            length = int(s[i:j])
            res.append(s[j + 1 : j + 1 + length])
            i = j + 1 + length
        return res

# Your Codec object will be instantiated and called as such:
# codec = Codec()
# codec.decode(codec.encode(strs))
```

[Product of Array Except Self](https://leetcode.com/problems/product-of-array-except-self/description/)
- 01/25: 基础写法一遍过
```Python
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        # T: o(n), S: o(n)
        # prefix, postfix= [0] * len(nums), [0] * len(nums)
        # res = [0] * len(nums)
        # prefix[0], postfix[-1] = 1, 1
        # for i in range(1, len(nums)):
        #     prefix[i] = prefix[i-1] * nums[i-1]
        # for i in range(len(nums)-2, -1, -1):
        #     postfix[i] = postfix[i+1] * nums[i+1]
        # for i in range(len(nums)):
        #     res[i] = prefix[i] * postfix[i]
        # return res

        # similar as previous, but construct the another on the fly
        # T: o(n), S: o(1)
        length = len(nums)
        answer = [0]*length
        answer[0] = 1
        for i in range(1, length):
            answer[i] = nums[i - 1] * answer[i - 1]
        
        # R contains the product of all the elements to the right
        R = 1;
        for i in reversed(range(length)):           
            # For the index 'i', R would contain the 
            # product of all elements to the right. We update R accordingly
            answer[i] = answer[i] * R
            R *= nums[i]
        return answer
```
[Encode and Decode Strings](https://leetcode.com/problems/encode-and-decode-strings/description/)
```Python
class Codec:
    # diffcult: any possible character
    def encode(self, strs: List[str]) -> str:
        """Encodes a list of strings to a single string.
        """
        res = ""
        for s in strs:
            res = res + str(len(s)) + '#'+ s
        return res

    def decode(self, s):
        res, i = [], 0

        while i < len(s):
            j = i
            while s[j] != "#":
                j += 1
            length = int(s[i:j])
            res.append(s[j + 1 : j + 1 + length])
            i = j + 1 + length
        return res

# Your Codec object will be instantiated and called as such:
# codec = Codec()
# codec.decode(codec.encode(strs))
```

[Longest Consecutive Sequence](https://leetcode.com/problems/longest-consecutive-sequence/description/)
- 01/26 没有想法
```Python
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        num_set = set(nums)
        longest = 0
        for i in range(len(nums)):
            if nums[i]-1 not in num_set: # set object is not subscriptable
                length = 1
                while nums[i] + length in num_set:
                    length += 1
                longest = max(longest, length)
        return longest
```

# Two Pointers
[Valid Palindrome](https://leetcode.com/problems/valid-palindrome/)
- 01/26 不知道怎么对string进行处理
```Python
class Solution:
    def isPalindrome(self, s: str) -> bool:
        l, r = 0, len(s)-1
        while l < r:
            while l < r and not s[l].isalnum():
                l += 1
            while l < r and not s[r].isalnum():
                r -= 1
            if s[l].lower() != s[r].lower():
                return False 
            l += 1
            r -= 1
        return True
```

[3Sum](https://leetcode.com/problems/3sum/)
- 01/26 不能bug free写出来 思路还是断裂的
```Python
from collections import defaultdict

class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        # no duplicated combination
        # sorted iterature the number and a nested for loop using two pointers
        # T: O(n^2 ), S: O(n)
        nums = sorted(nums) #n log n 
        res = []
        for ind in range(len(nums)): #要去重一下 因为不能用有重复的值
            if ind >0 and nums[ind-1] == nums[ind]:
                continue
            left, right = ind+1, len(nums)-1
            target = 0 - nums[ind]
            while left < right:
                if (nums[left] + nums[right]) < target:
                    left += 1 
                elif (nums[left] + nums[right]) > target:
                    right -= 1
                else:
                    res.append([nums[ind], nums[left], nums[right]])
                    left += 1
                    while nums[left-1] == nums[left] and left < right:
                        left += 1 #循环不变量又忘记了一定需要更新他们 要不然退不出循环
        return res
```
[Contain With Most Water](https://leetcode.com/problems/container-with-most-water/description/)
- 01/28 少了一个重要逻辑
```Python
class Solution:
    def maxArea(self, height: List[int]) -> int:
        l, r = 0, len(height) - 1
        max_contain = 0
        while l < r:
            min_height = min(height[l], height[r])
            contain = min_height * (r - l)
            max_contain = max(max_contain, contain)
            if height[r] > height[l]:
                l += 1 #这里错了不是每一次都要同时移动的
            else:
                r -= 1
        return max_contain
```

# Sliding Window
[Best Time to Buy and Sell Stock](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/)
- 01/28: 没有想法
```Python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        # for every element, calculate the different between the min element we have seen
        min_value, max_profit = float('Inf'), 0
        for i in range(len(prices)):
            min_value = min(min_value, prices[i])
            max_profit = max(max_profit, prices[i] - min_value)
        return max_profit
```

[Longest Substring without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/)
- 01/28: 没有想法 这个start用dict储存 并且之后更新到对应位置 很巧妙
```Python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        # T: o(n), S: o(min(m,n)) 
        # s: space for path, n: length of s, m: 24 chars
        start, end = 0, 0 
        path = dict() # char: ind 记录每个字母的位置
        ans = 0
        for end in range(len(s)):
            if s[end] in path:
                start = max(path[s[end]]+1, start)
            path[s[end]] = end
            ans = max(ans, end-start+1)
        return ans
```

[Longest Repeating Character Replacement](https://leetcode.com/problems/longest-repeating-character-replacement/)
- 01/28: 没有想法 这题的边界很需要想清楚 同时都要满足一个区间边界条件
```Python
from collections import defaultdict

class Solution:
    def characterReplacement(self, s: str, k: int) -> int:
        # sliding window, window_size - most freq <= k
        start, end = 0, 1 #左闭右开
        counter = defaultdict(int)
        res = 0
        counter[s[start]] = 1
        while end < len(s):
            length = end - start
            if length - max(counter.values()) <= k:
                counter[s[end]] += 1 
                end += 1   
            else:
                counter[s[start]] -= 1
                start += 1
            length = end - start
            if length - max(counter.values()) <= k:
                res = max(res, length)
        return res
```

[Minimum Window Substring](https://leetcode.com/problems/minimum-window-substring/)
- 01/28: 没有想法 且第一次写
```Python
from collections import Counter

class Solution:
    def minWindow(self, s: str, t: str) -> str:
        if t == "":
            return ""
        target_map = Counter(t)
        need = len(target_map)
        curr_map = {}
        have = 0
        res, res_length = [-1, -1], float('Inf')
        l = 0
        for r in range(len(s)):
            char = s[r]
            curr_map[char] = 1 + curr_map.get(char, 0) # if not in window, return default as 0

            if char in target_map and curr_map[char] == target_map[char]:
                have += 1 
            
            while have == need:
                if (r - l + 1) < res_length:
                    res = [l, r]
                    res_length = r - l + 1
                curr_map[s[l]] -= 1
                if s[l] in target_map and curr_map[s[l]] < target_map[s[l]]:
                    have -= 1
                l += 1
        l, r = res
        return s[l:r+1] if res_length != float('Inf') else ""
```
# Stack
[Valid Parentheses](https://leetcode.com/problems/valid-parentheses/)
- 01/30: 没有想法
```Python
class Solution:
    def isValid(self, s: str) -> bool:
        map = {')':'(', ']':'[', '}':'{'}
        stack = []
        for i in range(len(s)):
            if s[i] not in map.keys():
                stack.append(s[i])
            if s[i] in map.keys() and len(stack) != 0: #不加stack,直接pop
                popitem = stack.pop()
                if popitem != map[s[i]]:
                    return False
            elif s[i] in map.keys() and len(stack) == 0:
                return False
        if len(stack) != 0:
            return False
        return True
```

# Binary Search
[Find Minimum in Rotated Sorted Array](https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/)
- 01/30: 没有想法，binary search模版忘了
```Python
class Solution:
    def findMin(self, nums: List[int]) -> int:
        # binary search, T: O(logn)
        l, r = 0, len(nums)-1
        curr_min = nums[0]
        while l <= r:#为什么要等号呢
            if nums[l] < nums[r]:
                curr_min = min(curr_min, nums[l])
                break
            mid = (r - l)// 2 + l #双杠才是整数
            curr_min = min(curr_min, nums[mid])
            if nums[mid] > nums[r]: #edge case mid can be l
                l = mid + 1
            else:
                r = mid - 1
        return curr_min
```

[Search in Rotated Sorted Array](https://leetcode.com/problems/search-in-rotated-sorted-array/)
- 01/30: 没有想法
```Python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        l, r = 0, len(nums)-1
        while l <= r:
            mid = (r - l)//2 + l
            if nums[mid] == target:
                return mid 
            else:
                if nums[mid] >= nums[l]:
                    if target >= nums[l] and target < nums[mid]:
                        r = mid - 1
                    else:
                        l = mid + 1
                else:
                    if target < nums[l] and target >= nums[mid]:
                        l = mid + 1
                    else:
                        r = mid -1
        return -1
```

# Linked List
[Reverse Linked List](https://leetcode.com/problems/reverse-linked-list/)
- 01/30: 

[Merge Two Sorted Lists](https://leetcode.com/problems/merge-two-sorted-lists/)
- 01/30:

# Trees
# Tries
# Heap / Priority Queue
# Backtracking
# Graphs
# Advanced Graphs
# 1-D Dynamic Programming
# 2-D Dynamic Programming
# Greedy
# Intervals
# Math & Geometry
# Bit Manipulation