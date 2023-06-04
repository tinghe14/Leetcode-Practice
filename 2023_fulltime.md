Content
1. [Grokking the Coding Interview: Patterns for Coding Questions at Design Gurus](#Grokking)(PS: 适合参考用语言描述代码步骤以及他们总结的套路)
2. Blind75(1 month, May 21, 2023 - June 10, 2023)
3. Grid150（PS: 刷完之后要尝试总结适合自己的套路）
4. Meta Tag
5. 持续周赛 

# Grokking
<a id='Grokking'></a>

### warmup

### Sliding Window Pattern
- 背景：in many problems dealing with an array (or a LinkedList), we are asked to find or calculate something among all the subarrays(or sublists) of a given size, eg: calculating the sum of every k-element subarray of the given array
- 为什么要有这个trick: 没有trick之前，需要一个nested for loop, 外层starting point一共遍历n-k+1次，内层循环每次都是k次， T: O((n-k+1)*k)。这种方法低效的地方在于每两个相邻的计算都会重复其中k-1个的计算。sliding window的trick可以减少重复的计算使得在T O(N)内完成问题。
- 这个trick是什么: substract the element going out of the window and add the element now being included in the sliding window, in order to reuse the sum from previous subarray. 
- 适用于什么问题: 满足下面所有条件
    - the problem requires repeated computations on a contiguous set of data elements (a subarray or a substring). The size of the window may be fixed or variable, depending on the problem. The repeated computations may be a direct part of the final solution, or they may be intermediate steps building up towards the final solution
    - the computations performed every time the window moves take O(1) time or are a slow-growing function, such as log of a small variable, k where k << n
- 不适用于什么问题: 下面任何一个条件不满足
    - the input data structure doesn't support random access
    - you have to process the entire data without segmentation
- algorithm for the sliding window
~~~
def find_averages_of_subarrays(K, arr):
    result = []
    windowSum, windowStart = 0.0, 0
    for windowEnd in range(len(arr)):
        windowSum += arr[windowEnd]
        # slide the window, no need to slide if we've not hit the required window size 
        if windowEnd >= K - 1:
            result.append(windowSum / K)
            windowSum -= arr[windowStart] # subtract the element going out 
            windowStart += 1 
    return result
~~~
例题一， maximum sum subarray of size k
- Given an array of postive numbers and a positive number 'k', find the maximum sum of any contiguous subarray of size 'k'
~~~
# Best: T O(N), S O(1)
# 做错了 多用了space, 并且边界不正确
def max_sub_array_of_size_k(k, arr):
  # TODO: Write your code here
  if len(arr) <= k:
    return sum(arr)
  window_sum, window_start = 0, 0
  max_sum = 0
  for window_end in range(len(arr)):
    window_sum += arr[window_end]
    if window_end > k-1:
      window_sum -= arr[window_start]
      max_sum = max(max_sum, window_sum)
      window_start += 1
  return max_sum

def main():
  print("Maximum sum of a subarray of size K: " +
      str(max_sub_array_of_size_k(3, [2, 1, 5, 1, 3, 2])))
  print("Maximum sum of a subarray of size K: " +
      str(max_sub_array_of_size_k(2, [2, 3, 4, 1, 5])))

main()   
~~~
例题二，smallest subarray with a greater sum
- given an array of postive integers and a number 'S', find the length of the smallest contiguous subarray whose sum is greater than or equal to 'S'. Return 0 if no such subarray exisits
- hints: the problem follows the sliding window pattern but the sliding window size is not fixed. Firstly, add up elements from the begining of array until their sum become greater than or equal to S, remember the length of this window as the smallest window so far. After that, seems we aim to find the smallest such window. We keep adding one element in the sliding window and start to try to shrink the window from the begining untill is smaller than S again
~~~
# Best: T O(2N) which is asymptotically equivalent to O(N), S O(1)
# 没有转过弯来
def smallest_subarray_sum(s, arr):
  # TODO: Write your code here
  min_len = float('Inf')
  window_start = 0
  window_sum = 0
  for window_end in range(len(arr)):
    window_sum += arr[window_end]
    while window_sum >= s:
      min_len = min(min_len, window_end - window_start + 1)
      window_sum -= arr[window_start]
      window_start += 1
  if min_len == float('Inf'):
    return 0 
  return min_len
~~~

# Blind 75
### Arrays & Hashing
[Contains Duplicate](https://leetcode.com/problems/contains-duplicate/)
- 05/21: 秒

[Valid Anagram](https://leetcode.com/problems/valid-anagram/)
- 05/21: 想了一会，秒；但space complexity有错不是O(n)最大的都是固定O(26)所以是O(1)

[Two Sum](https://leetcode.com/problems/two-sum/description/)
- 05/21: 脑海中记得最优解法是one-pass harsh table 也写出来的了正确的形式 但是顺序错了 但是两个corner cases均没有考虑到（1）不能是同一个位置上的数字（2）他们是同一个数字 但是在不同位置 再多体会一下
```
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        map = dict()
        for ind, num in enumerate(nums):
            if target - num in map:
                return [ind, map[(target-num)]]
            map[num] = ind 
```

[Group Anagrams](https://leetcode.com/problems/group-anagrams/description/)
- 05/21: 知识点错误（1）忘记了Counter他是首字母大写的function并且反悔的是counter type(2)什么类型的可以做 dictionary的key:must be immutable type, eg: you can use an integer, float, string, boolean, tuple but not list nor another dictionary because they are mutable (2)dictionary.values()可以直接返回list of list的value 
```
from collections import defaultdict
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        map = defaultdict(list)
        for word in strs:
            count = [0]*26
            for i in word:
                count[ord(i) - ord('a')] += 1
            map[tuple(count)].append(word)
        return map.values()
```

[Top K Frequent Elements](https://leetcode.com/problems/top-k-frequent-elements/description/)
- 05/21: 没想出来，之后再看这里有quick sort，整理sorting时一起 (整理在github)
'''
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        '''
        bucket sort, T: O(n), S: O(n)
        '''
        if k == len(nums):
            return nums
        from collections import defaultdict
        map = defaultdict(int)
        # create hasmap of containing the count of each number
        for num in nums:
            map[num] += 1
        # create a list of lists to store the counts of in the correct order
        bucket_list = [[] for i in range(len(nums))]
        # store the numbers in the bucket list using the count as the index
        for num, count in map.items():
            bucket_list[count-1].append(num) 
        # unpack the bucket list into a new list
        result = []
        for bucket in bucket_list:
            result.extend(bucket)
        # fetch the top k elements by iterating through the list in reverse
        topk = []
        for i in range(-1, -(k+1), -1): 
            topk.append(result[i])
        return topk
'''

[Product of Array Except Self](https://leetcode.com/problems/product-of-array-except-self/description/)
- 05/21: 错得很离谱 知识性错误（1）range()倒过来一个strs怎么写 range(len(strs)-1, -1, -1)开头就要-1因为时能够取到这个位置的 (不要从负数开始算 很奇怪这样)
'''
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        '''
        generate 2 lists
        first from left to right accumulatly multiple the items before current index
        second from right to left accumulately multiple the items after current index
        T (n)
        S (n)
        '''
        # left_product, right_product = [0]*len(nums), [0]*len(nums)
        # left_product[0], right_product[-1] = 1, 1
        # for i in range(1, len(nums)):
        #     temp = left_product[i-1] * nums[i-1]
        #     left_product[i] = temp
        # for i in range(len(nums)-2, -1, -1):
        #     temp = right_product[i+1] * nums[i+1]
        #     right_product[i] = temp
        # result = [0]*len(nums)
        # for i in range(len(nums)):
        #     result[i] = left_product[i] * right_product[i]
        # return result
        '''
        S(1)
        '''
        result = [0]*len(nums)
        result[0] = 1
        for i in range(1, len(nums)):
            result[i] = nums[i-1] * result[i-1]
        R = 1
        for i in range(len(nums)-1, -1, -1):
            result[i] = result[i] * R
            R *= nums[i]
        return result
'''

[Encode and Decode Strings](https://leetcode.com/problems/encode-and-decode-strings/description/)
- 05/24: 没想法 但是应该要会实现基础的办法
'''
class Codec:
    def encode(self, strs: List[str]) -> str:
        """Encodes a list of strings to a single string.
        """
        result = ''
        for str in strs:
            result += str + len(str) + '#' + str
        return result

    def decode(self, s: str) -> List[str]:
        """Decodes a single string to a list of strings.
        """
        result, i = [], 0
        while i < len(s):
            j = i
            while s[j] != "#":
                j += 1
            length = int(s[i:j])
            res.append(s[j + 1 : j + 1 + length])
            i = j + 1 + length
        return res
'''

[Longest Consecutive Sequence](https://leetcode.com/problems/longest-consecutive-sequence/description/)
- 05/24: 不知道怎么o(n)完成, 但是可以写出o(nlogn)的
'''
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        # hashset and intelligent sequence checking
        # hashset allow lookup in O(1)
        # logic: if(!num_set.contains(num-1)).
        # That means, for example, 6,5,4,3,2,1 input, 
        # only the value 1 is valid for the loop(all other values have its value - 1 in the set), that is O(n).
        num_set = set(nums)
        max_length = 0
        for num in nums:
            if num-1 not in num_set:
                length = 1
                start_p = num 
                while start_p+1 in num_set:
                    length += 1
                    start_p += 1
            max_length = max(max_length, length)
        return max_length
'''

### Two Pointers
[Valid Palindrome](https://leetcode.com/problems/valid-palindrome/)
- 05/24: 有些bug, 知识点(1)whether is alphanumeric: isalnum() (2) while的时候小心出界
'''
class Solution:
    def isPalindrome(self, s: str) -> bool:
        # isalnum() method returns True if all the characters are alphanumeric, meaning alphabet letter (a-z) and numbers (0-9)
        if len(s) == 1: return True
        left, right = 0, len(s)-1
        while left < right:
            while left < right and not s[left].isalnum():
                left += 1 
            while left < right and not s[right].isalnum():
                right -= 1
            if s[left].lower() != s[right].lower():
                return False 
            left += 1
            right -= 1
        return True 
'''

[3Sum](https://leetcode.com/problems/3sum/)
- 05/25: nums.sort()才是in-place操作！没有掌握这道题的多、种解法
'''
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        '''
        sort first take o(nlogn)
        then use two pointer towards each other to find the combination
        # O(n^2) O(n)
        '''
        final_result = []
        nums.sort()
        if nums[0] > 0 : return final_result
        for i in range(len(nums)):
            if i > 1 and nums[i] == nums[i-1]:
                continue
            left, right = i+1, len(nums)-1
            while left < right:
                if left < right and nums[left] + nums[right] == 0 - nums[i]:
                    final_result.append([nums[i], nums[left], nums[right]])
                    left += 1 # 都要挪一个位置 是一个增 一个减的关系 因此可以用two pointer
                    right -= 1
                    while left < right and nums[left] == nums[left+1]: #第二层循环也要控制不是同一个数
                        left += 1
                elif left < right and nums[left] + nums[right] > 0 - nums[i]:
                    right -= 1 
                elif left < right and nums[left] + nums[right] < 0 - nums[i]: 
                    left += 1 
        return final_result
'''

[Contain With Most Water](https://leetcode.com/problems/container-with-most-water/description/)
- 05/25: （1）忘记trick了不知道怎么写（2）基础解法也不会
'''
class Solution:
    def maxArea(self, height: List[int]) -> int:
        '''
        basic method: O(n^2)
        one pass of the list, move the shorter one
        '''
        max_area = 0 
        left, right = 0, len(height)-1
        while left < right:
            width = right - left
            if height[left] < height[right]: 
                area = height[left] * width
                max_area = max(max_area, area) 
                left += 1 
            else:
                area = height[right] * width
                max_area = max(max_area, area)
                right -= 1
        return max_area
'''

# Sliding Window
[Best Time to Buy and Sell Stock](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/)
- 05/25: 忘记了trick
'''
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        '''
        一次便利，如果我再i点卖出，那么我一定是挑选(0,n-1)中最小的买入
        '''
        min_value = prices[0]
        max_profit = 0
        for i in range(1, len(prices)):
            if prices[i] < min_value:
                min_value = prices[i]
            else:
                max_profit = max(prices[i]-min_value, max_profit)
        return max_profit
'''

[Longest Substring without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/)
- 06/03: window list要pop开头的元素 让我感觉很奇怪 而且这个效率应该不是O(1)吧->it is O(1) to pop the last element of a Python list, and O(N) to pop an arbitrary element (since the whole rest of the list has to be shifted). Here's a great article on how Python lists are stored and manipulated: http://effbot.org/zone/python-list.htm
'''
# 尝试看了答案后自己写brute force版本 写出来了 但是边界条件没有留意 如果这个序列长度大小为零 这个需要先考虑 因为之后都会assume不为零
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        '''
        [naive way will be using nested for loop to search for the 
        longest substring which meets requirements, which will take O(N^2)] -> [enumerate all substrings 
        of a given string, we enumerate the start and end indices of them. O(N^3) need to check duplicates]
        [however, we can save some time complexity by using sliding window techniques] -> [since previously,
        we repeatedly check a substring to see if it has duplicate character, but it is unnecessary]
        starting from the index 0 as the start of the window, move the end of the window one step further
        untill it has repeating characters. When it happens, we move the starting pointer of the window
        until it meets the requirement again
        '''
        # brute force
        if len(s) == 0: return 0
        max_len = 1
        for start_point in range(len(s)-1):
            for end_point in range(start_point, len(s)):
                sub_s = list(s[start_point:end_point])
                incoming = s[end_point]
                if not self.contain_duplicate(sub_s, incoming):
                    max_len = max(max_len, len(sub_s)+1)
                    sub_s.append(incoming)
                else:
                    break
        return max_len
    def contain_duplicate(self, sub_s, incoming):
        if incoming in sub_s: # x in s: take O(n) time 
            return True
        else:
            return False

class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        # sliding window
        # previous: check s in list_s O(n)
        # now: check k in dict_s O(1)
        # T: O(n)->worse case O(2n), S O(n)
        if len(s) == 0: return 0
        chars = Counter()
        max_len = 1
        window_start = 0
        # 这里不用mantain一个window有s_p, e_p也就都知道了
        for window_end in range(len(s)):
            chars[s[window_end]] += 1 #这里卡住了 反正每一步都加，加完之后比较有没有符合条件
            while chars[s[window_end]] > 1:
                chars[s[window_start]] -= 1
                window_start += 1
            max_len = max(max_len, window_end-window_start+1)
        return max_len

# 第三种优化没有看懂
'''

[Longest Repeating Character Replacement](https://leetcode.com/problems/longest-repeating-character-replacement/)
- 06/03：不会，brute force的都不会

# Weekly Contest
1. 06/04, #348: 1st bug free, 2nd 2 debug, rank 13178/23114
    - [2717 Semi-Ordered Permutation](https://leetcode.com/problems/semi-ordered-permutation/)
    ~~~
    # 主要思考错了一个地方，当计算swap次数时，如果min和max位置交叉
    # min被移到头部，这个过程中max也只是被迫移动一次而已
    def semiOrderedPermutation(self, num: List[int]) -> int:
        if nums[0] == 1 and nums[len(nums)-1] == len(nums):
            return 0 
        op = 0
        min_idx = nums.index(1)
        max_idx = nums.index(len(nums))
        if min_idx < max_idx:
            op = min_idx + (len(nums)-1-max_idx)
        if min_idx > max_idx:
            op = min_idx + (len(nums)-1-max_idx)-1
        return op
    ~~~
    - [2718 Sum of Matrix After Queries](https://leetcode.com/problems/sum-of-matrix-after-queries/)
    ~~~
    # 主要错误的地方 并不用记录每一个位置被填充成了什么 只要知道有几个数被赋值成对应的数
    # 整体的思路都有些不对
    def matrixSumQueries(self, n: int, queries: List[List[int]]) -> int:
        rowSeenCount, ColSeenCount, total = 0, 0, 0
        rowSeen, colSeen = [False]*n, [False]*n
        for qi in range(len(queries)-1, -1, -1):
            typei, index, val = queries[qi][0], queries[qi][1], queries[qi][2]
            if typei == 0 and not rowSeen[index]:
                rowSeenCount += 1 
                rowSeen[index] = True
                total += (n - colSeenCount) * val
            if typei == 1 and not colSeen[index]:
                colSeenCount += 1
                colSeen[index] = True
                total += (n - rowSeenCount) * val
        return total
    ~~~