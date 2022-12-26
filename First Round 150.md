Dec 11, 2022 从基础开始练习 之后再按代码分享录的顺序学tag 基本上是按照cspiration的顺序

# Array and String
- An array is a collection of items. The items can be anything and they are stored in contiguous memory locations
- Common Techniques:
- Inserting Items into an Array: (1) duplicate zeros, (2) merged sorted array
- Deleting Items from an Array:
- Searching for Items in an Array:
- In-Place Operations:
- 

[Duplicate Zeros](https://leetcode.com/explore/learn/card/fun-with-arrays/525/inserting-items-into-an-array/3245/)
- Tag: Array
- Time: 12-12: X
- Logic of Solution: 
- 一题多解:

[Merge Sorted Array](https://leetcode.com/problems/merge-sorted-array/)
- Tag: Array
- Time: 12-12: X
- Logic of Solution: 
- [x]一题多解:
``` Python
class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        # three pointers, 2 read, 1 write starting from the end
        # T:O(n+m), S:O(1)
        p2, p3 = n-1, m-1
        for p1 in range(n+m-1, -1, -1): #end_pointer, -start_pointer, -1
            if p2 >= 0 and p3 >=0:
                if nums1[p3] >= nums2[p2]:
                    nums1[p1] = nums1[p3]
                    p3 -= 1
                else:
                    nums1[p1] = nums2[p2]
                    p2 -= 1
            elif p2 >= 0 and p3 <0:
                nums1[p1] = nums2[p2]
                p2 -= 1
            else:
                nums1[p1] = nums1[p3]
                p3 -= 1
        return

        
    # def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
    #     """
    #     Do not return anything, modify nums1 in-place instead.
    #     """
    #     # three pointers, 2 read, 1 write starting from the front
    #     # T:O(N+M), S:O(M)
    #     nums3 = nums1[:m]
    #     p2, p3 = 0, 0
    #     p1 = 0
    #     while p1 <= n+m-1:
    #         print(p1, p2, p3)                                       #惰性求值
    #         if (p3 >= m) or (p2 <= n-1 and nums2[p2] < nums3[p3]): #是从左到右判断的 如果前面以及爆了也不会做or后面的判断
    #             nums1[p1] = nums2[p2] 
    #             p2 += 1
    #         else:
    #             nums1[p1] = nums3[p3]
    #             p3 += 1
    #         p1 += 1
    #     return 
```

[Remove Element](https://leetcode.com/problems/remove-element/)
- Tag: Array
- Time: 12-12: X
- Logic of Solution: slow前面是符合条件的
- [x]一题多解:
```Python
class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        # fast: move every iteration
        # slow: only move when it meets requirement
        slow = 0
        for fast in range(len(nums)):
            if nums[fast] == val:
                fast += 1
            else:
                nums[slow] = nums[fast]
                slow += 1
        return slow

    # two pointers - when elements to remove are rare
    # swap the current item to the end
    # T: O(n), S: O(1)
    # consider for example [1,2,3,5,4], val=4; the previous algo will do unnecessary 
    # copy operation of the first four elements
    # another example is nums = [4, 1, 2, 3, 5], val=4; it do unnecessary to move elements
    # [1,2,3,5] one step left as the problem description mentions that the order of elements could be changed
    # two pointers - when elements to remove are rare
    # when we encounter nums[i] = val, we can swap the current element out with the last element and dispose the last one
    class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        start, end = 0, len(nums)-1
        while start <= end:
            if nums[start] == val:
                nums[start], nums[end] = nums[end], nums[start]
                end -= 1
            else:
                start += 1
        return start

```

[26 Remove Duplicates from Sorted Array](https://leetcode.com/problems/remove-duplicates-from-sorted-array/)
- Tag: Array, Two Pointer
- Time: 12-11 X 
- Logic of Solution: 
- [x]一题多解:
```Python
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        # fast : iterate everytime
        # slow : next ind to add item
        slow, fast = 1, 1
        while fast < len(nums):
            if nums[fast] == nums[slow-1]: # remove duplicates 所以有个自省的操作
                fast += 1
            else:
                nums[slow] = nums[fast]
                slow += 1
                fast += 1
        return slow
```

[80 Remove Duplicates from Sorted Array II](https://leetcode.com/problems/remove-duplicates-from-sorted-array-ii/description/)
- Tag: Array, Two Pointer
- Time: 12-12: x ??
- Logic of Solution: 
- [x]一题多解: 已经实现了最优解 就不用写次优解了
``` Python
# 1 points: overwite element
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        # fast: iterate everytime
        # slow: index of new array
        slow, fast = 2, 2
        while fast < len(nums):
            if nums[fast] == nums[slow-1] and nums[fast] == nums[slow-2]:
                fast += 1
            else:
                nums[slow] = nums[fast]
                slow += 1
                fast += 1
        return slow
```

[Find the Town Judge](https://leetcode.com/problems/find-the-town-judge/)
- Tag: Array, Hash Table, Graph
- Time: 12-12: O
- Logic of Solution: 算in-degree和out-degree
- 一题多解:
```Python
class Solution:
    def findJudge(self, n: int, trust: List[List[int]]) -> int:
        # T: O(E), S: O(n)
        from collections import defaultdict
        if n == 1 and len(trust) == 0: return 1
        in_degree, out_degree = defaultdict(int), defaultdict(int)
        for edge in trust:
            in_degree[edge[1]] += 1
            out_degree[edge[0]] += 1
        for person in in_degree.keys():
            if in_degree[person] == n - 1 and out_degree[person] == 0:
                return person 
        return -1
```
[277 Find the Celebirty](https://leetcode.com/problems/find-the-celebrity/description/)
- Tag: 
- Time: 12-12: X
- Logic of Solution: 如上 但是只知道n所以vertice为key
- [x]一题多解:
```Python
# Time: o(n^2+n)
# The knows API is already defined for you.
# return a bool, whether a knows b
# def knows(a: int, b: int) -> bool:
from collections import defaultdict 
class Solution:
    def findCelebrity(self, n: int) -> int:
        in_degree, out_degree = defaultdict(int), defaultdict(int)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if knows(i, j):
                    in_degree[j] += 1
                    out_degree[i] += 1                     
        for person in in_degree.keys():
            if in_degree[person] == n-1 and out_degree[person] == 0:
                return person
        return -1
# o(n)
class Solution:
    def findCelebrity(self, n: int) -> int:
        potential_c = 0
        # reutrn the ball to the potential celebirty
        for i in range(n):
            if knows(potential_c, i):
                potential_c = i
        # determine whether potential celebirty is valid 
        for j in range(n):
            if j == potential_c:
                continue 
            if knows(potential_c, j) or not knows(j, potential_c):
                return -1 
        return potential_c
```
[189 Rotate Array](https://leetcode.com/problems/rotate-array/description/)
- Tag: Array, Math, Two Pointer
- Time: 12-12: X
- Logic of Solution: 
- [x]一题多解:
```Python
class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        # new_array = [0]*len(nums)
        # temp = nums[:k]
        # new_array[:k] = nums[-1*k:] 
        # new_array[k:] = temp + nums[k:-1*k]
        # nums[:] = new_array[:]

        # k = k % len(nums)
        # nums[:] = nums[-k:] + nums[:(len(nums) - k)]

        # T&S: O(N)
        # copynum = nums[:]
        # for i in range(len(nums)):
        #     nums[i] = copynum[(i - k) % len(nums)]

        # T: O(n) S: O(1)
        k = k % len(nums)
        self.reverse(nums, 0, len(nums)-1)
        self.reverse(nums, 0, k-1)
        self.reverse(nums, k, len(nums)-1)

    def reverse(self, nums, start, end):
        # start: inclusive, end: inclusive
        while start < end:
            nums[end], nums[start] = nums[start], nums[end]
            end -= 1
            start += 1
```

[Find All Numbers Disappered in an Array](https://leetcode.com/problems/find-all-numbers-disappeared-in-an-array/description/)
- Tag: Array, Hash Table
- Time: 12-13
- Logic of Solution: in-place修改的用到了小技巧 mark对应位置的为负数 这样也没有修改它本身的值 剩下为正的位置就是我们缺失的数
- [x]一题多解:
```Python
class Solution:
    def findDisappearedNumbers(self, nums: List[int]) -> List[int]:
    # T:O(N), S:O(1)
    # given a 1~n n list of integer, highlight the position as negative when seeing corresponding number 
    # 2,3,3,3 2,-3,-3,3 -> 1, 4
        result = []
        for num in nums:
            # 注意这后面的判断 只有大于零才会做修改
            if num > 0 and nums[num-1] > 0:
                nums[num-1] = -nums[num-1]
            if num < 0 and nums[-num-1] > 0:
                nums[-num-1] = -nums[-num-1]
        for i in range(len(nums)):
            if nums[i] > 0:
                result.append(i+1)
        return result
    #T&S: O(N) 
    # def findDisappearedNumbers(self, nums: List[int]) -> List[int]:
    #     # hashset
    #     result = []
    #     hashset = set(nums)
    #     for num in range(1, len(nums)+1):
    #         if num not in hashset:
    #             result.append(num)
    #     return result
```

[Find the Duplicate Number](https://leetcode.com/problems/find-the-duplicate-number/description/)
- Tag: 
- Time: 12-11
- Logic of Solution: 
- []一题多解:
```Python
# 2.bit manipulation: XOR
# 1.binary search
# 3.two pointer: Floyd's Algorithm

[242 Valid Anagram](https://leetcode.com/problems/valid-anagram/description/)
- Tag: Hash Table, String, Sorting
- Time: 12-19
- Logic of Solution: 
- 一题多解:
```Python
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        # T: O(n), S: O(1)
        from collections import Counter
        s_count = Counter(s)
        t_count = Counter(t)
        return s_count == t_count
```

[1 Two Sum](https://leetcode.com/problems/two-sum/description/)
- Tag: Array, Hash Table
- Time: 12-19
- Logic of Solution: 
- []一题多解:
```Python
# one-pass hash map 
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        # one-pass hashmap
        dic = dict() #不能用default dict因为要不然永远都在 而且为零
        for ind, num in enumerate(nums):
            if (target-num) in dic.keys():
                return [ind, dic[target-num]]
            dic[num] = ind
```

[49 Group Anagrams](https://leetcode.com/problems/group-anagrams/description/)
- Tag: Array, Hash Table, String, Sorting
- Time: 12-19
- Logic of Solution: 
- []一题多解:
```Python
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        # T&S: O(NK), N: length of strs, K: maximum length of a word
        from collections import defaultdict
        d = defaultdict(list)
        for i in range(len(strs)):
            count = [0]*26
            for c in strs[i]:
                count[ord(c) - ord('a')] += 1 #unicode valye
            d[tuple(count)].append(strs[i])
        return d.values()
```

[347 Top K Frequent Elements](https://leetcode.com/problems/top-k-frequent-elements/description/)
- Tag: Array, Hash Table, Divide and Conquer, Sorting, Heap (Priority Queue), Bucket Sort, Counting, Quickselect
- Time: 12-19
- Logic of Solution: 有更优的解法 但是我没看？？
- [x]一题多解:
```Python
class Solution:
    # T: O: (N log K), n: nums, k: keep k in heap
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        from collections import Counter
        import heapq as hp 
        h = []
        c = Counter(nums)
        for (num, freq) in c.items():
            hp.heappush(h,(freq, num))
            if len(h) > k:
                hp.heappop(h)
        result = [0]*k
        for i in range(k-1, -1, -1):
            result[i] = hp.heappop(h)[1]
        return result

    # def topKFrequent(self, nums: List[int], k: int) -> List[int]:
    #     # Counter 没有k这个关键词 用的是n 然后输出的是list of tuple, each tuple is item with count
    #     # T for most_common: O(n log n)
    #     from collections import Counter
    #     c = Counter(nums)
    #     return [i for (i,j) in c.most_common(n=k)]
```

[128 Longest Consecutive Sequence](https://leetcode.com/problems/longest-consecutive-sequence/)
- Tag: Array, Hash Table, Union Find(x)
- Time: 12-21
- Logic of Solution: 
- [x]一题多解:
```Python
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        # optimal: set: O(1) to check
        # T: O(n) at most visit every num twice, S: O(n)
        s = set(nums)
        temp_max = 0
        for i in range(len(nums)):
            if nums[i]-1 not in s:
                count = 1
                while nums[i]+count in s: # check wehther following items in the set
                    count += 1 
                temp_max = max(temp_max, count)
        return temp_max

    # def longestConsecutive(self, nums: List[int]) -> int:
    #     # brute force: sort, then count the longest consectuive subarray
    #     # T (n log n), S: O(n)
    #     nums.sort() #increasing
    #     if len(nums) == 0: return 0
    #     count, temp_max = 1, 1
    #     for i in range(len(nums)-1):
    #         if nums[i+1] == nums[i] + 1: 
    #             count += 1
    #             temp_max = max(temp_max, count)
    #         # edge case没想到 如果是重复的数字话 这里就会断掉
    #         elif nums[i+1] == nums[i]:
    #             continue
    #         else:
    #             count = 1
    #     return temp_max     
```

[]()
- Tag: 
- Time: 12-21
- Logic of Solution: 
- 一题多解:


## Bit Manipulation: XOR 
[389 Find the Difference](https://leetcode.com/problems/find-the-difference/description/)
- Tag: Hash Table, String, Bit Manipulation, Sorting
- Time: 12-11: X
- Logic of Solution: bitwise XOR operation on all the elements: will eliminate the alike and only leave the odd duckling
- []一题多解:
```Python
# XOR 
class Solution:
    # T: O(n), S: O(1)
    def findTheDifference(self, s: str, t: str) -> str:
        xor = 0
        for char in s:
            # built-in funciton: ord() convert char into int
            xor ^= ord(char) 
        for char in t:
            xor ^= ord(char)
            # built-in function: chr() convert int into char 
        return chr(xor) 
```

[318 Maximum Product of Word Lengths](https://leetcode.com/problems/maximum-product-of-word-lengths/description/)
- Tag: Array, String, Bit Manipulation
- Time: 12-11: X
- Logic of Solution: ->这题是bitmastk 先跳过
- 一题多解:

[41 First Missing Positive](https://leetcode.com/problems/first-missing-positive/description/)
- Tag: Array, Hash Table
- Time: 12-14 X
- Logic of Solution: 
- 一题多解:

## Binary Search


## Two Pointer
[88 Merge Sorted Array](https://leetcode.com/problems/merge-sorted-array/submissions/859992440/)
- Tag: Array, Two Pointer, Sorting
- Time: 12-14:X
- Logic of Solution: 官方给的提示 inplace array修改推荐从后往前算
- [X]一题多解:
```Python
class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        # three pointers, 2 read, 1 write
        # T:O(n+m), S:O(1)
        p2, p3 = n-1, m-1
        for p1 in range(n+m-1, -1, -1): #end_pointer, -start_pointer, -1
            if p2 >= 0 and p3 >=0:
                if nums1[p3] >= nums2[p2]:
                    nums1[p1] = nums1[p3]
                    p3 -= 1
                else:
                    nums1[p1] = nums2[p2]
                    p2 -= 1
            elif p2 >= 0 and p3 <0:
                nums1[p1] = nums2[p2]
                p2 -= 1
            else:
                nums1[p1] = nums1[p3]
                p3 -= 1
        return

        
    # def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
    #     """
    #     Do not return anything, modify nums1 in-place instead.
    #     """
    #     # three pointers, 2 read, 1 write
    #     # T:O(N+M), S:O(M)
    #     nums3 = nums1[:m]
    #     p2, p3 = 0, 0
    #     p1 = 0
    #     while p1 <= n+m-1:
    #         print(p1, p2, p3)                                       #惰性求值
    #         if (p3 >= m) or (p2 <= n-1 and nums2[p2] < nums3[p3]): #是从左到右判断的 如果前面以及爆了也不会做or后面的判断
    #             nums1[p1] = nums2[p2] 
    #             p2 += 1
    #         else:
    #             nums1[p1] = nums3[p3]
    #             p3 += 1
    #         p1 += 1
    #     return 
```

[75 Sort Colors](https://leetcode.com/problems/sort-colors/description/)
- Tag: Array, Two Pointers, Sorting
- Time: 12-14 X
- Logic of Solution: 
- 一题多解:
```Python
class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        # dutch national flag problem
        # 3 pointer: left: the rightmost ind of 0, curr (operation pointer), right: the leftmost ind of 2
        # T: O(n), S:O(1)
        left, curr, right = 0, 0, len(nums)-1
        while curr <= right:
            if nums[curr] == 0:
                nums[left], nums[curr] = nums[curr], nums[left]
                left += 1
                curr += 1
            elif nums[curr] == 2:
                nums[right], nums[curr] = nums[curr], nums[right]
                right -= 1
            else:
                curr += 1
    # T: o(n), S: o(n)
    # def sortColors(self, nums: List[int]) -> None:
    #     """
    #     Do not return anything, modify nums in-place instead.
    #     """
    #     from collections import defaultdict 
    #     colordict = defaultdict(int)
    #     for num in nums:
    #         colordict[num] += 1
    #     nums[:] = [0]*colordict[0] + [1]*colordict[1] + [2]*colordict[2]
```

[283 Move Zeros](https://leetcode.com/problems/move-zeroes/description/)
- Tag: Array, Two Pointers
- Time: 12-15: X
- Logic of Solution: 同向双指针 [0:left]处理完的数据 [left:curr] 正在处理的 [curr:end] 未处理的
- [x]一题多解:
```Python
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        # 我好像一直比较难理解的是 python这个交换的作用
        # 头脑里也要保证对双指针我自己的定义
        # 同向双指针 [0:left]处理完的数据 [left:curr] 正在处理的 [curr:end] 未处理的
        # 好好想想 这是个常见的同向双指针套路
        # https://leetcode.com/problems/move-zeroes/solutions/979820/two-pointer-visual-python/?orderBy=most_votes&languageTags=python&topicTags=two-pointers
        # T:O(n), S:O(1)
        left, curr = 0, 0
        while curr <= len(nums)-1:
            if nums[curr] != 0:
                nums[left], nums[curr] = nums[curr], nums[left]
                left += 1
                curr += 1 
            else:
                curr += 1

    # def moveZeroes(self, nums: List[int]) -> None:
    #     """
    #     Do not return anything, modify nums in-place instead.
    #     """
    #     # two requirements are mutually exclusive: move all 0 to the end, keep the order of non-zero elements
    #     # T: O(n), S: O(n)
    #     n = len(nums)
    #     ind = 0
    #     while ind < n:
    #         print(ind)
    #         if nums[ind] == 0:
    #             nums.pop(ind) # pop() based on index, not the actual value
    #             nums.append(0) # add zero at the end
    #             n -= 1 # stop the index from accessing zeros which appended in the end -> for循环里改不了n但是while可以
    #         else:
    #             ind += 1
            
    # def moveZeroes(self, nums: List[int]) -> None:
    #     """
    #     Do not return anything, modify nums in-place instead.
    #     """
    #     # T: O(n^2), S: O(1)
    #     # fast, slow: iterate everytime, if this is 0, need to swap and then move
    #     slow = 0 
    #     for fast in range(len(nums)):
    #         if nums[fast] != 0:
    #             slow += 1
    #         else:
    #             while fast < len(nums)-1 and nums[fast] == 0:
    #                 fast += 1
    #             nums[slow], nums[fast] = nums[fast], nums[slow]
    #             slow += 1
    #     return 
```

[350 Intersection of Two Arrays II](https://leetcode.com/problems/intersection-of-two-arrays-ii/description/)
- Tag: Array, Hash Table, Two Pointers, Binary Search, Sorting
- Time: 12-15: X
- Logic of Solution: 
- [x]一题多解:
```Python
class Solution:
    def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
        # two pointer
        # 我nested for loop不理解 但是可以写成如下图中的样子
        # T: O(mlogm + nlogn), S: O(min(m, n))
        result = []
        p1, p2 = 0, 0
        nums1.sort()
        nums2.sort()
        while p1 < len(nums1) and p2 < len(nums2):
            diff = nums1[p1] - nums2[p2]
            if diff == 0:
                result.append(nums1[p1])
                p1 += 1
                p2 += 1
            elif diff < 0:
                p1 += 1
            else:
                p2 += 1 
        return result


    # def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
    #     # hash map on shorter list to save the space resource
    #     # T: O(m+n), S: O(min(m, n))
    #     l1, l2 = len(nums1), len(nums2)
    #     result = []
    #     from collections import Counter
    #     if l1 <= l2:
    #         counts = Counter(nums1)
    #     else:
    #         counts = Counter(nums2)
        
    #     if l1 <= l2:
    #         for i in range(l2):
    #             if nums2[i] in counts.keys() and counts[nums2[i]] > 1:
    #                 counts[nums2[i]] -= 1 
    #                 result.append(nums2[i])
    #             elif nums2[i] in counts.keys() and counts[nums2[i]] == 1:
    #                 del counts[nums2[i]]
    #                 result.append(nums2[i])
    #     else:
    #         for i in range(l1):
    #             if nums1[i] in counts.keys() and counts[nums1[i]] > 1:
    #                 counts[nums1[i]] -= 1 
    #                 result.append(nums1[i])
    #             elif nums1[i] in counts.keys() and counts[nums1[i]] == 1:
    #                 del counts[nums1[i]]
    #                 result.append(nums1[i])
    #     return result
```

[345 Reverse Vowels of a String](https://leetcode.com/problems/reverse-vowels-of-a-string/description/)
- Tag: Two Pointers, String
- Time: 12-15: X
- Logic of Solution: swap但是string是immutable的 所以要换成list 最后再''.join(lst)
- []一题多解:
```Python
class Solution:
    def reverseVowels(self, s: str) -> str:
        # 像洋葱一样交换顺序 reverse是这种意思吗
        # reverse: change the direction 
        # 注意string is immutable ->所以可以转化成
        # T: O(n), S:O(n)
        left, right = 0, len(s)-1
        s = list(s)
        while left < right:
            vowels = ['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U']
            if s[left] not in vowels and s[right] not in vowels:
                left += 1
                right -= 1
            elif s[left] not in vowels and s[right] in vowels:
                left += 1
            elif s[left] in vowels and s[right] not in vowels:
                right -= 1            
            if s[left] in vowels and s[right] in vowels:
                s[left], s[right] = s[right], s[left]
                left += 1
                right -= 1
        return ''.join(s) 
```

[246 Strobogrammatic Number](https://leetcode.com/problems/strobogrammatic-number/description/)
- Tag: Hash Table, Two Pointers, String
- Time: 12-15 
- Logic of Solution: 
- []一题多解:
```Python
class Solution:
    def isStrobogrammatic(self, num: str) -> bool:
        # True: 0, 1, 6->9, 8, 9->6
        # T: O(n), S: O(1)
        s1 = set(['0', '1', '8'])
        num1 = list(num)
        n = len(num1)
        left, right = 0, n-1
        while left <= right:
            if num1[left] in s1 and num1[right] in s1 and num1[left] == num1[right]:
                left += 1 
                right -= 1 
            elif (num1[left] == '6' and num1[right] == '9') or (num1[left] == '9' and num1[right] == '6'):
                left += 1
                right -= 1
            else: 
                return False
        return True
```

[125 Valid Palindrome](https://leetcode.com/problems/valid-palindrome/description/)
- Tag: Two Pointer, String
- Time: 12-15: X
- Logic of Solution: 
- []一题多解:
```Python
class Solution:
    def isPalindrome(self, s: str) -> bool:
        # isalnum() determine whether is number or letter 
        # T: O(n), S: O(1)
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
```
[392 Is Subsequence](https://leetcode.com/problems/is-subsequence/description/)
- Tag: Two Pointers, String, Dynamic Programming, Greedy
- Time: 12-16
- Logic of Solution: base case? 边界？
- []一题多解:
```Python
# Divide and Conquer with Greedy: reduce the problem into subproblems with smaller scales recursively
# until the problem becomes small enough to tackle with. We then use the results of subproblems 
# to construct the solution for the original problem
class Solution:
    # Divide and Conquer with Greedy
    # T: O(n) n is the length of target 
    # S: O(n) the recursion incurs some additional memory consumption in the function call stack
    def isSubsequence(self, s: str, t: str) -> bool: 
        left, right = len(s), len(t)
        def solveSub(p_left, p_right):
            if p_left == left: return True 
            if p_right == right: return False 
            if s[p_left] == t[p_right]:
                p_left += 1
            p_right += 1
            return solveSub(p_left, p_right)
        return solveSub(0,0)
    
    # # Two Pointer: T:O(n), S:O(1)
    # def isSubsequence(self, s: str, t: str) -> bool:
    #     left, right = len(s), len(t)
    #     p_left, p_right = 0, 0 
    #     while p_left < left and p_right < right:
    #         if s[p_left] == t[p_right]:
    #             p_left += 1
    #         p_right += 1
    #     return p_left == left

    # Divide and Conquer with Hashmap

    # Dynamic Programmging
```
[186 Reverse Words in a String II](https://leetcode.com/problems/reverse-words-in-a-string-ii/description/)
- Tag: Two Pointers, String
- Time: 12-16
- Logic of Solution: 
- 一题多解:
```Python
# T: O(n), S: O(1)
class Solution:
    def reverseWords(self, s: List[str]) -> None:
        # T: O(n), S: O(1)
        """
        Do not return anything, modify s in-place instead.
        """
        self.reverse(s, 0, len(s)-1)
        self.reverseEachWord(s)
        
            
    def reverse(self, s, start_p, end_p) -> None:
        left, right = start_p, end_p 
        while left <= right: 
            s[left], s[right] = s[right], s[left]
            left += 1
            right -= 1
    
    def reverseEachWord(self, s) -> None:
        # two pointer:
        # start: start of word, end: end of word
        n = len(s)
        start, end = 0, 0
        while start < n:
            while end < n and s[end] != ' ': 
                end += 1
            self.reverse(s, start, end-1)
            start = end + 1
            end += 1
```

### KMP
1. 理论篇 https://www.bilibili.com/video/BV1PD4y1o7nd/?vd_source=8b4794944ae27d265c752edb598636de 
2. 主要思想：当出现字符串不匹配时，可以知道一部分之前已经匹配的文本内容，可以利用这些信息避免从头再做匹配 解决字符串匹配 遇到不匹配的位置 回跳到某个匹配过的地方 再匹配 根据：前缀表
3. 前缀表：找到匹配子串的后缀 与之匹配的前缀的 后面一位开始重新开始匹配-》要求的就是一个字符串里面最长相等的前后缀
4. 前缀：包含首字母不包含尾字母的所有子串 都称为前缀 称为prefix 或者存为next告诉我们遇到冲突后 我们需要回退到哪里 后缀： 最长相等前后缀：最长相等前后缀的长度就是我们需要开始重新匹配的下标
2. kmp求前缀的代码 https://www.bilibili.com/video/BV1M5411j7Xx/?vd_source=8b4794944ae27d265c752edb598636de
```Python
// 方法一
class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        a = len(needle)
        b = len(haystack)
        if a == 0:
            return 0
        next = self.getnext(a,needle)
        p=-1
        for j in range(b):
            while p >= 0 and needle[p+1] != haystack[j]:
                p = next[p]
            if needle[p+1] == haystack[j]:
                p += 1
            if p == a-1:
                return j-a+1
        return -1

    def getnext(self,a,needle):
        next = ['' for i in range(a)]
        k = -1
        next[0] = k
        for i in range(1, len(needle)):
            while (k > -1 and needle[k+1] != needle[i]):
                k = next[k]
            if needle[k+1] == needle[i]:
                k += 1
            next[i] = k
        return next
```

[28 Find the Index of the First Occurrence in a String](https://leetcode.com/problems/find-the-index-of-the-first-occurrence-in-a-string/description/)
- Tag: Two Pointers, String, String Matching
- Time: 12-16: O(边界)
- Logic of Solution: 新算法：kMP 
- [x]一题多解:
```Python
?? KMP
```

[151 Reverse Words in a String](https://leetcode.com/problems/reverse-words-in-a-string/description/)
- Tag: Two Pointers, String
- Time: 12-16 X
- Logic of Solution: 
- [x]一题多解:
```Python
class Solution:
    def reverseWords(self, s: str) -> str:
        # deque
        # T & S: O(n)
        from collections import deque
        left, right = 0, len(s)-1
        word, d = [], deque()
        while left <= right and s[left] == ' ':
            left += 1
        while left <= right and s[right] == ' ':
            right -= 1 
        while left <= right:
            if s[left] == ' ' and word:
                d.appendleft(''.join(word))
                word = []
            if s[left] != ' ':
                word.append(s[left])
            left += 1
        d.appendleft(''.join(word))
        return ' '.join(d)

    # def reverseWords(self, s: str) -> str:
    #     # built-in split + reversed
    #     # T & S: O(n)
    #     return " ".join(reversed(s.split()))
```

[11 Container With Most Water](https://leetcode.com/problems/container-with-most-water/description/)
- Tag: Array, Two Pointers, Greedy
- Time: 12-16
- Logic of Solution: 
- [X]一题多解:
```Python
class Solution:
    def maxArea(self, height: List[int]) -> int:
        # towards direction of two pointer
        start, end = 0, len(height)-1
        temp_max = 0
        while start < end:
            temp_max = max(temp_max, min(height[start], height[end])* (end-start))
            if height[start] >= height[end]:
                end -= 1
            else:
                start += 1
        return temp_max

    # def maxArea(self, height: List[int]) -> int:
    #     # brue force
    #     # T: O(n^2), S: O(1) -> n <= 10^5 跑不过
    #     temp_max = 0
    #     for start in range(len(height)):
    #         for end in range(start+1, len(height)):
    #             height_hole = min(height[start], height[end])
    #             weight_hole = end - start
    #             temp_max = max(temp_max, height_hole * weight_hole)
    #     return temp_max
```
## Sliding Window
[3 Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/description/)
- Tag: Hash Table, String, Sliding Window
- Time: 12-17
- Logic of Solution: 不定长的好像对我来说都有难度，1.check if a character occurs before quickly: hash table; (1) intuitive but may cause a TLE, (2) uses a slide window to narrow down the search range (3) make further use of hashmap to reduce the search range faster
- [x]一题多解:
```Python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        # optimized sliding window
        # if we have duplicates, we don't need to check one by one, once we have index, we can directly skip to index =1 
        start_p, end_p = 0, 0
        res = dict()
        ans = 0
        for end_p in range(len(s)):
            if s[end_p] in res:
                start_p = max(res[s[end_p]] + 1, start_p) # ??
            ans = max(ans, end_p - start_p + 1)
            res[s[end_p]] = end_p 
        return ans 
                    
    # def lengthOfLongestSubstring(self, s: str) -> int:
    #     # sliding window: if there is a duplicate, drop the leftmost element
    #     # find the maximum length given the starting point, after find the duplicate move the starting point again
    #     # T & S: On)
    #     from collections import defaultdict 
    #     start_p, end_p = 0, 0 
    #     d = defaultdict(int)
    #     max_temp = 0
    #     for end_p in range(len(s)):
    #         if s[end_p] not in d:
    #             d[s[end_p]] += 1
    #             max_temp = max(max_temp, len(d))
    #         else: #如果没有重复的数 就不会进入下面的判断 知道max_temp
    #             max_temp = max(max_temp, len(d))
    #             while s[end_p] in d:
    #                 d.pop(s[start_p])
    #                 start_p += 1
    #             d[s[end_p]] += 1 #最后要加上去 因为下一次就跳入下一个循环了
    #     return max_temp

    # def lengthOfLongestSubstring(self, s: str) -> int:
    #     # brute force T: O(n^3), S: O(set)
    #     count_max = 0 
    #     for i in range(len(s)):
    #         for j in range(i, len(s)):# allow to have single item
    #             if self.isUnique(s[i:j+1]) is not False: 
    #                 count_max = max(count_max, self.isUnique(s[i:j+1]))
    #     return count_max
    # iterate every str and put it into set one by one 
    # def isUnique(self, substr): # we keep checking whether is unique, i to j-1 already check
    #     ori_len = len(substr) # use hashmap as a sliding window, to check whether one element is in substr costs O(1)
    #     if ori_len == len(set(substr)):
    #         return ori_len
    #     else:
    #         return False
```

[219 Contains duplicate II](https://leetcode.com/problems/contains-duplicate-ii/description/)
- Tag: Array, Hash Table, Sliding Window
- Time: 12-17: X
- Logic of Solution: 定长maintain a length of k window, remove the first one once we above the size of k
- [X]一题多解:
```Python
class Solution:
    def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
        # window size: k, unique set, when duplicated element, return True 
        # T: O(n), S: O(1)
        numset = set()
        for i in range(len(nums)):
            if len(numset) > k:
                # set.remove(element not index)
                numset.remove(nums[i-k-1]) #边界！？？
            if nums[i] in numset:
                return True
            numset.add(nums[i])
        return False

        # from collections import defaultdict
        # d = defaultdict(list)
        # # defaultdict list是没有values（）这个操作符的
        # for ind, num in enumerate(nums):
        #     d[num].append(ind)
        # for key in d.keys():
        #     if len(d[key]) > 1:
        #         for i in range(1, len(d[key])):
        #             if abs(d[key][i-1] - d[key][i]) <= k:
        #                 return True 
        # return False
```

[643 Maximum Average Subarray I](https://leetcode.com/problems/maximum-average-subarray-i/description/)
- Tag: Array, Sliding Window
- Time: 12-17
- Logic of Solution: 看下面备注 什么时候才是sliding window
- [X]一题多解:
```Python
class Solution:
    def findMaxAverage(self, nums: List[int], k: int) -> float:
        # sliding window，T: O(n), S: O(1)
        # https://www.educative.io/courses/grokking-the-coding-interview/JPKr0kqLGNP
        win_start, win_end = 0, 0
        temp_max = float('-Inf')
        temp_sum = 0
        while win_end < len(nums):
            temp_sum += nums[win_end]
            # 因为每次循环都要加后面那个 所以当长度为k是就要减去开头
            if win_end - win_start >= k-1:
                temp_max = max(temp_max, temp_sum)
                temp_sum -= nums[win_start]
                win_start += 1
            win_end += 1
        return temp_max/k

    # def findMaxAverage(self, nums: List[int], k: int) -> float:
    #     # 超时了 这不是sliding window
    #     # T: O(n*k) for every item caculating the sum for next k elements
    #     start, lk = 0, []
    #     temp_max = float('-Inf')#有复数所以temp_max不可以设置为0
    #     while start+k <= len(nums):
    #         lk = nums[start: start+k] #忘记了 python的后面一个index不能取到 所以就是k的index差值
    #         temp_max = max(temp_max, sum(lk))
    #         start += 1
    #     return temp_max/k
```

[121 Best Time to Buy and Sell Stock](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/description/)
- Tag: Array, Dynamic Programming, Sliding Window
- Time: 12-17
- Logic of Solution: 
- 一题多解:
```Python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        # for every element, we are calculating the difference between that element and the minimum of all the values before that element and we are updating the maximum profit if the difference thus found is greater than the current maximum profit.
        # sliding window T: O(n)
        min_price, max_profit = float('Inf'), 0
        for i in range(len(prices)):
            min_price = min(min_price, prices[i])
            max_profit = max(max_profit, prices[i] - min_price)
        return max_profit
   
    # def maxProfit(self, prices: List[int]) -> int:
    #     # T: O(n^2) n=10^5 会time limit exceeded (10^6-10^7极限)
    #     temp_max = 0
    #     for i in range(len(prices)):
    #         for j in range(i+1, len(prices)):
    #             if (prices[j] - prices[i]) > 0:
    #                 temp_max = max(temp_max, prices[j]-prices[i])
    #     return temp_max 
```

[159 Longest Substring with At Most Two Distinct Characters](https://leetcode.com/problems/longest-substring-with-at-most-two-distinct-characters/description/)
- Tag: Hash Table, String, Sliding Window
- Time: 12-17: O
- Logic of Solution: 
- []一题多解:
```Python
class Solution:
    def lengthOfLongestSubstringTwoDistinct(self, s: str) -> int:
        # 题目一开始理解错了
        from collections import defaultdict 
        d = defaultdict(int)
        start, end = 0, 0 
        temp_max = 0
        for end in range(len(s)):
            if (len(d) < 2) or (len(d) == 2 and s[end] in d):
                d[s[end]] += 1
            else:
                while len(d) >= 2: 
                    if d[s[start]] > 1:
                        d[s[start]] -= 1
                    else:
                        d.pop(s[start])
                    start += 1
                d[s[end]] += 1
            temp_max = max(temp_max, end-start+1)
        return temp_max 
```
[340 Longest Substring with At Most K Distinct Characters](https://leetcode.com/problems/longest-substring-with-at-most-k-distinct-characters/description/)
- Tag: Hash Table, String, Sliding Window
- Time: 12-17: O (edge case可以想 什么时候这样写下标可能溢出)
- Logic of Solution: 
- []一题多解:
```Python
class Solution:
    def lengthOfLongestSubstringKDistinct(self, s: str, k: int) -> int:
        # sliding window
        from collections import defaultdict 
        start, end = 0, 0
        temp_max = 0
        d = defaultdict(int)
        if k== 0: return 0 # edge case 要不然下面的跑不通 d里面什么东西也没有
        for end in range(len(s)):
            if (s[end] in d and len(d) <= k) or (s[end] not in d and len(d) < k):
                d[s[end]] += 1
            else:
                while len(d) >= k:
                    if d[s[start]] > 1:
                        d[s[start]] -= 1
                    else:
                        d.pop(s[start])
                    start += 1 
                d[s[end]] += 1
            temp_max = max(temp_max, end-start+1) 
        return temp_max
```

## Linked List
[206 Reverse Linked List](https://leetcode.com/problems/reverse-linked-list/description/)
- Tag: Linked List, Recursion
- Time: 12-17: X
- Logic of Solution: 
- [x]一题多解:
```Python
```


## Trie
[208 Implement Trie (Prefix Tree)](https://leetcode.com/problems/implement-trie-prefix-tree/description/)
- Tag: Hash Table, String, Design, Trie
- Time: 12-16
- Logic of Solution: 
- 一题多解:
??

[211 Add and Search Word - Data Structure Design]()
- Tag: 
- Time: 12-16
- Logic of Solution: 
- 一题多解:
??

## Prefix Sum

## Binary Search

[74 Search a 2D Matrix](https://leetcode.com/problems/search-a-2d-matrix/description/)
- Tag: Array, Binary Search, Matrix
- Time: 12-18
- Logic of Solution: row: i*j // n, column: i*j % n
- 一题多解:
```Python
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        # sorted array is prefect for binary search
        # T: O(logmn), S: O(1)
        m = len(matrix) # row
        if m == 0: return False
        n = len(matrix[0]) # col
        left_p, right_p = 0, n*m-1
        while left_p <= right_p:
            mid_p = left_p + (right_p-left_p)//2 
            mid = matrix[mid_p // n][mid_p % n] # column
            if mid == target: return True
            elif mid < target:
                left_p = mid_p + 1
            else:
                right_p = mid_p - 1
        return False
```
[875 Koko Eating Bananas](https://leetcode.com/problems/koko-eating-bananas/description/)
- Tag: Array, Binary Search
- Time: 12-18
- Logic of Solution: 
- 一题多解:
```Python
class Solution:
    def minEatingSpeed(self, piles: List[int], h: int) -> int:
    # binary search: 不用一个个increment add, 可以在一个区间范围快速搜
    # m: max number of bananas in a single pile from piles
    # T: O(n log m), S: O(1)
        if len(piles) == 0: return 0
        left, right = 1, max(piles) #要求是最小的k 
        while left < right:
            cumu_hour = 0
            mid = left + (right - left) // 2
            for b in piles:
                cumu_hour += math.ceil(b / mid)
            #if cumu_hour == h: #要保证没有目标解的时候也有返回值 否则会返回none type但是题目又要求是int
                #return mid #错误的因为要找到最小值 而不是当可以的时候
            if cumu_hour > h: #边界！！
                left = mid + 1
            else:
                right = mid  
        return left
    
    # def minEatingSpeed(self, piles: List[int], h: int) -> int:
    #     # brute force: starting from speed with 1, increment the speed until koko can eat all bananas
    #     # T: O(n*m), S: O(1)
    #     speed = 1
    #     cumu_hour = float('Inf')
    #     while cumu_hour > h:
    #         cumu_hour = 0
    #         for b in piles:
    #             cumu_hour += math.ceil(b / speed)
    #         speed += 1 
    #     return speed-1 #最后输出要减一
```
## Stack

[20 Valid Parentheses](https://leetcode.com/problems/valid-parentheses/description/)
- Tag: String, Stack
- Time: 12-19
- Logic of Solution: 
- []一题多解:
```Python
class Solution:
    def isValid(self, s: str) -> bool:
        # T & S: O(n)
        map = {')':'(', ']':'[', '}':'{'}
        stack = []
        for i in range(len(s)):
            if s[i] not in map.keys():
                stack.append(s[i])
            if s[i] in map.keys() and len(stack) != 0: #有edge case如果空怎么pop
                popitem = stack.pop()
                if popitem != map[s[i]]:
                    return False
            elif s[i] in map.keys() and len(stack) == 0:
                return False
        if len(stack) != 0:
            return False
        return True 
```

[150 Evaluate Reverse Polish Notation](https://leetcode.com/problems/evaluate-reverse-polish-notation/description/)
- Tag: Array, Math, Stack
- Time: 12-19
- Logic of Solution: 
- []一题多解:
```Python
class Solution:
    def evalRPN(self, tokens: List[str]) -> int:
        operations = {
            '+': lambda x, y: x+y,
            '-': lambda x, y: x-y,
            '*': lambda x, y: x*y,
            '/': lambda x, y: x/y
        }
        stack = []
        for i in range(len(tokens)):
            if len(stack) > 1 and tokens[i] in operations.keys():
                num2 = stack.pop()
                num1 = stack.pop()
                temp = operations[tokens[i]](int(num1), int(num2)) #可能可以问一下传入数字的类型 如果是string就要转化下
                stack.append(temp) 
            if tokens[i] not in operations.keys():
                stack.append(tokens[i])
        return int(stack[0])
```
## Monotonic Stack
- a stack where elements are always in sorted order
- monotonic decreasing means that the stack will always be sorted in descending order
- monotonic stacks are a good option when a problem involves comparing the size of numeric elements, with their order being relevant

[739 Daily Temperatures](https://leetcode.com/problems/daily-temperatures/description/)
- Tag: Array, Stack, Monotonic Stack
- Time: 12-19
- Logic of Solution: 
- []一题多解:
```Python
class Solution:
    # 还有更space efficient的算法没有看 ？？
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        # monotonic stack, T&S: O(n)
        result = [0]*len(temperatures)
        stack = []
        for ind, temp in enumerate(temperatures):
            if len(stack) == 0:
                stack.append([ind, temp])
            if stack[-1][1] >= temp:
                stack.append([ind, temp])
            else:
                while len(stack) > 0 and stack[-1][1] < temp:
                    result[stack[-1][0]] = ind - (stack[-1][0])
                    stack.pop()
                stack.append([ind, temp])
        return result

    # def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
    #     # brute force: T: O(n^2): TLE
    #     result = []
    #     for i in range(len(temperatures)):
    #         count = 0
    #         for j in range(i+1, len(temperatures)):
    #             count += 1
    #             if temperatures[j] > temperatures[i]:
    #                 result.append(count)
    #                 break
    #         if count == (len(temperatures) - i - 1) and temperatures[-1] <= temperatures[i]:
    #             result.append(0) 
    #     return result
```

[739 Daily Temperatures](https://leetcode.com/problems/daily-temperatures/)
- Tag: Array, Stack, Monotonic Stack
- Time: 12-19
- Logic of Solution: 
- 一题多解:

## Tree
- ??他的time and space我不是很清楚 :
- Master Therome: 见notion
- string match: string hashing or KMP （kmp下次要记一下）
-  分治法需要在这次搞懂

### DFS VS BFS
1. bfs在子节点数较多的情况下，消耗内存十分严重。queue很大
2. dfs难以寻找最优解，仅仅能寻找一个解，其优点是内存消耗小，克服了bfd的缺点

### 分治法
1. 由分divide和治conquer两部分组成，通过把原问题分为子问题，再将子问题进行处理合并，从而实现对原问题的求解
2. 时间复杂度：我们可以荣归数学表达式来表示这个火车，然后用master theorem来求解。另外自上而下的分治可以和memorization结合，来避免重复遍历相同的子问题
3. dfs和回溯都是很套路的题目，做的都是搜索，然后返回想要的东西，而divid and conquer每一题都有自己的写法，思想上更广泛
4. 想想有一个神奇的函数可以解决我们的问题 大部分的时候 这个神奇的函数就是 直接回答问题的 需要定义这个神奇的函数的参数和输出是什么 之后写出合并的公式conquer 最后想一想base case
5. 多去感受下分治法的不同(1)都是直接写在主函数里面 (2)像搜索题就用dfs 看着像特殊题就用分治法
6. 分治步骤（1）base case看哪些小问题不需要继续往下划 （2）参数和输出都是看递推公式需要哪些东西（3）有个分和治的步骤 分就是干super function的步骤 治就是干合起来的步骤 dfs一般不会又返回值也不会想把两个东西aggregate起来 （bfs不是一层层往下的写法 跟分治法不太大变）

[226 Invert Binary Tree](https://leetcode.com/problems/invert-binary-tree/)
- Tag: Tree, Depth-First Search, BFS, Binary Tree
- Time: 12-19
- Logic of Solution: 
- [x]一题多解:??之后要会 迭代的写法 小唐教了分治法的写法
```Python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        # divide and conquer 比dfs思想更广泛 dfs和回溯都是很套路的题 做的都是搜索 然后返回想要的 
        # dc 每一题都有自己的模版 dfs和回溯都长的差不多
        # beleive in your super function and do the merge function
        if root is None:
            return root 
        if root.left is None and root.right is None:
            return root 
        root.left, root.right = self.invertTree(root.right), self.invertTree(root.left)
        return root 

# class Solution:
#     def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
#         if root is None: return root 
#         self.backtracking(root)
#         return root
        
#     def backtracking(self, node):
#         if node is None: return
#         # pre-order:
#         node.left, node.right = node.right, node.left
#         if node.left:
#             self.backtracking(node.left)
#         if node.right:
#             self.backtracking(node.right)
```

[Minimum Depth of Binary Tree](https://leetcode.com/problems/minimum-depth-of-binary-tree/description/)
- Tag: Tree, DFS, BFS, binary tree 
- Time: 12-24
- Logic of Solution: 小唐教了分治法的写法
- [x]一题多解:
```Python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    # def minDepth(self, root: Optional[TreeNode]) -> int:
    #     if root is None:
    #         return 0
    #     return self.depth(root)
        
    # def depth(self, node: TreeNode) -> int: #depth
    #     # if in one side there is not any children, but the other side has children -> need to cacluate the children side only
    #     # if there is at least one child in each side, do the comparsion
    #     if node is None: return 0
    #     # postorder
    #     left_depth = self.depth(node.left)
    #     right_depth = self.depth(node.right)
    #     if left_depth and right_depth:
    #         return 1 + min(left_depth, right_depth)
    #     else:
    #         return 1 + left_depth + right_depth

    def minDepth(self, root: Optional[TreeNode]) -> int:
        # divide and conquer
        if root is None:
            return 0
        if root.left is None and root.right is not None:
            return 1 + self.minDepth(root.right)
        elif root.right is None and root.left is not None:
            return 1 + self.minDepth(root.left)
        else: 
            return min(self.minDepth(root.left), self.minDepth(root.right)) + 1
```

[Flatten Binary Tree to Linked List](https://leetcode.com/problems/flatten-binary-tree-to-linked-list/description/)
- Tag: Linked List, Stack, Tree, DFS, Binary Tree
- Time: 12-24
- Logic of Solution: 小唐教了分治法的写法
- 一题多解:
```Python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def flatten(self, root: Optional[TreeNode]) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        self.helper(root)
    def helper(self, node) -> tuple[TreeNode]:
        # 分治法：解决这个问题 只需要子问题的解 所以返回值包含必要的信息
        # f(root) -> head, tail
        # f(root) = root -> f(root.left) -> f(root.right)
        if node is None:
            return None
        if node.left is None and node.right is None:
            return node, node
        left_helper = self.helper(node.left) #分治法 assume里面都连接好了
        right_helper = self.helper(node.right)
        node.left = None
        if left_helper: #到了具体怎么练
            node.right = left_helper[0]
            if right_helper:
                left_helper[1].right = right_helper[0] 
                return node, right_helper[1]
            else:
                return node, left_helper[1]
        else:
            if right_helper:
                node.right = right_helper[0] 
                return node, right_helper[1]
            else:
                return node, node 
```

[104 Maximum depth of binary tree](https://leetcode.com/problems/maximum-depth-of-binary-tree/description/)
- Tag: Tree, DFS, BFS, Binary Tree
- Time: 12-20
- Logic of Solution: 
- 一题多解:
```Python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        # postorder
        # every node need to traverse once: T(n) S: depend on how many recurse call which is the height
        # worst case: completely unbalanced: O(n), balanced: o(logn)
        if root is None: return 0
        return self.backtracking(root)
    def backtracking(self, node):
        if node is None: return 0
        left_depth = self.backtracking(node.left)
        right_depth = self.backtracking(node.right)
        return max(left_depth, right_depth)+1
```

[543 Diameter of Binary Tree](https://leetcode.com/problems/diameter-of-binary-tree/description/)
- Tag: Tree, DFS, Binary Tree
- Time: 12-21
- Logic of Solution: 
- 一题多解: 我有戏啊 练出来了一点 就是如果不balanced了数值直接返回 平衡的话就返回当前树的高度 外面的时候就判断一下就好了 如果不是false而是返回数值 就代表是平衡的  
```Python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        # 需要再想一下？？这个参数和返回值
        # T: O(n), S: O(n)
        # Space complexity: O(N)O(N)O(N). The space complexity depends on the size of our implicit call stack during our DFS, which relates to the height of the tree. In the worst case, the tree is skewed so the height of the tree is O(N)O(N)O(N). If the tree is balanced, it'd be O(log⁡N)O(\log N)O(logN).
        # max depth from left tree + max depth from right tree 
        # issue: can not starting from the root
        # post order 
        if root is None: return 0
        # keep track of the longest path we find from the DFS
        self.diameter = 0
        self.backtracking(root)
        return self.diameter

    def backtracking(self, node) -> int:
        if node is None: return 0
        left_depth = self.backtracking(node.left)
        right_depth = self.backtracking(node.right)
        self.diameter = max(self.diameter, left_depth+right_depth) #update with bigger diameter # for overall path
        return max(left_depth, right_depth) + 1 #return the longest one between left and right path # for inidvidual
```

[110 Balanced Binary Tree](https://leetcode.com/problems/balanced-binary-tree/)
- Tag: 
- Time: 12-22
- Logic of Solution: 
- 一题多解:
```Python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        # postorder
        # 确定了肯定是要返回布尔值 但是不知道这个时候终止条件的返回值怎么办->错误的时候就报错，正确的时候就返回高度
        # if the subtree are balance use the current height from subtree to caculate whether current tree are balanced
        # T: O(n) for every subtree, we compute its height in constant time as well as compare the height of its children
        # S: O(n) the recursion stack may go up to O(n) if the tree is unbalanced
        if root is None: return True 
        if self.dfs(root) is False:
            return False
        else:
            return True 
        
    def dfs(self, node):
        if node is None:
            return 0
        if self.dfs(node.left) is not False and self.dfs(node.right) is not False:
            left_height = self.dfs(node.left)
            right_height = self.dfs(node.right)
            if abs(left_height - right_height) > 1:
                return False 
            else:
                return max(left_height, right_height) +1
        return False
```
#### 两棵树
- https://github.com/youngyangyang04/leetcode-master/blob/master/problems/0617.%E5%90%88%E5%B9%B6%E4%BA%8C%E5%8F%89%E6%A0%91.md
- 如何同时遍历两个二叉树的关键：其实和遍历一个树的逻辑是一样的，只不过传入两个树的节点，同时操作
- 例如：617合并二叉树 前序遍历最好理解 按照递归三部曲 （1）确定递归函数的参数和返回值：首先要合入两个二叉树，那么参数至少是要传入两个二叉树的根节点，返回值就是合并之后二叉树的根节点（2）确定终止条件：因为是传入了两个树，那么就有两个树的遍历节点t1,t2,如果t1==null了，两个树的合并就应该是t2了 如果t2也为null也无所谓，合并之后就是null；反过来也类似 (3)确定单层递归的逻辑：单层逻辑比较好写，这里我们重复利用t1这棵树，t1就是合并之后树的根节点（即修改了原来树的结构）。那么单层递归中，就要把两棵树的元素加到一起 合并的左子树就是合并t1左子树 t2左子树之后的左子树，t1的右子树，是合并t1右子树之后的右子树，最终t1就是合并之后的根节点

[100 Same Tree](https://leetcode.com/problems/same-tree/description/)
- Tag: Tree, DFS, BFS. Binary Tree
- Time: 12-22
- Logic of Solution: 这个还是要想一想 对于两个数来说 他们的base case是什么 不是碰到了叶子节点哦
- 一题多解: continue那里错过几次了 是会直接跳出循环 下面的也不会走过
```Python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        # bfs and compare the paths
        # 确实不能bfs直接写 因为如果一层只有一个节点的话并不能分清是左节点 还是右节点的区别
        # dfs -> recursion: how can we break the big question into the same small question
        # preorder: whether root is the same (if not the same immediately return False), whether the left subtrees and right subtrees are the same 
        # T&S: O(p+q) every node; if the tree is completely unbalanced the worse case will have n stack so the space is n
        if p is None and q is None: return True #base case他想的是如果给两个树都是null的话 是不是意味着一样
        elif p is not None and q is None: return False #如果一棵树是空 另外一棵还有树 是不是意味着一样
        elif p is None and q is not None: return False 
        elif p is not None and q is not None and p.val != q.val: return False #如果两棵树都不为空 但是数值不一样 是不是意味着一样
        #elif p is not None and q is not None and p.val == q.val: return True #不能直接会true因为要看子树的情况 并不是前序的头节点为真 就是真
        return (self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right))
```

[572 Subtree of Another Tree](https://leetcode.com/problems/subtree-of-another-tree/description/)
- Tag: Tree, DFS, String Matching, Binary Tree, Hash Function
- Time: 12-24
- Logic of Solution: subtree不要求是叶子节点树 最坏时间复杂度是o(p*q)每一个p节点都要检查下和q像不像 (还没看最好的解法optimized的)
- 一题多解:
```Python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
        # optimal: 下次看seralization + kmp

# class Solution:
#     def isSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
#         # T: O(m*n), S: O(m+n)
#         # bfs, if is identifical return True， 此处还可以dfs的divide and conquer(要递归的所以bfs不可以)写法 一个超级函数就能帮我做事 目标就是把东西分小交给他们做
#         # from collections import deque 
#         # if root is None and subRoot is None: return True 
#         # queue = deque()
#         # queue.append(root)
#         # while queue:
#         #     cur = queue.popleft()
#         #     if cur.val == subRoot.val and self.isIdentifical(cur, subRoot) is True:
#         #         return True 
#         #     #else:
#         #         #continue 这里错了 如果continue的话 下面也不会走 虽然我们这里值会不等 但是下面值有可能相等
#         #     if cur.left:
#         #         queue.append(cur.left)
#         #     if cur.right:
#         #         queue.append(cur.right)
#         # return False 
#         # divide and conquer
#         return self.dfs(root, subRoot)

#     def dfs(self, node, subRoot):
#         if node is None: return False 
#         if self.isIdentifical(node, subRoot): return True 
#         left = self.dfs(node.left, subRoot)
#         right = self.dfs(node.right, subRoot)
#         return left or right 
    
#     def isIdentifical(self, node1, node2)->bool:
#         if node1 is None and node2 is None: return True 
#         if (node1 is None and node2 is not None) or (node1 is not None and node2 is None): return False 
#         if node1 is not None and node2 is not None and node1.val != node2.val: return False 
#         # node1 and node2 is not none
#         left = self.isIdentifical(node1.left, node2.left)
#         right = self.isIdentifical(node1.right, node2.right) 
#         return node1.val == node2.val and left and right    # and 只有都为true时才是true  
```
[236 Lowest Common Ancestor of a Binary Tree](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/description/)
- Tag: Tree, DFS, Binary Tree
- Time: 12-24
- Logic of Solution: 小唐教的回溯模版
- []一题多解:
```Python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        self.paths = []
        self.dfs([root], p, q)
        return self.findLCA(self.paths[0], self.paths[1])

    def findLCA(self, path1, path2):
        i = 0
        while i < len(path1) and i < len(path2) and path1[i] is path2[i]:
            i += 1
        return path1[i-1]

    def dfs(self, cur_path, p, q): 
        # find the paths to p, q
        # 整条路都记下来 接下来的方向之和最后一个node相关
        if cur_path[-1] is p or cur_path[-1] is q:
            self.paths.append(cur_path[:])
        if len(self.paths) == 2:
            return
        node = cur_path[-1]
        for next_node in [node.left, node.right]:
            if next_node:
                cur_path.append(next_node)
                self.dfs(cur_path, p, q)
                cur_path.pop()
```
#### 二叉搜索树

[235 Lowest Common Ancestor of a Binary Search Tree](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/description/)
- Tag: Tree, BFS, DFS, Binary Tree
- Time: 12-24
- Logic of Solution: 
- []一题多解:
```Python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        # backtracking->记录路径 小唐说bst会更简单 因为告诉了方向都不用递归的 像二分法
        # iterate 遍历 to find the split node
        # T O(n), S O(1)
        cur = root
        while cur:
            if cur.val < p.val and cur.val < q.val:
                cur = cur.right 
            elif cur.val > p.val and cur.val > q.val:
                cur = cur.left
            else: 
                return cur 
```

#### 构造二叉树


[]()
- Tag: 
- Time: 12-24
- Logic of Solution: 
- 一题多解:

[]()
- Tag: 
- Time: 12-24
- Logic of Solution: 
- 一题多解:

[102 Binary Tree Level Order Traversal](https://leetcode.com/problems/binary-tree-level-order-traversal/)
- Tag: Tree, BFS, Binary Tree
- Time: 12-22
- Logic of Solution: 
- []一题多解:
```Python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        from collections import deque 
        # T&S: O(n)
        paths = []
        if root is None:
            return paths 
        d = deque()
        d.append(root)
        while d:
            size = len(d)
            result = []
            for i in range(size):
                curr = d.popleft()
                result.append(curr.val)
                if curr.left:
                    d.append(curr.left) # deque加一个数的操作是append
                if curr.right:
                    d.append(curr.right)
            paths.append(result)
        return paths 
```

[199 Binary Tree Right Side View](https://leetcode.com/problems/binary-tree-right-side-view/)
- Tag: Tree, DFS, BFS, Binary Tree
- Time: 12-22
- Logic of Solution: 
- 一题多解:
```Python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        # bfs 
        # T&S: O(n)
        from collections import deque
        paths = []
        if root is None: 
            return paths 
        d = deque()
        d.append(root)
        while d:
            size = len(d)
            result = []
            for i in range(size):
                curr = d.popleft()
                result.append(curr.val)
                if curr.left:
                    d.append(curr.left)
                if curr.right:
                    d.append(curr.right)
            paths.append(result)
        return [x[-1] for x in paths]
```
[1448 Count Good Nodes in Binary Tree](https://leetcode.com/problems/count-good-nodes-in-binary-tree/description/)
- Tag: Tree, DFS, BFS, Binary Tree 
- Time: 12-25
- Logic of Solution: 
- 一题多解:
```Python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def goodNodes(self, root: TreeNode) -> int:
        # 就是普通的search 只是记录了一个当前路径的最大值而已（不含自己） 所以是内涵回溯的dfs
        # 这里是top down的所以只有前序遍历 可以让父节点指向孩子节点
        # T&S: O(n)
        # 和257一样 要先加中间的节点 因为是靠中间节点do something的嘛 所以要放前面？？
        if root is None: return 0
        self.count = 0
        self.path = []
        self.dfs(root, root.val)
        return self.count
    def dfs(self, node, tempmax):
        if node.val >= tempmax:
            self.count += 1
            tempmax = node.val
        else:
            pass 
        if node.left is None and node.right is None: #叶子节点
            return 
        if node.left:
            self.dfs(node.left, tempmax)
        if node.right:
            self.dfs(node.right, tempmax)
```

[98 Validate Binary Search Tree](https://leetcode.com/problems/validate-binary-search-tree/description/)
- Tag: Tree, BFS, DFS, Binary Tree
- Time: 12-25
- Logic of Solution: 
- 一题多解:
```Python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        # 分治
        # 三部曲：神奇函数的参数和返回值 递推公式 base case 
        # bool, max, min
        return self.helper(root)[0]

    def helper(self, node):
        if node is None: 
            return True, float("-Inf"), float("Inf")
        left, right = self.helper(node.left), self.helper(node.right)
        resb = left[0] and right[0] and node.val > left[1] and node.val < right[2]
        resmax = max(left[1], right[1], node.val) #一个函数call好多遍时间复杂度会高很多
        resmin = min(left[2], right[2], node.val)
        return resb, resmax, resmin #因为都要用 所以返回三个值
```
## Recurrsion & Backtracking
- 回溯是递归的副产品，只要有递归就会有回溯
- 回溯的效率：虽然很难，很不好理解，但是回溯法并不是什么高效的算法，因为回溯的本质是穷举，即穷举出所有可能，然后选出我们想要的答案，如果想要回溯法高效一点，可以加一些剪枝的操作，但也改变不了回溯就是穷举的本质
- 虽然回溯不高效但是对于很多问题 如下面列出的 只有回溯这种暴力解法：（1）组合问题：n个数里面按照一定规则找出k个数的集合（2）切割问题：一个字符串按一定规则有几种切割方式（3）子集问题：一个N个数的集合中有多少符合条件的子集（4）排列问题：N个数按照一定规则全排列，有几种排列方式（5）棋盘问题：N皇后，解数独 （组合：不强调顺序；排列：有顺序）
- 如何理解回溯法：都可以抽象为树形结构。回溯法解决的都是在集合中递归查找子集，集合的大小就构成了树的宽度；递归的深度，构成了树的深度。递归就要有终止条件，所以必然是一颗高度有限的n叉树
- 回溯法的模板： 回溯三部曲：（1）回溯函数模版返回值以及参数（返回值一般为void，回溯法的参数没有二叉树那么简单，所以一般都是些逻辑，然后需要什么参数再加）（2）回溯终止条件（对于树，一般来说搜到叶子节点了，也就找到了满足条件的一般答案，把这个答案存起来，并结束本层递归）if(终止条件){存放结果; return;} （3）回溯搜索的编列过程（已经提及 回溯法一般是在集合中递归搜索，集合的大小构成了树的宽度，递归深度构成树的深度）for (选择：本层集合中元素（树中节点孩子的数量就是集合的大小）){处理节点；backtracking(路径，选择列表)//递归；回溯，撤销处理结果} 解释：for循环就是遍历集合区间，可以理解成一个节点有多少个孩子，这个for循环就执行多少次;backtracking这里调用自己，实现递归。大家可以看出for循环可以理解成横向遍历，backtracking（递归）就是纵向遍历，这样就把这棵树全遍历完了，一般来说，搜素叶子节点就是找的其中一个结果

[22 Generate Parentheses](https://leetcode.com/problems/generate-parentheses/description/)
- Tag: 
- Time: 12-19
- Logic of Solution: 
- 一题多解:

## Graph

[78 Subsets](https://leetcode.com/problems/subsets/description/)
- Tag: Array, Backtracking, Bit Manipulation(小唐说没必要) 
- Time: 12-21
- Logic of Solution: 
- 一题多解:
```Python
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        # dfs
        # T & S:  Time complexity: O(N×2^N) to generate all subsets and then copy them into output list. 子节点就那么多？？
        # 终止条件：剩余集合为空
        # startind: 这层递归从哪开始取
        self.path = []
        result = []
        self.backtracking(nums,0,result)
        return self.path
    def backtracking(self, nums, startind, result) -> None:
        self.path.append(result[:])
        if startind == len(nums):
            #result = [] #不能要 及时是每一个节点都要更新 但是下一层的递归还是要依靠上一层的prev结果
            return 
        for i in range(startind, len(nums)):#这层递归的子集 #之前写的变化的不是下标 就会有没从startind开始的数字 for num in nums[startind:]
            result.append(nums[i])
            self.backtracking(nums, i+1, result) #下层递归从哪可以开始取
            result.pop()
```

## Union Find
1. 并查集或者disjoint set可以动态的连通两个点，并且可以非常迅速的判断两个点是否连通
2. 假设存在n个节点，我们先将所有节点的父亲标为自己，每次要连接节点i和j时，我们可以将i的父亲标为j,每次要查询两个节点是否相连时，我们可以查找i和j的祖先是否最终为同一个人
3. 并查集，其中union操作可以将两个集合连在一起，find操作可以查找给定节点的祖先，并且可以的话，将集合的层数/高度降低



## Implement Question
[Smallest Number in Infinite Set](https://leetcode.com/problems/smallest-number-in-infinite-set/description/)
- Tag: Hash Table, Design, Heap (Priority Queue)
- Time: 12-14
- Logic of Solution: 
- 一题多解:
??

[155 Min Stack](https://leetcode.com/problems/min-stack/)
- Tag: Stack, Design
- Time: 12-19
- Logic of Solution: 
- 一题多解:


## Greedy 
[392 Is Subsequence](https://leetcode.com/problems/is-subsequence/description/)
- Tag: Two Pointers, String, Dynamic Programming, Greedy
- Time: 
- Logic of Solution: 
- 一题多解:
```Python
# Divide and Conquer with Greedy: reduce the problem into subproblems with smaller scales recursively
# until the problem becomes small enough to tackle with. We then use the results of subproblems 
# to construct the solution for the original problem
??
```

## Merge Interval

[252 Meeting Room](https://leetcode.com/problems/meeting-rooms/description/)
- Tag: Array, Sorting
- Time: 12-19
- Logic of Solution: lambda x: x[0]
- []一题多解:
```Python
class Solution:
    def canAttendMeetings(self, intervals: List[List[int]]) -> bool:
        # T: O(n log n), S: O(1)
        intervals.sort(key=lambda x: x[0]) # from early to late based on starting time 
        for i in range(len(intervals)-1):
            if intervals[i][1] > intervals[i+1][0]:
                return False
        return True 
```

[56 Merge Intervals](https://leetcode.com/problems/merge-intervals/description/)
- Tag: Array, Sorting
- Time: 12-19
- Logic of Solution: 
- []一题多解:
```Python
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        # T: O(n log n), S: O(1)
        intervals.sort(key=lambda x:x[0])
        merged = []
        for interval in intervals:
            if len(merged) == 0 or merged[-1][1] < interval[0]: # first element or no overlap
                merged.append(interval)
            else:
                merged[-1][1] = max(merged[-1][1], interval[1])
        return merged 
```
[57 Insert Interval](https://leetcode.com/problems/insert-interval/description/)
- Tag: Array
- Time: 12-25
- Logic of Solution: 
- [x]一题多解:
```Python
class Solution:
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        # make use of the property that the intervals were initially sorted according to their start times
        # one pass linear scan T: O(n), S: O(n)
        res = []
        # 前面没有重合 都是res[1]比新数组小 然后经历一段overlap 直到 res[0]比新数组的尾巴大
        # edge case考虑： 两个都为空 任意一个为空 题目规定newinterval不为空 两个都只有一个数
        # 需要多思考下这个逻辑
        for i in range(len(intervals)):
            if newInterval[1] < intervals[i][0]: # no overlap #1
                res.append(newInterval)
                res = res + intervals[i:]
                return res
            elif newInterval[0] > intervals[i][1]: #no overlap #2
                res.append(intervals[i])
            else:
                newInterval = [min(newInterval[0], intervals[i][0]), max(newInterval[1], intervals[i][1])]
        res.append(newInterval)
        return res 

        # # T: O(n log n), S: (n)
        # intervals.append(newInterval)
        # intervals.sort(key = lambda x: x[0]) #这样的写法要记住 x= lambda: a,b: a*b
        # res = []
        # for interval in intervals:
        #     if len(res) ==0 or res[-1][1] < interval[0]:
        #         res.append(interval)
        #     elif res[-1][1] >= interval[0]:
        #         res[-1][1] = max(res[-1][1], interval[1]) #前节点相交了 但是用哪个还不一定呢
        # return res 
```

[]()
- Tag: 
- Time: 12-23
- Logic of Solution: 
- 一题多解:

## Math & Geometry


## Dynamic Programming
[161 One Edit Distance](https://leetcode.com/problems/one-edit-distance/)
- Tag: 
- Time: 12-16
- Logic of Solution: 
- 一题多解:
??

---

[]()
- Tag: 
- Time: 12-24
- Logic of Solution: 
- 一题多解:
