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

[Find All Duplicates in an Array]()
- Tag: Array, Hash Table
- Time: 12-26
- Logic of Solution: 
- []一题多解: 有个更高效的解法可以省去memeory因为是（1～n）并且都为正数字 所以可以
```Python
class Solution:
    def findDuplicates(self, nums: List[int]) -> List[int]:
        from collections import Counter 
        count = Counter(nums)
        res = []
        for num in count.keys():
            if count[num] == 2:
                res.append(num)
        return res 
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

[Two Sum II - Input Array is Sorted](https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/description/)
- Tag: Array, Two Pointers, Binary Search
- Time: 12-26
- Logic of Solution: 
- [x]一题多解:
```Python
class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        # Two Pointer: given they are sorted
        # T: O(n),  S: O(1)
        left, right = 0, len(numbers)-1
        temp_sum = 0
        while left < right:
            temp_sum = numbers[left] + numbers[right]
            if temp_sum < target:
                left += 1
            elif temp_sum > target:
                right -= 1
            else:
                return [left+1, right+1]
        return [-1, -1]   

    # def twoSum(self, numbers: List[int], target: int) -> List[int]:
    #     # Hash Map
    #     # T&S: O(n)
    #     from collections import defaultdict 
    #     d = defaultdict(int)
    #     for ind, num in enumerate(numbers):
    #         if target - num in d.keys():
    #             if ind < d[target-num]:
    #                 return [ind+1, d[target-num]+1]
    #             else:
    #                 return [d[target-num]+1, ind+1]
    #         else:
    #             d[num] = ind 
```

[15 3Sum](https://leetcode.com/problems/3sum/description/)
- Tag: Array, Two Pointers, Sorting
- Time: 12-26
- Logic of Solution: 
- []一题多解:
```Python
from collections import defaultdict

class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
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
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        # quickselct O(n) 算了 下次 但是是分治法思想
        # textbook answer for find kth smallest or biggest or kth most frequent or kth lesss frequent
        # average case is O(n) and worse case is O(n^2) but can be ignore, everytime chose a bad pivot
        # as a result, we will have the perfect ordering in order, sort by frequentcy
        # all the elements on the left are more frequent than pivot and all the elements on the right are less frequent

        #bucket sort: 因为知道每个元素的重复次数是1～k次 所以我们有k个bucket 这样O(n)就可以算得 #O(n)
        from collections import defaultdict, Counter
        if len(nums) == 1 and k==1:
            return nums
        count_map = defaultdict(list) # key: value, key: frequent, value: which number 
        res = []
        counter = Counter(nums) #o(n)
        for num, freq in counter.items():
            count_map[freq].append(num)
        for i in range(len(nums), 0, -1): #最大的frequent往前 #注意！第三个参数是步长 #有edge case如果长度就为1 0是取不到的
            if i in count_map.keys():
                for j in count_map[i]: #哪个数 有这个i frequent次数
                    res.append(j)
                    if len(res) == k:
                        return res 
    # def topKFrequent(self, nums: List[int], k: int) -> List[int]:
    #     # k min-heap, T: O(n log k), S:O(n + k): n for counter, k for final heap
    #     import heapq as pq #priority heap
    #     from collections import Counter
    #     counter = Counter(nums) #O(n)
    #     heap = [] # heapq的initialize就是个list啊！！
    #     # 1. to add first k elements in to k-min heap: k log k 
    #     # 2. pop every item after n-k: n-k * log k (logk: time complexity for heap pop/push )
    #     # 3. conver the final min-heap into array: k log k 
    #     for num, c in counter.items(): #得到counter的pair需要用items
    #         pq.heappush(heap, (c, num)) #定义一个小顶堆 大小为k hq.heappush(pri_que, (freq, val))
    #         if len(heap) > k:
    #             pq.heappop(heap) #priority heap pop也是要有参数的 heap 本身
    #     res = [0]*k
    #     for i in range(k-1, -1, -1):
    #         res[i] = pq.heappop(heap)[1] #先pop最小的
    #     return res 
```
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

[238 Product of Array Except Self](https://leetcode.com/problems/product-of-array-except-self/description/)
- Tag: Array, Prefix Sum
- Time: 12-26
- Logic of Solution: 
- []一题多解:
```Python
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        #prefix product - i - postfix product
        # T&S: O(n)
        prefix_product, postfix_product, res = [0]*len(nums), [0]*len(nums), [0]*len(nums)
        # 0位置的左边没有数 所以应该赋值1， -1位置的右边没有数 所以也是1
        prefix_product[0] = 1
        postfix_product[-1] =1
        for i in range(1, len(nums)):
            prefix_product[i] = prefix_product[i-1] * nums[i-1]
        for i in range(len(nums)-2, -1, -1):
            postfix_product[i] = postfix_product[i+1] * nums[i+1]
        for i in range(len(nums)):
            res[i] = prefix_product[i] * postfix_product[i]
        return res 
```

[36 Valid Sudoku](https://leetcode.com/problems/valid-sudoku/)
- Tag: Array, Hash Table, Matrix
- Time: 12-26
- Logic of Solution: 
- 一题多解:
```Python
class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        from collections import defaultdict 
        row, col = defaultdict(set), defaultdict(set) #key: value, key: which row, value: visited number
        square = defaultdict(set) #key: value, key: (i//3, j//3), value: visited number
        for i in range(9):
            for j in range(9):
                if board[i][j] == '.':
                    continue
                if board[i][j] in row[i] or board[i][j] in col[j] or board[i][j] in square[(i//3, j//3)]:
                    return False
                row[i].add(board[i][j])
                col[j].add(board[i][j])
                square[(i//3, j//3)].add(board[i][j])
        return True
```
[271 Encode and Decode Strings](https://leetcode.com/problems/encode-and-decode-strings/description/)
- Tag: Array, String, Design
- Time: 12-26
- Logic of Solution: 直接抄的code
- 一题多解:
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

[Max Consecutive Ones II](https://leetcode.com/problems/max-consecutive-ones-ii/description/)
- Tag: 
- Time: 12-26
- Logic of Solution: 
- [x]一题多解:
```Python
class Solution:
    def findMaxConsecutiveOnes(self, nums: List[int]) -> int:
        #optimal: sliding window 
        # brute force method are doing repeated work because seuqnces overlap
        # we are checking consecutive sequences blindly
        # if our seq is valid, continue expanding since we want to get the largest
        # if our seq is not valid, stop expanding and contract our sequence #contract收缩
        left, right = 0, 0
        count_zeros =0
        longest = 0
        while right < len(nums):
            if nums[right] == 0:
                count_zeros += 1 
            while count_zeros ==2:
                if nums[left] == 0:
                    count_zeros -=1
                left += 1 
            longest = max(longest, right-left+1)
            right += 1
        return longest
        
        # # flip:交换 翻转
        # #暴力解法 以每一个数为开始 连续的1能有几个 T:O(n^2)
        # longest = 0
        # for i in range(len(nums)):
        #     count_zeros = 0
        #     for j in range(i, len(nums)):
        #         if count_zeros == 2:
        #             break #break这个循环去外面， continue是进行下一个循环
        #         if nums[j] == 0:
        #             count_zeros += 1 
        #         if count_zeros <=1:
        #             longest = max(longest, j-i+1)
        # return longest

#我的暴力写法也是有错的：我这只能替换第一个0 但是不能替换之后的0
# class Solution:
#     def findMaxConsecutiveOnes(self, nums: List[int]) -> int:
#         # flip:交换 翻转
#         flip0 = 1
#         count = 0
#         longest = 0
#         for i in range(len(nums)):
#             if nums[i] == 1:
#                 count += 1
#                 longest = max(longest, count)
#             elif nums[i] == 0 and flip0 == 1:
#                 count += 1
#                 flip0 -= 1
#                 longest = max(longest, count)
#             elif nums[i] == 0 and flip0 == 0: #还是有一个edge case错过 因为没有判断第二个0的时候 什么操作都没有记录 会丢失之前累加的信息
#                 longest = max(longest, count)
#                 count = 0
#         return longest
```
[Max Consectuice Ones III](https://leetcode.com/problems/max-consecutive-ones-iii/)
- Tag: Array, Binary Search, Sliding Window, Prefix Sum
- Time: 12-26
- Logic of Solution: 
- 一题多解:
```Python
```

[Split Array into Consecutive Subsequences](https://leetcode.com/problems/split-array-into-consecutive-subsequences/)
- Tag: Array, Hash Table, Greedy, Heap (Priority Queue)
- Time: 12-26
- Logic of Solution: 
- 一题多解:
```Python
```

[Longest Common Prefix](https://leetcode.com/problems/longest-common-prefix/)
- Tag: String
- Time: 12-26
- Logic of Solution: 
- 一题多解:
```Python
```

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

## Backtracking
[39 Combination Sum](https://leetcode.com/problems/combination-sum/description/)
- Tag: Array, Backtracking
- Time: 12-26
- Logic of Solution: given an array of distinct integers, return a list of all unique combinations of cancdidates where the chosen numbers sum to target. The same number may be chosen from cnadidates an unlimited number of times. Two combinations are unique if the frequency of at least one of the chosen numbers is different
- 一题多解:
```Python
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        # bracktracking 
        # 1.要inistiate的结果的list 
        # 2.搜索的起点是什么
        # 常见trick:把输入的东西sort一下 更好去重
        self.sums = []
        self.candidates = sorted(candidates) #这个才有返回值
        self.target = target
        self.backtracking([], 0)
        return self.sums
        #回溯要有两个参数：一个时当前的path,一个ind代表之后能走的方向（去重）
    def backtracking(self, cur, ind):
        # 1.解是否满足条件，满足时，记住
        if sum(cur) == self.target: 
            self.sums.append(cur[:])
        # 2.（optional）是否可以提前结束
        if sum(cur) >= self.target:
            return 
        # 3.搜索所有的可能方向
        for i in range(ind, len(self.candidates)):
            cur.append(self.candidates[i])
            self.backtracking(cur, i) # 这里是i因为可以重复自己
            cur.pop()
```

[40 Combination Sum II](https://leetcode.com/problems/combination-sum-ii/description/)
- Tag: Array, Backtracking
- Time: 12-26
- Logic of Solution: each candidate may only be used once in the combination
- 一题多解:
```Python
class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        #backtracking是爆搜
        #去重技巧 sort 
        #此时没有visited因为就是从左往右搜
        self.candidates = sorted(candidates)
        self.target = target
        self.paths = []
        self.backtracking(0, [])
        return self.paths
    
    def backtracking(self, start_ind, cur):
        #1.满足条件append
        #2.搜接下来所有可能方向 (optional 可以判断不能搜的时候 直接return)
        #一般没有返回值 会记录走过的方向作为参数 curr状态 和start_ind下面搜的方向
        if sum(cur) == self.target:
            self.paths.append(cur[:])
            return #这时候就满足条件 可以不用继续搜 （除非有负数）
        if sum(cur) > self.target:
            return 
        for i in range(start_ind, len(self.candidates)):
            if i > start_ind and self.candidates[i] == self.candidates[i-1]: #接下来不能重复 关键start_ind [1,2,2,3]->[1,2,3]可以 再一个[1,2,3]不可以 但是[1,2,2,3]可以
                continue
            cur.append(self.candidates[i])
            self.backtracking(i+1, cur)
            cur.pop()
```

[46 Permutations](https://leetcode.com/problems/permutations/description/)
- Tag: Array, Backtracking
- Time: 12-26
- Logic of Solution: 组合 比如说[1,0] 这两个数的所有不同顺序的组合为[1,0]和[0,1]
- 一题多解:
```Python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        self.perms = []
        self.nums = nums
        self.backtracking([],set())
        return self.perms
    def backtracking(self, cur, cur_ind): # cur走过的路径，cur_ind试过的方向
        if len(cur) == len(self.nums):
            self.perms.append(cur[:])
        for i in range(len(self.nums)):
            if i in cur_ind:
                continue 
            cur.append(self.nums[i])
            # cur_ind是set 
            cur_ind.add(i)
            self.backtracking(cur, cur_ind)
            cur.pop()
            # set要pop指定的数
            cur_ind.remove(i)
```

[79 Word Search](https://leetcode.com/problems/word-search/description/)
- Tag: Array, Backtracking, Matrix
- Time: 12-26
- Logic of Solution: 
- []一题多解:
```Python
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        # backtracking 一般是没有return值的 -》一般分治法有返回着 
        # 比如返回有没有这个解 下面搜索时也要判断有没有true 就要有分治法的思想 一层层往上传
        self.target = [str(i) for i in word]
        self.board = board
        self.m, self.n = len(board), len(board[0]) 
        self.exist = False
        #self.visited = [] #要频繁判断在不在里面肯定是set 而且是在变的 所以要传入
        #每个起点都要遍历一遍 
        for i in range(self.m):
            for j in range(self.n):
                if self.board[i][j] == self.target[0]:
                    self.backtracking(len(self.target)-1, i, j, {(i,j)})
                if self.exist:
                    return True
        return False

    def backtracking(self, remain, curx, cury, visited): #需要maintain一个visited走过的不能再走
        # 1.解是否满足条件，满足时，记住 (不是分治法 没有base case 是硬搜)
        # 2.（optional）是否可以提前结束
        # 3.搜索所有的可能方向
        if remain == 0:
            self.exist = True
            return
        for dx, dy in [(1,0), (0,1), (-1,0), (0,-1)]:
            newx, newy = dx + curx, dy + cury
            #可以往这一个方向走的条件1.不在visit 2.没有出借 3.满足下一个字母的要求
            if (newx, newy) not in visited and newx >= 0 and newx < self.m and newy >= 0 and newy < self.n \
                and self.board[newx][newy] == self.target[-remain]:
                visited.add((newx, newy))
                self.backtracking(remain-1, newx, newy, visited)
                visited.remove((newx, newy)) #set不用pop用remove因为没有序 一定要有参数 
```

[78 Subsets](https://leetcode.com/problems/subsets/description/)
- Tag: Array, Backtracking, Bit Manipulation
- Time: 12-26
- Logic of Solution: given an integer array nums of unique elements, return all possible subsets. The solution set must not contain duplicate substes
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

[90 Subsets II](https://leetcode.com/problems/subsets-ii/)
- Tag: Array, Backtracking, Bit Manipulation
- Time: 12-26
- Logic of Solution: given an integer array nums that may contain duplicates, return all possible subsets. The solution set must not contain duplicate substes
- 一题多解:
```Python
class Solution:
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        self.subsets = []
        self.nums = sorted(nums)
        self.backtracking([], 0)
        return self.subsets

    def backtracking(self, cur, ind):
        self.subsets.append(cur[:])
        for i in range(ind, len(self.nums)):
            # 去重：通过sort 找到附近重复的值
            # 第一种 先找前面再找后面 和先找后面再找前面是重复 所以可以通过找ind后的去重
            # 第二种 前进方向本身就是重复的 因为本身有重复 
            if i > ind and self.nums[i] == self.nums[i-1]: 
                #重复的那串只搜索第一个
                continue 
            cur.append(self.nums[i])
            self.backtracking(cur, i+1)
            cur.pop()
```

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

[Number of Connected Components in an Undirected Graph](https://leetcode.com/problems/number-of-connected-components-in-an-undirected-graph/description/)
- Tag: DFS, BFS, Union Find, Graph
- Time: 12-26
- Logic of Solution: 
- []一题多解:
```Python
class Solution:
    # T O(n + edge)
    def countComponents(self, n: int, edges: List[List[int]]) -> int:
        """Typical Union Find"""
        connect = UnionFind(n)
        for i, j in edges:
            connect.union(i, j)
        res = set()
        for i in range(n):
            res.add(connect.find(i))
        return len(res)


class UnionFind:
    def __init__(self, n):
        self.n = n
        self.parents = [i for i in range(n)] #range不是list 是interate
        self.ranks = [0] * n 

    def find(self, i):
        while self.parents[i] != i:
            i, self.parents[i] = self.parents[i], self.parents[self.parents[i]]
        return i 

    def union(self, i, j):
        x, y = self.find(i), self.find(j)
        if x == y:
            return 
        if self.ranks[x] > self.ranks[y]:
            x, y = y, x 
        self.parents[x] = y 
        if self.ranks[x] == self.ranks[y]:
            self.ranks[y] += 1 
        return 
           
#  # 用一个元素代表连通分量       
# class UnionFind:
#     def __init__(self, n):
#         self.n = n
#         # 每个元素的parent都是他自己： 反过来的树，所有点都是singleton不相连的
#         self.parent = list(range(n))
#         # 大概代表这个元素到叶子节点有多远 为了高效实现union find
#         self.rank = [1] * n
#         # 可以维护也可以不 代表有多少个连通分类 不维护的话 扫描一遍 找到他们的prepresent放到set
#         # self.ncomp = n
    
#     # find():给你元素 要知道他的represent
#     # T: 接近O(1)
#     def find(self, x):
#         while self.parent[x] != x:
#             # 短接 会很快
#             x, self.parent[x] = self.parent[x], self.parent[self.parent[x]]
#         return x
#     # union()：告诉你两个数是相连的 就要更新他们的状态
#     # T: 接近O(1)
#     def union(self, x, y):
#         x, y = self.find(x), self.find(y)
#         if x == y:
#             return
#         # 比较rank 让小的指向大的
#         if self.rank[x] < self.rank[y]:
#             x, y = y, x
#         self.parent[y] = x
#         if self.rank[x] == self.rank[y]:
#             self.rank[x] += 1
#         # self.ncomp -= 1
#         return
```

[200 Number of Islands](https://leetcode.com/problems/number-of-islands/description/)
- Tag: Array, DFS, BFS, Union Find, Matrix
- Time: 12-25
- Logic of Solution: 
- [x]一题多解:
```Python
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        # bfs 
        # T&S: O(m*n)
        from collections import deque
        visited = set() #需要频繁判断在不在里面 并且没有value
        count = 0 
        m, n = len(grid), len(grid[0])
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '0' or (i, j) in visited:
                    continue 
                #一次bfs就是一个岛 
                count += 1
                queue = deque([(i, j)])
                visited.add((i, j))
                while queue:
                    curi, curj = queue.popleft()
                    for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                        nexti, nextj = curi + di, curj + dj
                        if (nexti < m and nexti >= 0 and nextj < n and nextj >= 0) \
                            and grid[nexti][nextj] == '1' and (nexti, nextj) not in visited:
                            queue.append((nexti, nextj))
                            visited.add((nexti, nextj))
        return count 


                

#     def numIslands(self, grid: List[List[str]]) -> int:
#         # 关于（i，j）的union find 下标
#         # T: O(m*n)
#         m = len(grid)
#         n = len(grid[0])
#         islands = UnionFind(m, n)
#         for i in range(m):
#             for j in range(n):
#                 if grid[i][j] == "0":
#                     continue
#                 for di, dj in [(1, 0), (-1, 0), (0, -1), (0, 1)]: #寻找的方向
#                     newi, newj = i + di, j + dj
#                     if newi < m and newi >= 0 and newj < n and newj >= 0 and grid[newi][newj] == "1": #判断有没有出界的技巧
#                         islands.union((i, j), (newi, newj))
#         res = set()
#         for i in range(m):
#             for j in range(n):
#                 if grid[i][j] == '1':
#                     res.add(islands.find((i,j)))
#         return len(res)          

# # unionfind是在做一个封装的活 除非特殊情况
# class UnionFind:
#     # 变成一个dict后面union find都不用改 可以是关于任何元素的union find 这里是关于index的 比如说还可以是string的
#     def __init__(self, m, n):
#         self.parents = {(i, j) : (i, j) for i in range(m) for j in range(n)}
#         self.ranks = {(i, j) : 0 for i in range(m) for j in range(n)}

#     def find(self, x):
#         while self.parents[x] != x: # tuple可以逻辑比较 相等 etc 
#             x, self.parents[x] = self.parents[x], self.parents[self.parents[x]]
#         return x 
#     def union(self, x, y):
#         x, y = self.find(x), self.find(y)
#         if x == y:
#             return 
#         if self.ranks[x] > self.ranks[y]:
#             y, x = x, y
#         self.parents[x] = y #接上去
#         if self.ranks[x] == self.ranks[y]:
#             self.ranks[y] += 1 
#         return 
```

[Max Area of Island](https://leetcode.com/problems/max-area-of-island/description/)
- Tag: Array, DFS, BFS, Union Find, Matrix
- Time: 12-25
- Logic of Solution: 
- []一题多解:
```Python
class Solution:
    def maxAreaOfIsland(self, grid):
        # 先把岛接起来
        m, n = len(grid), len(grid[0])
        islands = UnionFind(m, n)
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 0:
                    continue 
                for di, dj in [(1,0), (-1,0), (0,1), (0, -1)]:
                    newi, newj = i+di, j+dj 
                    if (newi < m and newi >= 0 and newj < n and newj >= 0) and grid[newi][newj] == 1:
                        islands.union((i,j), (newi,newj))
        # union接完了 才可以做操作
        from collections import defaultdict 
        areas = defaultdict(int)
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1: #什么时候才要find 并不是所有的元素都要找represent
                    areas[islands.find((i,j))] += 1
        if len(areas) == 0: return 0 #如果都为0
        return max(areas.values())

class UnionFind:
    def __init__(self, m, n):
        self.parents = {(i,j):(i,j) for i in range(m) for j in range(n)}
        self.ranks = {(i,j):0 for i in range(m) for j in range(n)}

    def find(self, x):
        while x != self.parents[x]:
            x, self.parents[x] = self.parents[x], self.parents[self.parents[x]]
        return x 
    def union(self, x, y):
        x, y = self.find(x), self.find(y)  # represents 
        if x == y:
            return 
        # 不是同一个represent就要union 
        if self.ranks[x] > self.ranks[y]:
            x, y = y, x 
        # union 
        self.parents[x] = y
        # 更新rank 
        if self.ranks[x] == self.ranks[y]:
            self.ranks[y] += 1 
        return 

    
    # def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
    #     # bfs
    #     m, n = len(grid), len(grid[0])
    #     max_area = 0
    #     visited = set() 
    #     from collections import deque 
    #     for i in range(m):
    #         for j in range(n):
    #             #什么时候不用搜
    #             if (i, j) in visited or grid[i][j] == 0:
    #                 continue 
    #             queue = deque()
    #             queue.append((i,j))
    #             temp_area = 0 
    #             # 每次加东西就更新 说明此点被考虑了
    #             visited.add((i,j))
    #             while queue:
    #                 curi, curj = queue.popleft()
    #                 # do something这里算面积
    #                 temp_area += 1 
    #                 for di, dj in [(1,0), (-1,0), (0,1), (0,-1)]:
    #                     #什么时候要把这个点算面积
    #                     #\后不能打东西
    #                     if (curi+di < m and curi+di >= 0 and curj+dj < n and curj+dj >=0) and \
    #                         grid[curi+di][curj+dj] == 1 and (curi+di, curj+dj) not in visited:
    #                         queue.append((curi+di, curj+dj))
    #                         visited.add((curi+di, curj+dj))
    #             max_area = max(max_area, temp_area)
    #     return max_area 
```

[]()
- Tag: 
- Time: 12-26
- Logic of Solution: 
- 一题多解:

### Topological Sort
[207 Course Schedule]()
- Tag: DFS, BFS, Graph, Topological Sort
- Time: 12-26
- Logic of Solution: 
- 一题多解:
```Python
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        # topological sort: 有一堆顺序patrail order关系 但只知道一对一对的关系 这些顺序可以传递
        # 输出时某一种toplogical order 
        # 一般给定edges,然后计算in-degree计算谁指向他
        # BFS图的复杂度: T: O(e+v)， s:O(e+v)
        from collections import defaultdict, deque 
        in_degree, out_edges = defaultdict(int), defaultdict(list)
        queue = deque()
        for lst in prerequisites:
            in_degree[lst[0]] += 1
            out_edges[lst[1]].append(lst[0])
        for course in range(numCourses):
            if in_degree[course] == 0:
                queue.append(course)
        # bfs
        res = [] #最后要跟num courses比较大小 
        while queue:
            cur = queue.popleft()
            res.append(cur)
            for pointed in out_edges[cur]:
                in_degree[pointed] -= 1
                if in_degree[pointed] == 0: #减到零的才加
                    queue.append(pointed)
        return len(res) == numCourses 
```
[Parallel Courses](https://leetcode.com/problems/parallel-courses/description/)
- Tag: Graph, Topological Sort
- Time: 12-26
- Logic of Solution: 
- 一题多解:
```Python
class Solution:
    def minimumSemesters(self, n: int, relations: List[List[int]]) -> int:
        from collections import defaultdict, deque
        in_degree, out_edges = defaultdict(int), defaultdict(list)
        for item in relations:
            in_degree[item[1]] += 1 
            out_edges[item[0]].append(item[1])
        queue = deque()
        for course in range(1, n):
            if in_degree[course] == 0:
                queue.append(course)
        res_order = []
        res = 0
        while queue:
            size = len(queue)
            res += 1
            for i in range(size):
                cur = queue.popleft()
                res_order.append(cur)
                for pointed in out_edges[cur]:
                    in_degree[pointed] -= 1
                    if in_degree[pointed] == 0: #topo的判断
                        queue.append(pointed)
        if len(res_order) < n:
            return -1
        return res
```

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

## Design Question
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

[Moving Average from Data Stream](https://leetcode.com/problems/moving-average-from-data-stream/description/)
- Tag: Array, Design, Queue, Data Steam
- Time: 12-26
- Logic of Solution: 
- 一题多解:
```Python
from collections import deque 

class MovingAverage:
    #这个数据结构最开始需要什么 比如union find 是singlton
    def __init__(self, size: int):
        self.size = size 
        self.queue = deque()
        self.sum = 0
    
    # def next(self, val: int) -> float:
    #     # T&S: O(k), sum需要k
    #     # optimized init 就记录sum 多了就减一 T: O(1), S:O(k)
    #     self.queue.append(val)
    #     cur_size = len(self.queue)
    #     if cur_size <= self.size:
    #         return sum(self.queue)/cur_size 
    #     else:
    #         self.queue.popleft()
    #         return sum(self.queue)/self.size  
    def next(self, val):
        self.sum += val 
        self.queue.append(val)
        if len(self.queue) > self.size:
            self.sum -= self.queue.popleft() 
        return self.sum / len(self.queue)

# Your MovingAverage object will be instantiated and called as such:
# obj = MovingAverage(size)
# param_1 = obj.next(val)
```

[146](LRU Cache)
- Tag: Hash Table, Linked List, Design, Doubly-Linked List
- Time: 12-26
- [x]Logic of Solution: 
- 一题多解:
```Python
# class LRUCache:
#     # design题就是考虑需要什么操作
#     # 这里有个get需要o(1) 所以一定要知道hash map
#     # 使用过插入到头o(1) 所以只能linked list 但是pop出去的时候 不是o(1)需要找到她 所以需要双向linked list可以直接知道他们前面的顺序
#     # 因为还需要找到存在哪 需要一个hash map
#     # 只需要一个hash map（key和node）和一个doubly linked list (最近使用的东西 放在左头 最远使用的放在右边 并且记住了value值 如果有个key找打了她就会告诉value值在哪 我们需要维护顺序 并且可以把使用过的东西在放在最前面 所以需要双向链表 就可以直接知道前面的坐标)

#     def __init__(self, capacity: int): # argument不需要count和其他 因为不需要外部实现 这是我实现的细节
#         #要维护当前多少个node 存起来 要不然算就不是o（1）
#         self.count = 0
#         self.capacity = capacity # head是个需的head
#         self.head = LinkedNode() #必须知道头和尾 前面最近用 后面最少用 所以要知道头节点和尾节点 否则遍历到尾节点需要o(n)
#         self.tail = self.head # linked list一般的head是第一个元素前面的元素 #当没有元素的时候head和tail指的是同一个东西
#         self.nodemap = dict() # key 和node pair (node里有key) #pop的知道是哪个key 
#         self.valmap = dict() # key 和val 

#     def get(self, key: int) -> int:
#         if key not in self.nodemap:
#             return -1 
#         # 找到了要把放前面 #分类讨论如果是尾部掉到头 还需要更新尾
#         cur = self.nodemap[key]
#         if cur is self.head.next:
#             return self.valmap[key] #没有循环 就放在这
#         #记下来改变之后的所有指针方向 对于未知的node都用current, head, tail表示 然后就之后就可以直接带
#         # 把操作中涉及的node先记下来 然后直接按照改的方式更改 
#         # 写一个正常的情况：有些edge case, r是空，p和q一样， etc
#         p = self.head.next 
#         q = cur.prev 
#         r = cur.next
#         #更新链接的时候tail可能会变
#         if cur is self.tail: 
#             self.tail = q 
#         self.head.next = cur 
#         cur.prev = self.head
#         cur.next = p 
#         p.prev = cur 
#         q.next = r 
#         if r:
#             r.prev = q  #如果cur是tail r是none的话 none.prev会报错
#         return self.valmap[key] 
#         # count没变 
#         # nodemap 不用变 node还是那个node 只是连接方式变了 

#     def put(self, key: int, value: int) -> None:
#         if key in self.valmap:
#             self.valmap[key] = value #修改了值也是一次使用 trick：get一遍就行
#             self.get(key) #会有一个值 但是没有return 
#         else:
#             newnode = LinkedNode(val=key) 
#             self.nodemap[key] = newnode 
#             self.valmap[key] = value 
#             self.count += 1 
#             p = self.head.next 
#             self.head.next = newnode 
#             newnode.prev = self.head 
#             newnode.next = p 
#             if p is not None:
#                 p.prev = newnode
#             else:
#                 self.tail = newnode 
#         if self.count > self.capacity:
#             t = self.tail
#             p = self.tail.prev 
#             self.tail = p 
#             p.next = None
#             self.count -= 1
#             self.nodemap.pop(t.val)
#             self.valmap.pop(t.val)
#             del t #删除节点 python有garbaby collection可能就没有
        
# class LinkedNode:
#     # linked list只是node的集合而已 把他们连起来 所以基本单元是node 本身就是node 定义的是每一个node 他们的头就是linked list
#     # tree也是一样 只是node集合
#     def __init__(self, val=0, next=None, prev=None): #定义初始值方便些
#         self.val = val
#         self.next = next 
#         self.prev = prev 

# # Your LRUCache object will be instantiated and called as such:
# # obj = LRUCache(capacity)
# # param_1 = obj.get(key)
# # obj.put(key,value)

class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.hashmap = dict() # key: value
        self.next = dict() # curr key: next key
        self.prev = dict() # cur key: prev key
        self.connect("HEAD", "TAIL")

    def connect(self, node1, node2):
        self.next[node1] = node2
        self.prev[node2] = node1

    def get(self, key):
        if key not in self.next:
            return -1 
        self.connect(self.prev[key], self.next[key]) #需要连前也练后 否则会只存在一个dic中
        self.connect(key, self.next["HEAD"])
        self.connect("HEAD", key)
        return self.hashmap[key]
    
    def put(self, key, value):
        if key in self.next:
            self.hashmap[key] = value
            self.get(key)
        else:
            self.hashmap[key] = value
            self.connect(key, self.next["HEAD"])
            self.connect("HEAD", key)
        if len(self.hashmap) > self.capacity:
            cur = self.prev["TAIL"]
            self.connect(self.prev[cur], "TAIL")
            self.hashmap.pop(cur)
            self.prev.pop(cur)
            self.next.pop(cur)            
```

[Insert Delete GetRandom O(1)]()
- Tag: 
- Time: 12-26
- Logic of Solution: 
- 一题多解:
```Python
from random import randrange

class RandomizedSet:
    # set不能根据index来remove他是hashmap 是根据key来删除
    # hashmap (key: value, value:ind)
    # array (python list) (#remove时 对换 删除最后一个 因为不需要保持顺序 #随机取 就随机)
    def __init__(self):
        self.hashmap = dict()
        self.array = []

    def insert(self, val: int) -> bool:
        if val in self.hashmap:
            return False 
        self.array.append(val)
        self.hashmap[val] = len(self.array)-1
        return True 

    def remove(self, val: int) -> bool:
        if val not in self.hashmap:
            return False 
        cur_ind = self.hashmap[val]
        tail_ind = len(self.array)-1
        tail = self.array[-1]
        self.hashmap[tail] = cur_ind 
        self.hashmap.pop(val)
        self.array[cur_ind] = tail 
        self.array.pop()
        return True        

    def getRandom(self) -> int:
        rem_ind = randrange(0, len(self.array))
        return self.array[rem_ind]        

# Your RandomizedSet object will be instantiated and called as such:
# obj = RandomizedSet()
# param_1 = obj.insert(val)
# param_2 = obj.remove(val)
# param_3 = obj.getRandom()
```

[]()
- Tag: 
- Time: 12-26
- Logic of Solution: 
- 一题多解:
```Python
```

## Dynamic Programming
[161 One Edit Distance](https://leetcode.com/problems/one-edit-distance/)
- Tag: 
- Time: 12-16
- Logic of Solution: 
- [x]一题多解:
```Python
from collections import deque 

class MovingAverage:
    #这个数据结构最开始需要什么 比如union find 是singlton
    def __init__(self, size: int):
        self.size = size 
        self.queue = deque()
        self.sum = 0
    
    # def next(self, val: int) -> float:
    #     # T&S: O(k), sum需要k
    #     # optimized init 就记录sum 多了就减一 T: O(1), S:O(k)
    #     self.queue.append(val)
    #     cur_size = len(self.queue)
    #     if cur_size <= self.size:
    #         return sum(self.queue)/cur_size 
    #     else:
    #         self.queue.popleft()
    #         return sum(self.queue)/self.size  
    def next(self, val):
        self.sum += val 
        self.queue.append(val)
        if len(self.queue) > self.size:
            self.sum -= self.queue.popleft() 
        return self.sum / len(self.queue)

# Your MovingAverage object will be instantiated and called as such:
# obj = MovingAverage(size)
# param_1 = obj.next(val)
```

## Other

### Text File
最近面经考的多：file modification + string matching + toplogucal sort
[Word Frequency](https://leetcode.com/problems/word-frequency/)
- Tag: shell
- Time: 12-26
- Logic of Solution: 
- 一题多解:
```Python
# Read from the file words.txt and output the word frequency list to stdout.
python3 -c '
import re
a = open("words.txt")
b = a.read().strip() #remove spaces at begining and at the end of the strings
a.close()
z = b.split("\n") #根据分隔符 分成一个list of items
y=" ".join(z) #把list转换成 string
# I used regex to split on any whitespace bc I failed a test case
c=re.split(r"\s+", y) #按照pattern slit y成一个个符合要求的list里的元素 \s+一个个或者多个space, \S: not white space, \d: digit, \D: not digit, \w+一个或者多个word, \W: not word *：0 or more, ?:0 or 1, {2:}: 2 or more, {3,5}" 3, 4,5个， 
d=dict()
for w in c:
	x=d.get(w,0)
	d[w]=x+1
out=[(w[1],w[0]) for w in list(d.items())]
out.sort(reverse=True)
for o in out:
	print(o[1],o[0])
'
```

[Transpose File](https://leetcode.com/problems/transpose-file/description/)
- Tag: shell
- Time: 12-26
- Logic of Solution: 
- 一题多解:
```Python
python3 -c '
from collections import defaultdict
data = defaultdict(list)
with open("file.txt") as f:
   for line in f:
       for i, word in enumerate(line.split()):
           data[i].append(word)
for line in data.values():
    print(" ".join(line))
'
```

[Find Duplicate File in Sytem](https://leetcode.com/problems/find-duplicate-file-in-system/description/)
- Tag: Array, Hash Table, String
- Time: 12-26
- Logic of Solution: followup: (1)real file system, how will you search files: DFS or BFS: BFS is easiler to paralleizes, 如果太深会爆栈 (2) if the file content is very large(GB file), how will you modify your solution: compare the size if not equal, then files are different can stop early,  (3) if you can only read the file by 1kb each time, how will you modify your solution: 文件的值要1kb 1kb度 我的算法是用hash tbable,就要找1kb读可以算hash value值的算法that is the file cannot fit the whole rame. use a buffer to read controlled by a loop, read until not needed or to the end (4)what is the most time-consuming part and memory consuming part of it? how to optmize: bfs和dfs永远是o(v+e) 一颗树的e是v-1所以就是顶点树(5)how to make sure the duplicated files you find are not false positve: compare byte to byte to avoid false postive due to collision
- []一题多解:
```Python
class Solution:
    def findDuplicate(self, paths: List[str]) -> List[List[str]]:
        # manipulate string and save in dictionary
        files = collections.defaultdict(list)
        for x in paths:
            sep = x.split()
            path_x = sep[0]
            for f in sep[1:]:
                fname, content = f.split("(")
                files[content].append(path_x + "/" + fname)
        return [v for k, v in files.items() if len(v) > 1]
```

[Making File Names Unique](https://leetcode.com/problems/making-file-names-unique/description/)
- Tag: Array, Hash Table, String
- Time: 12-26
- Logic of Solution: 
- 一题多解:
```Python
class Solution:
    def getFolderNames(self, names: List[str]) -> List[str]:
        from collections import defaultdict
        used = set()
        counter = defaultdict(int)
        result = []
        for name in names:
            count = counter[name]
            candidate = name
            while candidate in used:
                count += 1
                candidate = f'{name}({count})'
            counter[name] = count
            result.append(candidate)
            used.add(candidate)
        return result
```

---

[]()
- Tag: 
- Time: 12-26
- Logic of Solution: 
- 一题多解:
```Python
```
