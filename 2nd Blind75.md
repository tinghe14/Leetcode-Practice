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
- 01/19: 一遍过, 03/22:一遍过，他的升级版II不能写出最优解 只能写出brute force这题还能提现举一反三的有two sum和sliding window的解法
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
[Contains Duplicate II](https://leetcode.com/problems/contains-duplicate-ii/description/)
```Python
class Solution:
    def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
        # T nk, S 1
        # for i in range(len(nums)):
        #     j = 1
        #     while i+j <= len(nums)-1 and i+j <= i+k:
        #         if nums[i] == nums[i+j]:
        #             return True 
        #         j += 1
        # return False
        # similar to two sum using one pass to check duplicates 
        # T n, S n
        # dic = {} # num:ind # duplicates?->update
        # for ind, num in enumerate(nums):
        #     if num in dic and ind - dic[num] <= k: 
        #         print(dic)
        #         return True
        #     dic[num] = ind # dic get updates by the most closest one
        # return False
        # sliding window: hash map support constant time in search, delete and insert
        # T n S k
        # search, insert, and delete an element is O(1) at set
        # maintain a set which will store k elements we have seen before at time i
        # when we traverse the list, if we see this number in the set before
        # return True
        # otherwise, add this number and when we add it also keep the length of set still being k limitation
        seen = set()
        for ind, num in enumerate(nums):
            if num in seen:
                return True
            seen.add(num) 
            if len(seen) > k:
                seen.remove(nums[ind-k])
        return False
```

[Valid Anagram](https://leetcode.com/problems/valid-anagram/)
- 01/19: 可以bug free写出来, 03/12: 忘记了dict的del, 还忘了检查最后长度
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
- 01/26 不知道怎么对string进行处理, 03/12: right index包含不包含，边界如何保持不出界的小tips
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

[Best Time to Buy and Sell Stock II](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/description/)
- 03/14: 没有想法
```Python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        # one-pass, T O(n), S O(1)
        total = 0
        for i in range(1, len(prices)):
            if prices[i] - prices[i-1] > 0:
                total += prices[i] - prices[i-1]
        return total
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
```Python
# def linked list
class ListNode:
    def __init__(self, val, next=None):
        self.val = val
        self.next = next
```

[Reverse Linked List](https://leetcode.com/problems/reverse-linked-list/)
- 01/30: 没有想法，linked list next指针好像又不好理解
```Python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        # dummy head 翻转后要被指向的头
        # a->b->c
        if head is None:
            return head
        dummy_head, p1, p2 = None, head, head.next #初始状态 还没轮到c
        while p2 != None: #不为空
            p1.next = dummy_head # b->a
            dummy_head, p1, p2 = p1, p2, p2.next #往后一个
        p1.next = dummy_head #最后的节点转一下
        return p1

#迭代法
class Solution:
    def reverseList(self, head: ListNode)-> ListNode:
        pre, cur = None, head
        while(cur != None):
            temp = cur.next #保存一下 cur的下一个节点，因为接下来要改变cur.next
            cur.next = pre #反转
            pre = cur #更新pre, cur指针
            cur = temp
        return pre 

#递归法
class Solution:
    def reverseList(self, head: ListNode)-> ListNode:
        def reverse(pre, curr):
            if not cur:
                return pre
            temp = cur.next
            cur.next = pre
            return reverse(cur, temp)
        return reverse(None, head)
``` 

[Merge Two Sorted Lists](https://leetcode.com/problems/merge-two-sorted-lists/)
- 01/30: 没有想法
```Python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode() # don't need to worry about edge case which need inserting to empty list
        tail = dummy
        while list1 and list2:
            if list1.val < list2.val:
                tail.next = list1
                list1 = list1.next
            else:
                tail.next = list2
                list2 = list2.next
            tail = tail.next
        if list1:
            tail.next = list1
        elif list2:
            tail.next = list2
        return dummy.next

#迭代法
class Solution:
    def mergeTwoLists(self, l1, l2):
        #maintain an unchanging reference to node ahead of the return node
        prehead = ListNode(-1)
        prev = prehead
        while l1 and l2:
            if l1.val <= l2.val:
                pre.next = l1
                l1 = l1.next
            else:
                prev.next = l2
                l2 = l2.next
            prev = prev.next 
        #at least one of l1 and l2 can still have nodes at this point.
        #so connect the non-null list to the end of the merged list
        prev.next = l1 if l1 is not None else l2
        return prehead.next 

#递归法
class Solution:
    def mergeTwoLists(self, l1, l2):
        if l1 is None:
            return l2
        elif l2 is None:
            return l1
        elif l1.val < l2.val:
            l1.next = self.mergeTwoLists(l1.next, l2)
            return l1
        else:
            l2.next = self.mergeTwoLists(l1, l2.next)
            return l2
```

[Linked List Cycle](https://leetcode.com/problems/linked-list-cycle/)
- 02/02: 没有想法
```Python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        # T: O(n), S: O(1)
        if head is None:
            return False
        slow, fast = head, head.next
        while slow != fast:
            if fast is None or fast.next is None:
                return False 
            slow = slow.next #走一格
            fast = fast.next.next #走两格
        return True
```

[Reorder List](https://leetcode.com/problems/reorder-list/description/)
- 02/06: 没有想法
```Python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reorderList(self, head: Optional[ListNode]) -> None:
        """
        Do not return anything, modify head in-place instead.
        """
        # can divide into 3 steps
        # 2 pointers: slow and fast to find medium
        # reverse second part in-place
        # connect them together
        # 1st part
        # T: o(n), S: o(1)
        if not head:
            return None
        slow, fast = head, head
        while fast and fast.next: #这个boundary不是很确定怎么设置
            slow = slow.next
            fast = fast.next.next
        # 2nd part
        prev, cur = None, slow
        while cur:
            temp = cur.next
            cur.next = prev
            prev = cur
            cur = temp
            # more elegant way in python
            #cur.next, prev, cur = prev, cur, cur.next
        # 3rd part
        first, second = head, prev
        while second.next:
            first.next, first = second, first.next
            second.next, second = first, second.next
```

[Remove Nth Node from End of List]()
- 02/06: 有想法 但是不会实现 dummy head的思想要多应用（删除head的情况 更简单）
```Python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        # the dummy head is set to avoid the case that we need t o delete head
        dummy = ListNode(0, head)
        left = dummy
        right = head
        while n:
            right = right.next
            n -= 1 
        while right:
            left = left.next
            right = right.next 
        # delete
        left.next = left.next.next
        return dummy.next
```
# Trees
[Invert Binary Tree](https://leetcode.com/problems/invert-binary-tree/)
- 02/06: 基本操作忘了
```Python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        # recursive
        # T: o(n), S: o(h) h: height of recursive tree, the largest is n
        # if root is None: 
        #     return root
        # right = self.invertTree(root.right)
        # left = self.invertTree(root.left)
        # root.right, root.left = left, right
        # return root
        # iterative, here is the same as BFS
        from collections import deque
        if root is None:
            return root
        queue = deque([root])
        while queue:
            curr = queue.popleft()
            curr.left, curr.right = curr.right, curr.left
            if curr.left:
                queue.append(curr.left)
            if curr.right:
                queue.append(curr.right)
        return root
```

[Maximum Depth of Binary Tree](https://leetcode.com/problems/maximum-depth-of-binary-tree/)
- 02/06: 基础不牢固 三种写法不能立刻写出
``` Python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        # recursive, same as divide and conquer here
        # T, S: o(n), best case of s is tree is completely balanced, o(logn)
        # if not root:
        #     return 0
        # return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))

        # iterative
        # stack = [[root, 1]]
        # res = 0 
        # while stack:
        #     node, depth = stack.pop()
        #     if node:
        #         res = max(res, depth)
        #         stack.append([node.left, depth+1])
        #         stack.append([node.right, depth+1])
        # return res 

        # BFS
        from collections import deque
        q = deque()
        if root:
            q.append(root)
        level = 0 
        while q:
            for i in range(len(q)):
                node = q.popleft()
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
            level += 1
        return level
```

[Same Tree](https://leetcode.com/problems/same-tree/description/?orderBy=most_votes)
- 02/06: 没有想法
```Python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        # recursive, same as divide and conquer
        # if p is None and q is None:
        #     return True 
        # if p is not None and q is not None and p.val == q.val:
        #     return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
        # else:
        #     return False

        # iterative
        stack = [(p, q)]
        while stack:
            p, q = stack.pop()
            if p and q and p.val == q.val:
                stack.extend([
                    (p.left, q.left),
                    (p.right, q.right)
                ])
            elif p or q:
                return False 
        return True 
```

[Subtree of Another Tree](https://leetcode.com/problems/subtree-of-another-tree/)
- 02/07: 没有想法 (这里面有kmp 小唐说考了就认命)
```Python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
        # DFS, T: O(mn) n: num nodes in root, m: num nodes in subroot
        # S: O(m+n) max num nodes in stack call
    #     if root is None:
    #         return False
    #     if subRoot is None:
    #         return True
    #     return self.identTree(root.left, subRoot) or self.identTree(root.right, subRoot)

    # def identTree(self, p, q):
    #     if p is None and q is None:
    #         return True 
    #     if p is not None and q is not None and p.val == q.val:
    #         return self.identTree(p.left, q.left) and self.identTree(p.right, q.right)
    #     return False

        # string matching
        # tree serialzation problem, then string matching
        # (node value + structure of tree)-> single traversal is not enough
        # -> two traversals also not enough (inorder and preorder)(inorder and postorder)
        # speical char for null node using one traversal (either preorder or postorder)
        # one more limitation, 2# vs 22##, add one more special char to encode
        # space or ^ char
        # string matching: find() function or KMP algorithm
        def serialize(node, tree_str):
            if node is None:
                tree_str.append('#')
                return 
            tree_str.append('^')
            tree_str.append(str(node.val))
            serialize(node.left, tree_str)
            serialize(node.right, tree_str)
        p_lst, q_lst = [], []
        serialize(root, p_lst)
        serialize(subRoot, q_lst)
        p = ''.join(p_lst)
        q = ''.join(q_lst)
        # if p.find(q) != -1: # O(nm)
        #     return True 
        # else:
        #     return False
        # KMP algorithm T: O(m+n)
```

[Lowest Common Ancestor of a Binary Search Tree](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/description/)
- 02/07: 没有想法
```Python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        # Binary Search Tree: 有特殊性质
        # iterative approach, T: O(n), S: O(1)
        cur = root
        while cur:
            if p.val > cur.val and q.val > cur.val:
                cur = cur.right 
            elif p.val < cur.val and q.val < cur.val:
                cur = cur.left 
            else:
                return cur 
```

[Binary Tree Level Order Traversal](https://leetcode.com/problems/binary-tree-level-order-traversal/)
- 02/07: 没有想法 忘记模板了
```Python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        # iterative, T&S: O(n)
        # from collections import deque
        # if root is None:
        #     return root 
        # queue = deque([root])
        # paths = []
        # while queue:
        #     size = len(queue)
        #     tree_path = []
        #     for i in range(size):
        #         node = queue.popleft()
        #         tree_path.append(node.val)
        #         if node.left:
        #             queue.append(node.left)
        #         if node.right:
        #             queue.append(node.right)
        #     paths.append(tree_path)
        # return paths
        # recursive, T&S: O(n)
        levels = []
        if not root:
            return levels 
        def recursive(node, level):
            if len(levels) == level:
                levels.append([])
            levels[level].append(node.val)
            if node.left:
                recursive(node.left, level+1)
            if node.right:
                recursive(node.right, level+1)
        recursive(root, 0)
        return levels
```

[Validate Binary Search Tree](https://leetcode.com/problems/validate-binary-search-tree/)
- 02/07: 没有想法
```Python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        # divide and conquer, T&S: O(n)
        def isBST(node, low=-float('Inf'), high=float('Inf')):
            if not node:
                return True 
            if node.val <= low or node.val >= high:
                return False 
            return (isBST(node.right, node.val, high) and isBST(node.left, low, node.val))
        return isBST(root)
```

[kth Smallest Element in a BST](https://leetcode.com/problems/kth-smallest-element-in-a-bst/description/)
- 02/07: 没有想法
```Python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        # recursive inorder traversal in BST
        # k-1th element in the array
        # T&S: O(n)
        # def inorder(node):
        #     return inorder(node.left) + [node.val] + inorder(node.right) if node else []
        # return inorder(root)[k-1]
        # iterative inorder traversal
        # T: O(H+k), H: tree height: H, balanced tree: logN + K\
        # completely unbalanced tree O(N+K)
        # S: O(H)-> N ～ logN
        stack = []
        while True:
            while root:
                stack.append(root)
                root = root.left 
            root = stack.pop()
            k -= 1 
            if not k:
                return root.val
            root = root.right
```

[Construct Binary Tree from Preorder and Inorder Traversal](https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)
- 02/07: 没有想法
```Python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        # recursive, T&S: O(n)
        # base case
        if not preorder or not inorder:
            return None
        root = TreeNode(preorder[0])
        mid = inorder.index(preorder[0])
        root.left = self.buildTree(preorder[1:mid+1], inorder[:mid])
        root.right = self.buildTree(preorder[mid+1:], inorder[mid+1:])
        return root
```

# Tries
[Implement Trie Prefix Tree](https://leetcode.com/problems/implement-trie-prefix-tree/)
- 02/08: 没有想法
```Python
class TrieNode:
    def __init__(self):
        self.children = {} # trie node list 
        self.endWord = False

class Trie:
    def __init__(self):
        self.root = TrieNode() # search in O(26)->O(1)

    def insert(self, word: str) -> None:
        cur = self.root 
        for c in word:
            if c not in cur.children:
                cur.children[c] = TrieNode()
            cur = cur.children[c]
        cur.endWord = True
        
    def search(self, word: str) -> bool:
        cur = self.root
        for c in word:
            if c not in cur.children:
                return False
            cur = cur.children[c]
        if cur.endWord:
            return True
        return False       

    def startsWith(self, prefix: str) -> bool:
        cur = self.root 
        for c in prefix:
            if c not in cur.children:
                return False 
            cur = cur.children[c]
        return True      
# Your Trie object will be instantiated and called as such:
# obj = Trie()
# obj.insert(word)
# param_2 = obj.search(word)
# param_3 = obj.startsWith(prefix)
```

[Design Add and Search Words Data Structure](https://leetcode.com/problems/design-add-and-search-words-data-structure/)
- 02/08: 没有想法
```Python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.word = False

class WordDictionary:
    def __init__(self):
        self.root = TrieNode()
        
    def addWord(self, word: str) -> None:
        curr = self.root
        for c in word:
            if c not in curr.children:
                curr.children[c] = TrieNode()
            curr = curr.children[c]
        curr.word = True

    def search(self, word: str) -> bool:
        def dfs(j, root):
            curr = root 
            for i in range(j, len(word)):
                c = word[i]
                if c == '.':
                    for child in curr.children.values():
                        if dfs(i+1, child):
                            return True 
                    return False 
                else:
                    if c not in curr.children:
                        return False
                    curr = curr.children[c]
            return curr.word 
        return dfs(0, self.root)

# Your WordDictionary object will be instantiated and called as such:
# obj = WordDictionary()
# obj.addWord(word)
# param_2 = obj.search(word)
```

# Heap / Priority Queue
[Find Median From Data Stream]()

# Backtracking
[Combination Sum](https://leetcode.com/problems/combination-sum/description/)
- 02/08: 没有想法
```Python
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        # T: 看几个选择 就是几的n次方 N**N
        # S: N**N 也是 因为要记住这些东西
        self.path_lst = []
        self.target = target
        self.candidates = sorted(candidates)
        self.backtracking([],0)
        return self.path_lst
    def backtracking(self, path, start_index): # start_index: can't look back
        if sum(path) == self.target:
            self.path_lst.append(path[:]) #注意了
            return #这时就满足条件 可以不用搜了 除非有复数
        elif sum(path) > self.target:
            return 
        for num in range(start_index, len(self.candidates)):
            path.append(self.candidates[num])
            self.backtracking(path, num)
            path.pop()
```

[Word Search](https://leetcode.com/problems/word-search/description/)
- 02/08: 没有想法
```Python
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        # T&S: O(4**N)
        self.target = [str(i) for i in word]
        self.row, self.col = len(board), len(board[0])
        self.board = board 
        self.exist = False
        for i in range(self.row):
            for j in range(self.col):
                if board[i][j] == self.target[0]:
                    self.backtracking(len(self.target)-1, i, j, {(i,j)})
                if self.exist:
                    return True 
        return False
        
    def backtracking(self, remain, curx, cury, visited):
        if remain == 0:
            self.exist = True 
            return 
        for x, y in [(1,0), (-1,0), (0,1), (0,-1)]:
            newx, newy = curx+x, cury+y 
            if (newx, newy) not in visited and newx>=0 and newx <self.row \
               and newy>=0 and newy<self.col and self.board[newx][newy] == self.target[-remain]:
                visited.add((newx, newy))
                self.backtracking(remain-1, newx, newy, visited)
                visited.remove((newx, newy))
```

# Graphs
[Number of Islands]()

[Clone Graph]()

[Pacific Atlantic Water Flow]()

[Course Schedule]()

[Number of Connected Components In An Undirected Graph]()

# Advanced Graphs
# 1-D Dynamic Programming
- 如果一个问题有很多重叠子问题，使用动态规划是最有效的
- 动态规划中每一个状态一定是由上一个状态推导出来的，这一点就区别于贪心，贪心没有状态推导，而是直接从局部直接选最优的 （每一次拿最大的就行，和上一个状态无关）
- 动态规划的解题步骤： 动规五部曲
1. 确定dp数组(dp table)以及下标的含义
2. 确定递推公式
3. dp数组如何初始化
4. 确定遍历顺序
5. 举例推导dp数组

[Fibonacci Number](https://leetcode.com/problems/fibonacci-number/)
- 03/13: 第一次学dp 这个解法很简单 但是思想很深刻
```Python
class Solution:
    def fib(self, n: int) -> int:
    # 4. DP: iterative bottom-up, T O(n), S O(1)
    # T each value from 2 to n is computed once
    # S 3 units of space to store the computed values for every loop iteration
    # only needed to look at resulys of fib(n-1) and fib(n-2) to determine fib(n)
        if n <= 1:
            return n
        # prev1:fib(n-1) prev2:fib(n-2)
        current, prev1, prev2 = 0, 1, 0
        for i in range(2, n+1):
            current = prev1 + prev2
            temp = prev1
            prev2 = prev1
            prev1 = current
        return current
    # 3. top-down approach recursion using memoization, T O(n) S O(n)
    # use memoization to store the pre-computed answer, then return the answer for n
    # we will leverage recursion, but in a smarter way by not repeating the work to 
    # calculate existing values
    # 这个方法好像有点问题 速度不在一个级别 而且为什么要加self不是很理解
        # self.memo = {}
        # self.memo[0], self.memo[1] = 0, 1
        # # at every recursive call of fib(n), if n exists in the map, return the cached value for n
        # if n in self.memo:
        #     return self.memo[n]
        # else:
        #     self.memo[n] = self.fib(n-1) + self.fib(n-2)
        #     return self.memo[n] 
    # 2. DP: Bottom-up approach iteration using memoization, T O(n) S O(n)
    # improve by using iteration, still solving for all of the sub-problems and return the answer for n
    # using already computed Fibonacci values. while using a bottom-up appraoch, we can iteratively and 
    # store the values, only returning once we reach the result
        # memo = {}
        # memo[0], memo[1] = 0, 1
        # for i in range(2, n+1):
        #     memo[i] = memo[i-1] + memo[i-2]
        # return memo[n]
    # 1. Recursion, T O(2^n) depth of recursive tree; S O(n) stack
    # it has the potential to be bad in cases that there isn't enough physical memory
    # to handle the increasingly growing stack, leading to a StackOverflowError
        # if n <= 1:
        #     return n
        # else:
        #     return self.fib(n-1) + self.fib(n-2)
```

[Climbing Stairs](https://leetcode.com/problems/climbing-stairs/)
- 03/13: recursion方法 在等号左边不能用这个自调用 但是为什么呢
```Python
class Solution:
    # def __init__(self):
    #     self.memo = {0:0, 1:1, 2:2}
    def climbStairs(self, n: int) -> int:
        # 4. iterative bottom up, T O(n) S O(1)
        if n <= 2:
            return n
        curr, prev1, prev2 = 0, 2, 1
        for i in range(3, n+1):
            curr = prev1 + prev2
            temp = prev1
            prev1 = curr
            prev2 = temp
        return curr
        #3. top-down recursion approach with memoization, T O(n) S O(n)
        # if n in self.memo:
        #     return self.memo[n]
        # else:
        #     self.memo[n] = self.climbStairs(n-1) + self.climbStairs(n-2)
        #     return self.memo[n]
        #2. bottom-up iteration approach with memoization, T O(n) S O(n)
        # memo = {}
        # memo[0], memo[1], memo[2] = 0, 1, 2
        # for i in range(3, n+1):
        #     memo[i] = memo[i-1] + memo[i-2]
        # return memo[n] 
        # climb(n) = climb(n-1) + climb(n-2)
        # 1. recursion, T O(2^n) S O(n)
        # if n <= 2:
        #     return n
        # else:
        #     return self.climbStairs(n-1) + self.climbStairs(n-2)
```

[House Robber](https://leetcode.com/problems/house-robber/)
- 03/21: dp在想法还还是有步骤落下，optimized的临界条件有错
```Python
class Solution:
    def rob(self, nums: List[int]) -> int:
        # optimized DP
        # T n, S 1
        if len(nums) == 0:
            return 0
        if len(nums) == 1:
            return nums[0]
        prev1, prev2 = nums[0], max(nums[0], nums[1])
        current = max(prev1, prev2) #input [1,1]时会报错
        for i in range(2, len(nums)):
            current = max(prev1+nums[i], prev2)
            prev1 = prev2 
            prev2 = current
        return current

        # 1. dp[i] maximum amount of money can get until the ith
        # 2. dp[i] = max(dp[i-1], dp[i-2]+nums[i])
        # 3. dp[0] = nums[0], dp[1] = max(nums[0], nums[1])
        # 4. 从前到后
        # 5. 
        # T n, S n 
        # if len(nums) == 0:
        #     return 0
        # if len(nums) == 1:
        #     return nums[0]
        # dp = [0] * len(nums)
        # dp[0], dp[1] = nums[0], max(nums[0], nums[1]) #有edge case
        # for i in range(2, len(nums)):
        #     dp[i] = max(dp[i-1], dp[i-2]+nums[i])
        # return dp[-1]
```

[hourse robber ii](https://leetcode.com/problems/house-robber-ii/description/)
- 03/22
```Python
class Solution:
    def rob(self, nums: List[int]) -> int:
        if len(nums) == 0:
            return 0
        elif len(nums) == 1:
            return nums[0]
        elif len(nums) == 2:
            return max(nums[0], nums[1])
        else:
            return max(self.rob_simple(nums[1:]), self.rob_simple(nums[:-1]))
        
    def rob_simple(self, nums):
        prev1 = nums[0]
        prev2 = max(nums[0], nums[1])
        for i in range(2, len(nums)):
            temp = prev2
            prev2 = max(nums[i]+prev1, prev2)
            prev1 = temp
        return prev2
```
[longest palindromic substring](https://leetcode.com/problems/longest-palindromic-substring/description/)
- 03/22: 很难的感觉 brute force也写不出来, 二维的dp解法之后学到二维要自己写，这个code属于是被模版的 要么我记住模版记住题型 什么东西都往上套 要么我就需要活学活用 这样的话很多boundary case都要临时考虑 考验功底 就需要平时自己都写出答案来联系 看解答是不行的 只是记住了
```Python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        # brute force: pick all possible starting and ending positions for a substring
        # and then verify if it is a panlindrome
        # T (n^3) check all possible substring of s (two nested loops), and then checking if each substring is a palindrome (one additional loop)
        # S (1) only use a constant amount of extra space to store the longest palindrome so far 
        # if len(s) < 2:
        #     return s 
        # longest_palindrome = s[0] # default case应该是第一个字母
        # for i in range(len(s)):
        #     for j in range(i+1, len(s)):
        #         substring = s[i:j+1]
        #         if substring == substring[::-1] and len(substring) > len(longest_palindrome):
        #             longest_palindrome = substring
        # return longest_palindrome
        # dp， T（n^2）, S(n^2)
        # 1. dp(i,j), whether s[i:j+1] is palindrome 
        # 2. dp(i,j) = dp(i+1, j-1) and s[i] == s[j]
        # 3. base cases
        # another way of checking, traverse each character as the center, looking for expansion whether
        # T (n^2)
        max_len, max_pal = 0, ''
        for i in range(len(s)):
            # odds
            l, r = i, i
            while l >= 0 and r <= len(s) -1 and s[l] == s[r]:
                if (r-l+1) > max_len:
                    max_len = r-l+1 
                    max_pal = s[l:r+1]
                l -= 1
                r += 1
            # even
            l, r = i, i+1
            while l >= 0 and r <= len(s) -1 and s[l] == s[r]:
                if (r-l+1) > max_len:
                    max_len = r-l+1
                    max_pal = s[l:r+1]
                l -= 1
                r += 1
        return max_pal
```

[palinfromic substrings](https://leetcode.com/problems/palindromic-substrings/description/)
- 03/22: 写二维dp要回来看全部的解法
```Python
class Solution:
    def countSubstrings(self, s: str) -> int:
        # traverse each character, then expand to check if it is palindrom
        # T n^2
        count = 0
        for i in range(len(s)):
            if len(s) == 0:
                return count
            # odds
            l, r = i, i
            while l >= 0 and r <= len(s)-1 and s[l] == s[r]: # trick to fulfill the bounardy issue
                count += 1
                l -= 1
                r += 1
            # evens
            if len(s) >= 2:
                l, r = i, i+1
                while l >= 0 and r<= len(s)-1 and s[l] == s[r]:
                    count += 1
                    l -= 1
                    r += 1
        return count 
```
[word wrap]()
- 03/15
```Python

```

# 2-D Dynamic Programming
# Greedy
# Intervals
[merge intervals](https://leetcode.com/problems/merge-intervals/)
- 03/15: given an array of meeting time intervals where intervals[i] = [starti, endi], merge all overlapping intervals, and return an array of the non-overlapping intervals that cover all the intervals in the input. 加一个是与之前相比没有交叉，但是说不定与后面一个怎么样 所以都要用merged的-1与waiting的那个相互比较
- Is the space O(N) since you use a new list to keep the result? We donot take results space into account. You can always check with interviewer if he want to consider the space for result.
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

[meeting room](https://leetcode.com/problems/meeting-rooms/)
- given an array of meeting time intervals where intervals[i] = [starti, endi], determine if a person could attend all meetings
- 03/15
```Python
class Solution:
    def canAttendMeetings(self, intervals: List[List[int]]) -> bool:
        intervals.sort(key=lambda x: x[0]) #怎么sort开头我忘记了
        for i in range(1, len(intervals)):
            if intervals[i-1][1] > intervals[i][0]:
                return False
        return True 
```

[meeting room II](https://leetcode.com/problems/meeting-rooms-ii/)
- 03/15:
- given an array of meeting time intervals where intervals[i] = [starti, endi], return the minimum number of conference rooms required
```Python
class Solution:
    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        # mini num of conference rooms = maximum overlap meeting given any time
        start_times = [interval[0] for interval in sorted(intervals,key=lambda x:x[0])]
        end_times = [interval[1] for interval in sorted(intervals, key=lambda x:x[1])]
        count = 0
        max_count = 0
        s, e = 0, 0
        while s < len(intervals) and e < len(intervals):
            if start_times[s] < end_times[e]:
                count += 1
                s += 1
            else:
                count -= 1
                e += 1
            max_count = max(max_count, count)
        return max_count
```

[Insert Interval](https://leetcode.com/problems/insert-interval/)
- 03/20: 没有熟练掌握
```Python
class Solution:
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        # T n, S 1
        # insert the newInterval by starting pointer
        # then reuse the merge interval code
        i = 0 
        while i < len(intervals) and intervals[i][0] < newInterval[0]:
            i += 1 
        intervals.insert(i, newInterval)
        res = []
        for cur_inter in intervals:
            if len(res) == 0 or res[-1][1] < cur_inter[0]:
                res.append(cur_inter)
            else:
                res[-1][1] = max(res[-1][1], cur_inter[1])
        return res

        # T n, S n
        # intervals can be divide into 3 parts,
        # no overlap in the left; no overlap in the right; overlap in between
        # start, end = 0, 1 # constant helping us to access the starting and end point
        # s, e = newInterval[start], newInterval[end]
        # left, right = [], []
        # for cur_interval in intervals:
        #     if cur_interval[end] < s:
        #         left.append(cur_interval)
        #     elif cur_interval[start] > e:
        #         right.append(cur_interval)
        #     else:
        #         s = min(s, cur_interval[start])
        #         e = max(e, cur_interval[end])
        # return left + [[s, e]] + right
```


# Math & Geometry

# Bit Manipulation

# Greedy - 代码随想录
- 核心： 当通过局部最优我们可以得到整体最优的时候
- 如何验证这个核心：靠自己手动模拟，如果模拟可行，就试一试贪心策略，如果不可行，可能需要动态规划
- 最好的的实际策略：举反例，如果想不到反例，就试一试贪心策略

[Assign Cookies](https://leetcode.com/problems/assign-cookies/)
- 03/21: 多考虑下什么题目可以greedy解答，以及这里为了避免out of boundary的trick
```Python
class Solution:
    def findContentChildren(self, g: List[int], s: List[int]) -> int:
        # 为了满足更多的小孩，就不要造成饼干尺寸的浪费
        # 大尺寸的饼干既可以满足胃口大的孩子也可以满足胃口小的孩子，那么就应该优先满足胃口大的
        # 局部最优大饼干喂给胃口大的，充分利用饼干尺寸喂饱一个，全局最优就是喂饱尽可能多的小孩
        g.sort(reverse=True)
        s.sort(reverse=True)
        j, res = 0, 0
        for i in range(len(g)):
            if j <= len(s)-1 and g[i] <= s[j]: # add ahead to aviod out of boundary
                res += 1
                j += 1 
        return res
```

[Maximum Subarray](https://leetcode.com/problems/maximum-subarray/)
- 03/21: 需要多想想
```Python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        # brute force, T n^2, S 1
        # max_res = -float('inf')
        # for i in range(len(nums)):
        #     curr_subarray = 0
        #     for j in range(i, len(nums)): # 这里不是i+1呢 需要考虑
        #         curr_subarray += nums[j]
        #         max_res = max(max_res, curr_subarray)
        # return max_res
        # greedy: 当开始时为负数 那么直接抛弃 换下一个为正的起点
        # 相当于暴力解法中不断调整最大子区间的起始位置 T n, S 1
        # 贪心理论很直白 但是实际并不好想 不要小看贪心
        # curr_max, res_max = 0, -float('inf')
        # for i in range(len(nums)):
        #     curr_max += nums[i]
        #     if curr_max > res_max:
        #         res_max = curr_max
        #     if curr_max < 0:
        #         curr_max = 0 
        # return res_max
        # dp: T n, S n
        # 动规五部曲： 
        # 1. 确定dp数组以及下表含义: dp[i] 包括下标i之前的最大连续子序列和为dp[i]
        # 2. 确定递推公式: dp[i]只有两个方向可以推出来, nums[i]加入当前连续子序列和, 从头开始计算当前连续子序列和
        # 3. dp数组如何初始化: 从递推公式可以看出来dp[i]是依赖于dp[i - 1]的状态，dp[0]就是递推公式的基础。根据dp[i]的定义，很明显dp[0]应为nums[0]即dp[0] = nums[0]
        # 4. 确定遍历顺序: 递推公式中dp[i]依赖于dp[i - 1]的状态，需要从前向后遍历。
        # 5. 举例推导dp数组
        if len(nums) == 0:
            return 0
        dp = [0] * len(nums)
        dp[0] = nums[0]
        result = dp[0]
        for i in range(1, len(nums)):
            dp[i] = max(dp[i-1]+nums[i], nums[i]) #状态转移公式
            result = max(result, dp[i]) #保存dp[i]最大的值
        return result

        # maxSum = currenSum = nums[0]
        # for i in range(1,len(nums)):
        #     currenSum = max(nums[i],currenSum+nums[i])
        #     maxSum = max(maxSum, currenSum)
        # return maxSum
```