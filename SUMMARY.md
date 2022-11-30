# Table of contents

## Basic

Big O notation:
- O(1): constant time
- O(log n): logirithmic time
- O(N): linear time
- O(NlogN): log-linear time
- O(N^2): quadratic time
- O(2^N): expoential time
- O(N!): factorial time

Average Time Complexity for Basic Data Structure:
|Data Structure|Access|Search|Insertation|Deletion|
|--------------|--------------|--------------|--------------|--------------|
|Array|O(1)|O(N)|O(N)|O(N)|
|Linked List|O(N)|O(N)|O(1)|O(1)|
|Hash Map| N/A| O(1)[O(N) in worst case]|O(1)[O(N) in worst case]|O(1)[O(N) in worst case]|

## Day1
Question: [704 binary search](https://leetcode.com/problems/binary-search/description/)  
Outcome with Date: 11-23:X  
First Impression: I know need to use left, right, mid pointers but I don't know how to set the stop criteria in the loop  
Good Video/Blog:  https://juejin.cn/post/7145742873009324040 https://www.bilibili.com/video/BV1fA4y1o715/?vd_source=8b4794944ae27d265c752edb598636de  
Difficulty during Implementation: closed interval? open interval? need to consider in the questions, otherwise will have some erros -> once decide the definition of interval, all the parameters regarding the interval should follow the definition. 
Logic of Solution:  
1. definitions of left and right pointers
2. while the left < right or left <= right:
3. search for target and update the left or right pointers
4. return result
AC Code:  
```Python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left = 0
        right = len(nums)-1
        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] < target:
                left = mid+1
            elif nums[mid] > target:
                right = mid-1
            else:
                return mid
        return -1
```

Question: [35 Search Insert Position](https://leetcode.com/problems/search-insert-position/)  
Outcome with Date: 11-23:X  
First Impression: don't know how to use binary search to implement, and it seems if index out of boundary or within the array are different cases  
Good Video/Blog: https://www.bilibili.com/video/BV1dA411a7CB/?spm_id_from=333.337.search-card.all.click&vd_source=8b4794944ae27d265c752edb598636de https://www.bilibili.com/video/BV13i4y197w2/?spm_id_from=333.999.0.0&vd_source=8b4794944ae27d265c752edb598636de  
Learnt: the objective changes to find the index in the array which is the most clostest to the target but bigger than target  
Difficulty during Implementation: why return left, and why gets error when return idex is 0  
Logic of Solution: the same as binary select exclude the last step  
Need Help: Don't know why we need to do the == condition flow first, otherwise, some cases can't pass  
AC Code:  
```Python
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        left = 0
        right = len(nums)-1
        while left <= right:
            mid = left + (right - left) //2
            if nums[mid] == target:
                return mid 
            elif nums[mid] < target:
                left = mid+1
            else:
                right = mid-1
        return left
            
```
## Day2
Question: [34 Find First and Last Position of Element in Sorted Array](https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/)  
Outcome with Date: 11-24:X  
First Impression:know that I can apply binary search twice by finding the target-1, and target+1 -> can't work, also the same thing will happen in target-1 and target+1->wrong  
Good Video/Blog:https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/solutions/1136731/find-first-and-last-position-of-element-in-sorted-array/  
Learnt:
Difficulty during Implementation:  
Logic of Solution: 
AC Code:  (need help!!!) 

Question: [27 Remove Element](https://leetcode.com/problems/remove-element/)  
Outcome with Date: 11-24:X  
First Impression: no idea  
Good Video/Blog: https://www.bilibili.com/video/BV12A4y1Z7LP/?vd_source=8b4794944ae27d265c752edb598636de  
Learnt: use fast and slow pointers. fast pointer will update at everystep by 1, for slow pointer only update when it are not the target value. 覆盖, 清楚知道fast,slow pointers定义  
Difficulty during Implementation: definitation of fast and slow pointers need to be clear  
Logic of Solution:  
1. iterate the faster pointer one by one
2. assign the nums[slow] = nums[fast] only when nums[fast] not equal to target
3. return slow  
AC Code:
```Python
class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        fast, slow = 0,0
        for fast in range(len(nums)):
            if nums[fast] != val:
                nums[slow] = nums[fast]
                slow += 1
        return slow
```

## Day3
Question: [977 Squares of a Sorted Array](https://leetcode.com/problems/squares-of-a-sorted-array/)  
Outcome with Date: 11-25:O  
First Impression: have idea that I can apply two pointers in left and right to compare each time to find the bigger one, but has some error  
GoodVideo/Blog:https://programmercarl.com/0977.%E6%9C%89%E5%BA%8F%E6%95%B0%E7%BB%84%E7%9A%84%E5%B9%B3%E6%96%B9.html#%E5%85%B6%E4%BB%96%E8%AF%AD%E8%A8%80%E7%89%88%E6%9C%AC  
Learnt: make a same size array can't use directly equal such as new_arr = arr, the right approach is new_arr = [-1]*len(arr)  
Difficulty during Implementation: none  
Logic of Solution:
1. two pointers while they didn't meet
2. assign the larger one into the end of the new array
AC Code:
```Python
class Solution:
    def sortedSquares(self, nums: List[int]) -> List[int]:
        left, right = 0, len(nums)-1
        new = [-1] * len(nums)
        new_ind = len(nums)-1
        while left <= right:
            left_sq = nums[left]**2
            right_sq = nums[right]**2
            if left_sq <= right_sq:
                new[new_ind] = right_sq
                right -= 1
            else:
                new[new_ind] = left_sq
                left += 1
            new_ind = new_ind - 1
        return new
```

Question: [209 Minimum Size Subarray Sum](https://leetcode.com/problems/minimum-size-subarray-sum/)
Outcome with Date: 11-25:X  
First Impression: don't know how to do  
Good Video/Blog: https://www.bilibili.com/video/BV1tZ4y1q7XE/?vd_source=8b4794944ae27d265c752edb598636de  
Learnt: key idea is two pointers, and we fix the right pointer (for loop) and slide the left pointer  
Difficulty during Implementation: the order of codes is very important, what will be change which can't put in the latter; float("inf")  
Logic of Solution: many similar questions need to do afterwards!  
AC Code:
```Python
class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        sum_res = 0
        res = float("inf")
        left = 0
        for right in range(len(nums)):
            sum_res += nums[right]
            while sum_res >= target:
                res = min(res, right-left+1)
                sum_res -= nums[left]
                left += 1
        return 0 if res==float("inf") else res
```

Question: [59 Spiral Matrix II](https://leetcode.com/problems/spiral-matrix-ii/)  
Outcome with Date: 11-25:X  
First Impression: no idea  
Good Video/Blog: https://www.bilibili.com/video/BV1SL4y1N7mV/?vd_source=8b4794944ae27d265c752edb598636de https://blog.csdn.net/PolyCozy/article/details/126990506?spm=1001.2014.3001.5501  
Learnt: 1. initiate n array: (a) [0]* n, (b) [0 for _ in range(n)], 2. initiate 2D nn matrix: nums = [[0]* n for _ in range(n)] 3.使用range()函数逆序遍历 for i in range(start, end - 1, -1)  
Difficulty during Implementation: 1. initiate 2D nn matrix: nums = [[0]* n for _ in range(n)]， 2.how to create the loop function -> num++ at the end it shoud be n*n numbers， 3.停顿在区间定义 要不要equal呢,同理奇数的时候应该怎么办->还是不知道  
Logic of Solution: (need help!!!)  
AC Code:  (need help!!!代码有误)  
```Python
class Solution:
    def generateMatrix(self, n: int) -> List[List[int]]:
        nums = [[0] * n for _ in range(n)]
        upperbound, lowerbound = 0, n-1
        leftbound, rightbound = 0, n-1
        num = 1
        while num <= n*n:
            if upperbound <= lowerbound: 
                for j in range(leftbound, rightbound+1):
                    nums[upperbound][j] = num
                    num += 1
                upperbound += 1
            if leftbound <= rightbound: 
                for i in range(upperbound, lowerbound+1):
                    nums[i][rightbound] = num
                    num += 1
                rightbound -= 1
            if lowerbound >= upperbound:
                for j in range(rightbound, leftbound-1, -1):
                    nums[lowerbound][j] = num
                    num += 1
                lowerbound -= 1
            if leftbound <= rightbound:
                for i in range(lowerbound, upperbound-1,-1):
                    nums[i][leftbound] = num
                    num += 1
                leftbound += 1
            return nums
```
## Day4 

### Summarization for array
https://programmercarl.com/%E6%95%B0%E7%BB%84%E6%80%BB%E7%BB%93%E7%AF%87.html  
1. 经典题型： 二分法，双指针，滑动窗口，模拟行为  
2. 二分法：o(logn) 循环不变量loop invariant(只有这个条件为假，我们才跳出这个循环)注意区间的定义，保持这个定义，保持区间合法性  
3. 双指针：o(n)通过一个快指针和慢指针在一个for循环下完成两个for循环的工作  
4. 滑动窗口：o(n)主要要理解滑动窗口如何移动 窗口起始位置，达到动态更新窗口大小的，从而得出长度最小的符合条件的长度  
5. 模拟行为：相信大家有遇到过这种情况： 感觉题目的边界调节超多，一波接着一波的判断，找边界，拆了东墙补西墙，好不容易运行通过了，代码写的十分冗余，毫无章法，其实真正解决题目的代码都是简洁的，或者有原则性的，大家可以在这道题目中体会到这一点  
6. while 循环不变量  
7. backward iterate in loop: range(end, start-1, -1)  

### linked list
https://programmercarl.com/%E9%93%BE%E8%A1%A8%E7%90%86%E8%AE%BA%E5%9F%BA%E7%A1%80.html  
链表是一种通过指针串联在一起的线性结构，每一个节点由两部分组成，一个是数据域一个是指针域（存放指向下一个节点的指针），最后一个节点的指针域指向null（空指针的意思）。链表的入口节点称为链表的头结点也就是head。

Question: [203 Remove Linked List Elements](https://leetcode.com/problems/remove-linked-list-elements/)  
Outcome with Date: 11-27:X  
First Impression: have some idea but don't know how to iterate(stopping criteria)  
Good Video/Blog:
Learnt: 1.delete a head node is different than others 2.the linked list can't access by index, every operation need to start with head node    
Difficulty during Implementation: (need to try again)
Logic of Solution:  
1. create dummy head, in this way we don't need additional codes to delete head node
2. while didn't iterate the end of the linkedlist
3. if find the value, delete it
4. if not, pass to the next node
5. return head
AC Code:
```Python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def removeElements(self, head: Optional[ListNode], val: int) -> Optional[ListNode]:
        dummyhead = ListNode(next=head)
        cur = dummyhead
        while(cur.next != None):
            if(cur.next.val == val):
                cur.next = cur.next.next
            else:
                cur = cur.next
        return dummyhead.next
```

Question: [707 Design Linked List](https://leetcode.com/problems/design-linked-list/)  
Outcome with Date: 11-27: X  
First Impression: 总体上看上去不难，但是init那里不知道怎么写，看了下答案  
Good Video/Blog: https://www.bilibili.com/video/BV1FU4y1X7WD/?vd_source=8b4794944ae27d265c752edb598636de  
Learnt: 问题比想象的多，比如需要dummy head这样不用分类讨论(而且加在哪)，其次插入节点的时候要注意顺序  
Difficulty during Implementation: 1.while循环不变量用什么还是很困难的 特别是临界值 需要靠edge case来判定2.dummy head放在哪里    
Logic of Solution:
AC Code:
```Python
class Node:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class MyLinkedList:

    def __init__(self, size=0):
        self.head = Node()
        self.size = size 
    
    def get(self, index: int) -> int:
        if index<0 or index>=self.size:
            return -1 
        else:
            curr = self.head.next
            while(index>0):
                curr = curr.next
                index -= 1
            return curr.val

    def addAtHead(self, val: int) -> None:
        new_node = Node(val)
        new_node.next = self.head.next
        self.head.next = new_node
        self.size += 1
        return 

    def addAtTail(self, val: int) -> None:
        new_node = Node(val)
        curr = self.head
        while(curr.next!=None):
            curr = curr.next
        curr.next = new_node
        self.size += 1
        return 

    def addAtIndex(self, index: int, val: int) -> None:
        new_node = Node(val)
        if index<0 or index>=self.size+1:
            return 
        else:
            curr = self.head
            while(index>0):
                curr = curr.next
                index -= 1
            new_node.next = curr.next
            curr.next = new_node
            self.size += 1
            return
        
    def deleteAtIndex(self, index: int) -> None:
        if index<0 or index>=self.size:
            return 
        else:
            curr = self.head
            while(index):
                curr = curr.next
                index -= 1
            curr.next = curr.next.next
            self.size -= 1
            return
```

## Day 5
Question: [206 Reverse Linked List](https://leetcode.com/problems/reverse-linked-list/)  
Outcome with Date: 11-27:X  
First Impression:还是被搞晕了 看了视频才清楚  
Good Video/Blog:https://www.bilibili.com/video/BV1nB4y1i7eL/  
Learnt: Two methods (1) two pointers, (2) recursion; when to stop in the while loop->draw a illustration plot, then I can know
Difficulty during Implementation: (1) two pointers 交换混了 exceed time limit ->一个递归单元就是pre和curr其他的再下一个递归单元也会碰到  
Logic of Solution: recuerse
1. recurse helper function
2. parameters for the helper funciton -> what do we need in the initalization
3. basic case -> when we stop the recursive function and return the output we want
4. call itself -> update on the parameters 
AC Code:
```Python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        pre = None
        curr = head
        while(curr!=None):
            temp = curr.next
            curr.next = pre
            pre = curr
            curr = temp
        return pre
```
```Python
class Solution:
    def reverse(self, pre, curr):
        if (curr == None): 
            return pre
        temp = curr.next
        curr.next = pre
        return self.reverse(curr, temp)

    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        return self.reverse(None, head)
```

Question: [24 Swap Nodes in Pairs](https://leetcode.com/problems/swap-nodes-in-pairs/)  
Outcome with Date: 11-28:X  
First Impression:有点想法 但是自己的想法的分类条件有点多  
Good Video/Blog: https://www.bilibili.com/video/BV1YT411g7br/?vd_source=8b4794944ae27d265c752edb598636de  
Learnt:逻辑十分清楚，希望自己下一次不看视频就可以写出这个逻辑链  
Difficulty during Implementation: curr应该怎么更新，为什么是像视频里所说的  
Logic of Solution:  
AC Code:(need help!!!)  

Question: [19 remove nth node from the end of lis](https://leetcode.com/problems/remove-nth-node-from-end-of-list/)  
Outcome with Date: 11-28:X  
First Impression:题目看上去很简单 但是他是从后往前数 还是不知道终止条件  
Good Video/Blog: https://www.bilibili.com/video/BV1vW4y1U7Gf/?vd_source=8b4794944ae27d265c752edb598636de  
Learnt: 使用fast and slow pointer, 先移动faster pointer n+1 步，再同时一步步移动  
Difficulty during Implementation: fast先移动几步，让他移动几步的while循环条件又怎么写  
Logic of Solution:
AC Code:
```Python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        dummy_head = ListNode(0, next=head)
        fast, slow = dummy_head, dummy_head
        while(n!=0 and fast.next!=None):
            fast = fast.next 
            n -= 1
        while(fast.next!=None):
            fast = fast.next 
            slow = slow.next 
        slow.next = slow.next.next
        return dummy_head.next
```
Question: [160 Intersection of two linked list](https://leetcode.com/problems/intersection-of-two-linked-lists/)  
Outcome with Date: 11-28: X  
First Impression:以为很简单 但是没有注意到两个指针同时移动一步会没有用  
Good Video/Blog:https://programmercarl.com/%E9%9D%A2%E8%AF%95%E9%A2%9802.07.%E9%93%BE%E8%A1%A8%E7%9B%B8%E4%BA%A4.html  
Learnt: 先移动长的指针那个的差值 再一起一步步移动  
Difficulty during Implementation: 还是没有处理好空指针的问题 以及最后一个循环的return那句的顺序需要考量一下 自己给自己一些edge case看看能不能发现问题  
Logic of Solution:
AC Code:
```Python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        pointA, pointB = headA, headB
        tempA, tempB = headA, headB
        lenA, lenB = 1, 1
        while(tempA.next!=None):
            lenA += 1
            tempA = tempA.next 
        while(tempB.next!=None):
            lenB += 1
            tempB = tempB.next 
        if lenA > lenB: #让b成为长的那条
            pointA, pointB = pointB, pointA
            lenA, lenB = lenB, lenA 
        for _ in range(lenB - lenA):
            pointB = pointB.next
        while(pointA!=None and pointB!=None):
            if pointA == pointB:
                return pointA 
            pointA = pointA.next
            pointB = pointB.next 
        return None
```

Question: [142 Linked List Cycle II](https://leetcode.com/problems/linked-list-cycle-ii/)  
Outcome with Date: 11-28:X  
First Impression: 用fast快一步去写 但是超过运行时间了  
Good Video/Blog:https://www.bilibili.com/video/BV1if4y1d7ob/?vd_source=8b4794944ae27d265c752edb598636de
Learnt:之前想法是错的 
Difficulty during Implementation:
Logic of Solution:
AC Code:

## Day 6
### Summarization for linkedlist

### Hash Table
1. Hash Table哈希表=hash table，store the data in key-value pairs for faster access on the keys
2. it could be possible that different keys map to the same hash. in that scenario, we would use the collision resolution techniques. For example, two keys have the same hash codes. Then the collision resolution technique is used
3. Hash Collision哈希碰撞，在同一索引想放多个值解决办法：拉链法（储存在链表中），线性探测法（冲突放在哈希表的空位上）
4. 当使用哈希法来解决问题时，我们一般会选择如下三种数据结构（数组，set集合，map映射）
5. 。。。一些看不懂的底层逻辑
6. hashing is an efficient way to sort key values in memory, which is used to get the values associated with the key very fast
7. in general, the key can be of any type, they don't need to be a sequential integer. to handle this, a hash function is used that takes a key and converts it to a more efficient key. this new key can be stored in a sorted way and information is extracted in O(1) time.
8. https://www.educative.io/courses/data-science-interview-handbook/N8MnNQ13oEz

Question: [242 Valid Anagram](https://leetcode.com/problems/valid-anagram/)  
Outcome with Date: 11-29:Y  
First Impression:  
Good Video/Blog:  
Learnt:  
Difficulty during Implementation:  
Logic of Solution:  
AC Code:
```Python
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        dct_s = {} # key: character, value: num
        for ind_s in range(len(s)):
            char_s = s[ind_s]
            if char_s not in dct_s.keys():
                dct_s[char_s] = 1
            else:
                dct_s[char_s] += 1
        for ind_t in range(len(t)):
            char_t = t[ind_t]
            if char_t not in dct_s.keys():
                return False
            elif char_t in dct_s.keys() and dct_s[char_t]>1:
                dct_s[char_t] -= 1
            elif char_t in dct_s.keys() and dct_s[char_t]==1:
                dct_s.pop(char_t)
        if len(dct_s)==0:
            return True 
        return False
```
```Python
#更清楚的解答
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        record = [0] * 26
        for i in s:
            #并不需要记住字符a的ASCII，只要求出一个相对数值就可以了
            record[ord(i) - ord("a")] += 1
        for i in t:
            record[ord(i) - ord("a")] -= 1
        for i in range(26):
            if record[i] != 0:
                #record数组如果有的元素不为零0，说明字符串s和t 一定是谁多了字符或者谁少了字符。
                return False
        return True
```
```Python更清楚的解答
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        from collections import defaultdict
        
        s_dict = defaultdict(int)
        t_dict = defaultdict(int)

        for x in s:
            s_dict[x] += 1
        
        for x in t:
            t_dict[x] += 1

        return s_dict == t_dict
```
```Python更清楚的解答
class Solution(object):
    def isAnagram(self, s: str, t: str) -> bool:
        from collections import Counter
        a_count = Counter(s)
        b_count = Counter(t)
        return a_count == b_count
```
Question: [349 Intersection of Two Arrays](https://leetcode.com/problems/intersection-of-two-arrays/)  
Outcome with Date: 11-28:Y  
First Impression:  
Good Video/Blog:https://programmercarl.com/0349.%E4%B8%A4%E4%B8%AA%E6%95%B0%E7%BB%84%E7%9A%84%E4%BA%A4%E9%9B%86.html#%E6%80%9D%E8%B7%AF  
Learnt:使用set不仅占用空间比数组大，而且速度要比数组慢，并且set把数组映射到key上都要做hash计算的，这个耗时很大    
Difficulty during Implementation:  
Logic of Solution:  
AC Code:
```Python
class Solution:
    def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
        if len(nums2) > len(nums1):
            return set(nums2).intersection(set(nums1))
        else:
            return set(nums1).intersection(set(nums2))
```
```Python
#更好的方法
class Solution:
    def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
        val_dict = {}
        ans = []
        for num in nums1:
            val_dict[num] = 1

        for num in nums2:
            if num in val_dict.keys() and val_dict[num] == 1:
                ans.append(num)
                val_dict[num] = 0
        
        return ans
```

Question: [202 Happy Number](https://leetcode.com/problems/happy-number/)  
Outcome with Date: 11-29:X  
First Impression:没有想法  
Good Video/Blog: https://programmercarl.com/0202.%E5%BF%AB%E4%B9%90%E6%95%B0.html   
Learnt: 这里说了会无限循环-》在求和过程中，sum会重复出现-〉在哈希表里面我们说了，当遇到了要快速判断一个元素是否出现集合里的时候就要考虑哈希法了  
Difficulty during Implementation:  (1)计算各个位数之和 记得用while(n!=0),(2)set加一个数是add不是append  
Logic of Solution:(need help!!!)  
AC Code:  

Question: [1 two sum](https://leetcode.com/problems/two-sum/)  
Outcome with Date: 11-29: X  
First Impression:  
Good Blog: https://programmercarl.com/0001.%E4%B8%A4%E6%95%B0%E4%B9%8B%E5%92%8C.html#%E5%85%B6%E4%BB%96%E8%AF%AD%E8%A8%80%E7%89%88%E6%9C%AC  
Learnt: 之前疑惑在为什么可以同时建立哈希表和同时搜索找target-num,看了如上链接的图知道了即使搜索pair中的第一个数还没有存入第二个数，但是当搜索第二个数的时候一定就能返回第一个数  
Difficulty during Implementation:  
Logic of Solution:  
AC Code:  
```Python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        records = dict()

        for index, value in enumerate(nums):  
            if target - value in records:   # 遍历当前元素，并在map中寻找是否有匹配的key
                return [records[target- value], index]
            records[value] = index    # 遍历当前元素，并在map中寻找是否有匹配的key
        return []
```

Question: [454 4SUMII](https://leetcode.com/problems/4sum-ii/)  
Outcome with Date: 11-29:X  
First Impression: no idea  
Good Video/Blog:https://www.bilibili.com/video/BV1Md4y1Q7Yh/  
Learnt:（1）题目看错了没有对下标有要求（2）暴力是n^4,这里可以把a+b看出一个dict,去寻找-c-d  
Difficulty during Implementation: 看到思路就写出来了  
Logic of Solution:
AC Code:
```Python
class Solution:
    def fourSumCount(self, nums1: List[int], nums2: List[int], nums3: List[int], nums4: List[int]) -> int:
        sum12 = {} #key: sum between 2 lists, value:count
        sum34 = {}
        def sumofTwoLists(list1, list2):
            sum12 = {}
            for i in list1:
                for j in list2:
                    comij = i+j
                    if comij not in sum12.keys():
                        sum12[comij] = 1
                    else:
                        sum12[comij] += 1
            return sum12
        sum12 = sumofTwoLists(nums1, nums2)
        sum34 = sumofTwoLists(nums3, nums4)
        res = 0
        for i in sum12.keys():
            if -i in sum34.keys():
                res += sum12[i]*sum34[-i]
        return res 
```
```Python
#更简洁的写法，第二次build dict的时候就可以着了
class Solution(object):
    def fourSumCount(self, nums1, nums2, nums3, nums4):
        # use a dict to store the elements in nums1 and nums2 and their sum
        hashmap = dict()
        for n1 in nums1:
            for n2 in nums2:
                if n1 + n2 in hashmap:
                    hashmap[n1+n2] += 1
                else:
                    hashmap[n1+n2] = 1
        
        # if the -(a+b) exists in nums3 and nums4, we shall add the count
        count = 0
        for n3 in nums3:
            for n4 in nums4:
                key = - n3 - n4
                if key in hashmap:
                    count += hashmap[key]
        return count
```
```Python
#不用分类讨论用defaultdict
class Solution:
    def fourSumCount(self, nums1: list, nums2: list, nums3: list, nums4: list) -> int:
        from collections import defaultdict # You may use normal dict instead.
        rec, cnt = defaultdict(lambda : 0), 0
        # To store the summary of all the possible combinations of nums1 & nums2, together with their frequencies.
        for i in nums1:
            for j in nums2:
                rec[i+j] += 1
        # To add up the frequencies if the corresponding value occurs in the dictionary
        for i in nums3:
            for j in nums4:
                cnt += rec.get(-(i+j), 0) # No matched key, return 0.
        return cnt
```
Question: [383 Random Note](https://leetcode.com/problems/ransom-note/)  
Outcome with Date: 11-29: Y  
First Impression:  
Good Video/Blog:  
Learnt:  
Difficulty during Implementation:  
Logic of Solution:  
AC Code:
```Python
class Solution:
    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        dct_mag = {}
        for s in magazine:
            if s not in dct_mag.keys():
                dct_mag[s] = 1
            else:
                dct_mag[s] += 1
        for s in ransomNote:
            if s in dct_mag.keys() and dct_mag[s] != 1:
                dct_mag[s] -= 1
            elif s in dct_mag.keys() and dct_mag[s] == 1:
                del dct_mag[s]
            else:
                return False 
        return True
```

Question: [15 3Sum](https://leetcode.com/problems/3sum/)  
Outcome with Date: 11-29:X  
First Impression: 有想法知道用一个nested for循环再查找，但是不知道具体怎么写  
Good Video/Blog:
Learnt:
Difficulty during Implementation:
Logic of Solution:
AC Code:

Question: [18
Outcome with Date: MM-DD:X|Y|O
First Impression:
Good Video/Blog:
Learnt:
Difficulty during Implementation:
Logic of Solution:
AC Code:

Question: [344
Outcome with Date: MM-DD:X|Y|O
First Impression:
Good Video/Blog:
Learnt:
Difficulty during Implementation:
Logic of Solution:
AC Code:

Question: [541
Outcome with Date: MM-DD:X|Y|O
First Impression:
Good Video/Blog:
Learnt:
Difficulty during Implementation:
Logic of Solution:
AC Code:

Question: [剑指offer05
Outcome with Date: MM-DD:X|Y|O
First Impression:
Good Video/Blog:
Learnt:
Difficulty during Implementation:
Logic of Solution:
AC Code:

Question: [151
Outcome with Date: MM-DD:X|Y|O
First Impression:
Good Video/Blog:
Learnt:
Difficulty during Implementation:
Logic of Solution:
AC Code:

Question: [剑指offer58II
Outcome with Date: MM-DD:X|Y|O
First Impression:
Good Video/Blog:
Learnt:
Difficulty during Implementation:
Logic of Solution:
AC Code:

Question: [28
Outcome with Date: MM-DD:X|Y|O
First Impression:
Good Video/Blog:
Learnt:
Difficulty during Implementation:
Logic of Solution:
AC Code:

Question: [459
Outcome with Date: MM-DD:X|Y|O
First Impression:
Good Video/Blog:
Learnt:
Difficulty during Implementation:
Logic of Solution:
AC Code:

Question: [20
Outcome with Date: MM-DD:X|Y|O
First Impression:
Good Video/Blog:
Learnt:
Difficulty during Implementation:
Logic of Solution:
AC Code:

Question: [1047
Outcome with Date: MM-DD:X|Y|O
First Impression:
Good Video/Blog:
Learnt:
Difficulty during Implementation:
Logic of Solution:
AC Code:

Question: [150
Outcome with Date: MM-DD:X|Y|O
First Impression:
Good Video/Blog:
Learnt:
Difficulty during Implementation:
Logic of Solution:
AC Code:

Question: [239
Outcome with Date: MM-DD:X|Y|O
First Impression:
Good Video/Blog:
Learnt:
Difficulty during Implementation:
Logic of Solution:
AC Code:

Question: [347
Outcome with Date: MM-DD:X|Y|O
First Impression:
Good Video/Blog:
Learnt:
Difficulty during Implementation:
Logic of Solution:
AC Code:
