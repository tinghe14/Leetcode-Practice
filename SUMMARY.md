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
|Data Structure|Access|Search or Look-up|Insertation|Deletion|
|--------------|--------------|--------------|--------------|--------------|
|Array|O(1)|O(N)|O(N)|O(N)|
|Linked List|O(N)|O(N)|O(1)|O(1)|
|Hash Map| N/A| O(1)[O(N) in worst case]|O(1)[O(N) in worst case]|O(1)[O(N) in worst case]|

## Day 1
Question: [704 binary search](https://leetcode.com/problems/binary-search/description/)  
Outcome with Date: 11-23:X; 12-4:X  
First Impression: I know need to use left, right, mid pointers but I don't know how to set the stop criteria in the loop  
First Impression at 2nd: don't know how to achive at o(log n), in my mind no idea about common algorithm achived in o(log n )  
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
Outcome with Date: 11-23:X; 12-4:X  
First Impression: don't know how to use binary search to implement, and it seems if index out of boundary or within the array are different cases
First Impresion at 2nd: know can use binary search and know the objective is to find the largest smaller value. but don't know how to do that if there is not a match    
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
## Day 2
Question: [34 Find First and Last Position of Element in Sorted Array](https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/)  
Outcome with Date: 11-24:X, 12-4: X 
First Impression:know that I can apply binary search twice by finding the target-1, and target+1 -> can't work, also the same thing will happen in target-1 and target+1->wrong  
First Impression at 2nd: 看懂了答案 但是自己写的时候逻辑乱了起来  
Good Video/Blog:https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/solutions/1136731/find-first-and-last-position-of-element-in-sorted-array/  
Learnt:
Difficulty during Implementation:  
Logic of Solution: 
1. binary框架 但是到等到发现相等时 额外几步 如果是在找first,那么mid-1的地方怎么样，如果再找last, mid+1的地方怎么养 继续binary search
AC Code:  
```Python
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        lower_pos = self.FirstorLastPos(nums, target, True)
        upper_pos = self.FirstorLastPos(nums, target, False)
        if (lower_pos == -1):
            return [-1, -1]
        return [lower_pos, upper_pos]

    def FirstorLastPos(self, nums: List[int], target: int, isFirst: bool) -> int:
        n = len(nums)
        left, right = 0, n-1
        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] == target:
                if isFirst:
                    if mid == left or nums[mid - 1] < target:
                        return mid 
                    right = mid - 1 
                else:
                    if mid == right or nums[mid + 1] > target:
                        return mid
                    left = mid + 1
            elif nums[mid] > target:
                right = mid - 1 
            else:
                left = mid + 1
        return -1 
```

Question: [27 Remove Element](https://leetcode.com/problems/remove-element/)  
Outcome with Date: 11-24:X, 12-4:X     
First Impression: no idea
First Impression at 2nd: 有想法去遍历 但是不知道对应应该做什么操作     
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

## Day 3
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
## Day 4 

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
```Python
#更清楚的解答
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
```Python
#更清楚的解答
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
First Impression: 有想法知道用一个nested for循环再哈希表查找，但是因为要去重不知道具体怎么写  
Good Video/Blog:https://www.bilibili.com/video/BV1GW4y127qo/?vd_source=8b4794944ae27d265c752edb598636de  
Learnt:（1）这题因为要去重 哈希法要考虑的太多了 更合适的是双指针的滑动窗口  
Difficulty during Implementation:
Logic of Solution:
1. since we are not return index, we can sort the array(even need index, we can use enumerate)
2. fix i and loop it in the list
3. define left and right at two ends
4. if large, move right
5. if small, move left
AC Code:(need help!!!不知道哪里写错了)

Question: [18 4Sum](https://leetcode.com/problems/4sum/)  
Outcome with Date: 11-29:  
First Impression:我写的一个nest loop再slide windows然后用set去重但是time limite exceed了->和视频里的思路是一样的 2.时间超过了所以要仔细做下剪支操作  
Good Video/Blog:https://www.bilibili.com/video/BV1DS4y147US/?spm_id_from=333.788&vd_source=8b4794944ae27d265c752edb598636de  
Learnt:(1)set()and{}both are create set object, but for set()it can only take single iteratable object  
Difficulty during Implementation:  
Logic of Solution: (need help!!)  
AC Code:  

## Day 7
### Hash Table Summarization
1. 哈希表是用来快速判断一个元素是否出现集合里
2. 对于哈希表，要知道哈希函数和哈希碰撞在哈希表重的作用
3. 哈希函数是把传入的key映射到符号表的索引上
4. 哈希碰撞会处理多个key映射到相同索引的情景，处理碰撞的普遍方式是拉链法和线性探测法
5. 题型总结：纯哈希表(实际例子 有无无限循环的出现 注意怎么计算每位相加), nested哈希表（我named例子2 sum)，滑动窗口（当有麻烦的去重条件和其他限制时）
```Python
while n!=0:
    res += (n%10)**2
    n = n //10
    return res
```
7. 纯哈希表：defaultdict可以让表达更简洁 不用判断这个key是否已经存入了 用法 d=defaultdict(int或者其他类型) 需要import包from collections import defaultdict;有时候或许可以用Counter(from collections import Counter)可以计算 dict, set,list, tuple这些可iterate的东西里面出现的次数 用法 d_count = Counter(num)
8. nested哈希表：边建立遍查找想要的东西在不在里面
9. 滑动窗口：这类题型看起来也是two sum的变型但是 要求条件多 比如互不重复的值，互不重复的ind 考虑滑动窗口双指针比哈希法简单 但是这些去重 剪枝操作还需要熟悉 

### String
1. 和数组类的题目在题型上是类似的

Question: [344 Reverse String](https://leetcode.com/problems/reverse-string/)  
Outcome with Date: 11-30:X  
First Impression: 我写成了return一个新list但是他的要求时in-place修改（reverse not in place: for ind in range(end_ind, start_ind-1, -1)  
Good Video/Blog: https://www.bilibili.com/video/BV1fV4y17748/?vd_source=8b4794944ae27d265c752edb598636de  
Learnt: 题目要求o(1）space complexity所以用双指针  
Difficulty during Implementation: 有个严重问题 while循环我已经很多次忘记在里面更新了 这个很严重 会死循环（一直没变 条件一直成立）  
Logic of Solution: 洋葱结构一层层对调  
AC Code:
```Python
class Solution:
    def reverseString(self, s: List[str]) -> None:
        """
        Do not return anything, modify s in-place instead.
        """
        left, right = 0, len(s) - 1
        while(left < right):
            s[left], s[right] = s[right], s[left]
            left += 1
            right -= 1
```

```Python
#内置函数 大部分都很快
s.reverse
s[::-1]
```
### pythonic 写法
1. 短行写一起比如上面的left, right初始化
2. 对调的话 python里面有个tuple复制 这个时候是同时tuple交换-》python里面不写（）也是tuple 
3. for, while loop那里都不用写括号

Question: [541 Reverse StringII](https://leetcode.com/problems/reverse-string-ii/)  
Outcome with Date:11-30:X  
First Impression:我想混乱了 理不清楚-》没有尝试着在脑海里去找 这题和我之前做过的哪一题类似    
Good Video/Blog:https://www.bilibili.com/video/BV1dT411j7NN/  
Learnt: （1）利用齐前面的翻转 slice之后写一个help function做字符翻转  现在class的method里的help function就是普通的本method里可以调用的函数 所以不用写self  
Difficulty during Implementation: (1)不知道怎么slice会有不能整除的情况->不是那里出错 这个range会帮忙处理 是revser那地方的报错 (2) string能做tuple的对调 要转换成list,然后list换成str的写法是  
```Python
s = list(abd)
''.join(s)
```
Logic of Solution:
AC Code:
```Python
class Solution:
    def reverseStr(self, s: str, k: int) -> str:
        def reverse(s:List[str]):
            # two pointers
            left, right = 0, len(s) - 1
            while left < right:
                s[left], s[right] = s[right], s[left]
                left += 1
                right -= 1
            return s

        res = list(s)
        for ind in range(0, len(s), 2*k):
            if len(s[ind:]) >= k:
                res[ind:ind+k] = reverse(res[ind:ind+k])
            else:
                res[ind:ind+k] = reverse(res[ind:])

        return ''.join(res)
```

Question: [剑指offer05 替换空格]  
描述：实现一个函数，把字符串s中的每个空格替换成“%20”  
Outcome with Date: 11-30:X  
First Impression:想把str换成list在一个个循环碰到空格变成%20  
Good Video/Blog: https://programmercarl.com/%E5%89%91%E6%8C%87Offer05.%E6%9B%BF%E6%8D%A2%E7%A9%BA%E6%A0%BC.html#%E5%85%B6%E4%BB%96%E8%AF%AD%E8%A8%80%E7%89%88%E6%9C%AC  
Learnt: 先把str根据空格的个数扩张 然后从后向前替换空格（left指向旧长度的末尾，right指向新长度的末尾)(后往前的好处 从前往后就是o(n^2)算法了，因为每次添加元素所有的东西都要向后移动 -》tips:很多数组填充题，都可以先预留给数组扩容填充后的大小，然后再从后向前进行操作  
Difficulty during Implementation:没有自己尝试编译，其实需要的多练练双指针问题  
Logic of Solution: (need help !!!). 
AC Code:
```Python
class Solution:
    def replaceSpace(self, s: str) -> str:
        counter = s.count(' ')
        
        res = list(s)
        # 每碰到一个空格就多拓展两个格子，1 + 2 = 3个位置存’%20‘
        res.extend([' '] * counter * 2)
        
        # 原始字符串的末尾，拓展后的末尾
        left, right = len(s) - 1, len(res) - 1
        
        while left >= 0:
            if res[left] != ' ':
                res[right] = res[left]
                right -= 1
            else:
                # [right - 2, right), 左闭右开
                res[right - 2: right + 1] = '%20'
                right -= 3
            left -= 1
        return ''.join(res)
            
```
```Python
class Solution:
    def replaceSpace(self, s: str) -> str:
        # method 1 - Very rude
        return "%20".join(s.split(" "))

        # method 2 - Reverse the s when counting in for loop, then update from the end.
        n = len(s)
        for e, i in enumerate(s[::-1]):
            print(i, e)
            if i == " ":
                s = s[: n - (e + 1)] + "%20" + s[n - e:]
            print("")
        return s
```
## Day 09
Question: [151 Reverse Words in a String](https://leetcode.com/problems/reverse-words-in-a-string/)  
Outcome with Date: 12-02:X  
First Impression: 感觉要用left right pointer但是从两头开始会很奇怪 感觉实现不了  
Good Video/Blog: 卡尔的python讲的不是很好 需要看官方解答。
1. https://www.bilibili.com/video/BV1uT41177fX/?vd_source=8b4794944ae27d265c752edb598636de
2. https://programmercarl.com/0151.%E7%BF%BB%E8%BD%AC%E5%AD%97%E7%AC%A6%E4%B8%B2%E9%87%8C%E7%9A%84%E5%8D%95%E8%AF%8D.html#%E5%85%B6%E4%BB%96%E8%AF%AD%E8%A8%80%E7%89%88%E6%9C%AC
3. https://leetcode.com/problems/reverse-words-in-a-string/solutions/441675/reverse-words-in-a-string/
Learnt:（1）用删除array中数字的方法原地o(n)删除多余空格，再整个字符串反转，接着每个单词反转-》卡尔视频中删除空格python中不能使用因为python string is not immutable  
(2) https://careerkarma.com/blog/python-str-object-does-not-support-item-assignment/  
(3) strings in python are immutable. This means that they can't be changed. the most common sceneria in which this error is raised is when you try to change a string by its index values (c++ is mutable string)->不能原地 可以存入列表
```Python
#下行会报错 s->str
s[slow] = s[fast]
# str object doesn't support item asisgnment
string = 'Banana'
string[0] = 'A'
```
Difficulty during Implementation: 自己就删除space实现那里就失败了 python的string is immutable但是之前数组删除对应元素的思想是要in place修改的
Logic of Solution: （!!!need help还是不会）
AC Code:
```Python
# in-place修改版本
class Solution:
    def reverseWords(self, s: str) -> str:
        # delete space
        s = " ".join(reversed(s.split()))
        return s
# time: O(n)
# space: O(n)
```
## Day 10
Question: [剑指offer58II 左旋字符串](https://leetcode.cn/problems/zuo-xuan-zhuan-zi-fu-chuan-lcof/) 
Outcome with Date: 12-03: Y(但是如果不能用切片法就不知道了）  
First Impression: 可以使用双向链表deque   
Good Video/Blog: https://programmercarl.com/%E5%89%91%E6%8C%87Offer58-II.%E5%B7%A6%E6%97%8B%E8%BD%AC%E5%AD%97%E7%AC%A6%E4%B8%B2.html  
Learnt:（1）不能直接使用切片法。单词反转+句子反转 （2）如果不能用内置reversed函数 自己写一个
Difficulty during Implementation:
Logic of Solution:
AC Code:
```Python:
#方法一：直接使用切片
class Solution:
    def reverseLeftWords(self, s: str, n: int) -> str:
    return s[n:] + s[0:n]
#方法二： 有些面试不允许使用切片，那就使用文章中的方法
class Solution:
    def reverseLeftWords(self, s: str, n: int) -> str:
    s = list(s)
    s[0:n] = list(reversed(s[0:n]))
    s[n:] = list(reversed(s[n:]))
    s.reverse()
    return ''.join(s)
#方法三：如果不让用reverse函数自己写
def reverse_sub(lsb, left, right):
    while left < right:
        lst[left], lst[right] = lst[right], lst[left]
        left += 1
        right -= 1
```
Question: [28 Find the Index of the First Occurrence in a String](https://leetcode.com/problems/find-the-index-of-the-first-occurrence-in-a-string/)  
Outcome with Date: 12-03:X  
First Impression:没有想法  
Good Video/Blog:
1. kmp理论篇 https://www.bilibili.com/video/BV1PD4y1o7nd/?vd_source=8b4794944ae27d265c752edb598636de 
2. kmp求前缀的代码 https://www.bilibili.com/video/BV1M5411j7Xx/?vd_source=8b4794944ae27d265c752edb598636de
Learnt:
Difficulty during Implementation:
Logic of Solution:
AC Code: (!!!need help)

Question: [459
Outcome with Date: MM-DD:X|Y|O
First Impression:
Good Video/Blog:
Learnt:
Difficulty during Implementation:
Logic of Solution:
AC Code: (!!!need help)

### String Summarization
1.题型：双指针，反转系列,kmp

### Two Pointer Review
1. frequent use in array and string （！！！他给的题目还没有练过）
2. 数组篇，字符串篇
3. 数组篇：原地删除数组上的元素，不能真正的删除，只能覆盖
4. 字符串篇：在替换空格中介绍，使用双指针填充字符串的方法，如果把这道题做到极致，就不用额外的空间了：首先扩充数组到每个空格替换成%20之后的大小，然后双指针从后向前替换空格
5. 链表题：使用快慢指针，分别定义fast和slow指针，从头节点出发，fast指针每次移动两个节点，slow指针每次移动一个节点，如果fast和slow指针在途中相遇，就说明这个链表有环
6. n数之和：哈希表解决了两数之和，n数之和使用双指针：通过前后两个指针不断向中间逼近，在一个for循环下完成两个for循环的工作；四数之和也是一样的，在三数之和的基础上再套一层for循环，依然使用双指针法
7. 总结：除了链表一些题目一定要使用双指针，其他题目都是使用双指针来提高效率，一般是将o(n^2)的时间复杂度，降为o(n)

## Day 8

### Stack栈与Queue队列 
https://zhuanlan.zhihu.com/p/43505915. 
1. 队列是先进先出，栈是先进后出
2. 他们都是比较特殊的线性表，对于队列来说，元素只能从队列尾部插入，从队列头访问和删除
3. 对于栈来说，访问，插入和删除元素只能在栈顶进行。该位置是表的末端，叫做栈顶。对栈顶的基本操作有push进栈和pop出栈，前者相当于插入，后者相当于删除最后一个元素。栈被叫做LIFO(last in first out)表，即后进先出。
4. 因为栈也是一个表，所以任何实现表的方式都能实现栈。我们可以用python的list来模拟栈的相关操作如pop(), append()
5. 队列是一种特殊的线性表，特殊在于它只允许在表的前段(front)进行删除操作，而在表的后端（rear)进行插入操作。和栈一样，队列是一种受操作限制的线性表。进行插入操作的端叫做队尾，进行删除操作的端称为对头。python的队列实现
```Python
import Queue
q = Queue.Queue()
for i in range(5):
    q.put(i)
while not q.empty():
    print q.get()
```

Question: [232 implement queue using stacks](https://leetcode.com/problems/implement-queue-using-stacks/)  
Outcome with Date: 12-01:X  
First Impression:没有想法 因为一开始对这个数据结构都不了解 基本的栈是先进后出我知道 但是不知道怎么用python实现  
Good Video/Blog:https://www.bilibili.com/video/BV1nY4y1w7VC/?vd_source=8b4794944ae27d265c752edb598636de  
Learnt:（1）一个队列可以由两个栈形成  一个是进栈代替进队列的行为 一个是出栈为了膜拜出队列的行为  
Difficulty during Implementation:(1)init的时候有问题 但其实就是很简单的self.stack_in=[]然后out也是一个list就行（2）实现pop的时候也有疑惑 里面需要弹出所有的stack_in我们也会用pop不过这是list built-in function和要实现的pop无关
Logic of Solution: pop那部分 如果列表为空 回复none 如果stack_out有东西 回复pop 如果stak_in有东西 先pop再append再pop
AC Code:
```Python
class MyQueue:

    def __init__(self):
        self.stack_in = []
        self.stack_out = []
        
    def push(self, x: int) -> None:
        self.stack_in.append(x)
    
    def pop(self) -> int:
        if self.empty():
            return None
        if self.stack_out:
            return self.stack_out.pop()
        else:
            for i in range(len(self.stack_in)):
                self.stack_out.append(self.stack_in.pop())
            return self.stack_out.pop()


    def peek(self) -> int:
            peek_value = self.pop()
            self.stack_out.append(peek_value)
            return peek_value
        
    def empty(self) -> bool:
        if len(self.stack_in) == 0 and len(self.stack_out) == 0:
            return True 
        else:
            return False


# Your MyQueue object will be instantiated and called as such:
# obj = MyQueue()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.peek()
# param_4 = obj.empty()
```

Question: [225 Implement Stack using Queues](https://leetcode.com/problems/implement-stack-using-queues/)  
Outcome with Date: 12-01:X  
First Impression:没有想法。
Good Video/Blog:
1. https://www.bilibili.com/video/BV1Fd4y1K7sm/
2. https://programmercarl.com/0225.%E7%94%A8%E9%98%9F%E5%88%97%E5%AE%9E%E7%8E%B0%E6%A0%88.html
3. http://jalan.space/2019/02/05/2019/python-queue/
Learnt:放入也是一样的 关键在于如何弹出(1)用两个队列的话就是 一共pop出n-1个 逐个放入队列中 （2）用一个队列的话就是 一共pop出n-1个 逐个放回本身队列中(3)python里实现队列的方法我不知道 查了一下中文leetcode '你所使用的语言也许不支持队列。你可以使用list列表或者deque双端队列来模拟一个队列，只要是标准的队列操作即可'（4）python中实现队列的方法 借用list列表，借用deque双端队列（既可以实现栈也可以实现队列）（5）队列是一种特殊的线性表，是一种先进先出数据结构，只允许在表的前段front进行删除操作，而在表的后端rear进行插入操作。进行插入操作的端称为队尾，进行删除操作的端称为对头  
```Python
#列表list
#创建列表list
q = []
#入队
q.append("a")
#出对
del q[0]
```
```Python
from collections import deque
#创建队列
d = deque()
#入队
d.append(1)#从队尾
d.appendleft(2)#从对头
#出队
d.pop(）#从队尾
d.popleft()#从队头
```
Difficulty during Implementation: 知道逻辑之后 自己实现确实很简单。
Logic of Solution:  
AC Code:
```Python
from collections import deque

class MyStack:

    def __init__(self):
        self.queue = deque()
        
    def push(self, x: int) -> None:
        self.queue.append(x)

    def pop(self) -> int:
        if self.empty():
            return None
        size = len(self.queue)
        while (size-1) > 0:
            prev = self.queue.pop()
            self.queue.append(prev)
            size -= 1
        return self.queue.pop()

    def top(self) -> int:
        top_item = self.pop()
        self.queue.append(top_item)
        return top_item
        
    def empty(self) -> bool:
        size = len(self.queue)
        if size == 0:
            return True
        return False
```

Question: [20 Valid Parentheses](https://leetcode.com/problems/valid-parentheses/)  
Outcome with Date: 12-01: X
First Impression: 这题我想着用hash table不就行了 但是发现只是个数一样不可以 需要时那种结构 先push进去然后遇到了end parentheses把对应的pop出来 所以该用的是栈  ->但是我感觉洋葱结构的和()[]这样结构的不一样 不知道怎么处理 甚至是不是需要分类处理->题意用中文解释成为 例如当有左括号时，相应的位置必须要有右括号    
Good Video/Blog:
1. https://www.bilibili.com/video/BV1AF411w78g/
2. https://programmercarl.com/0020.%E6%9C%89%E6%95%88%E7%9A%84%E6%8B%AC%E5%8F%B7.html#%E9%A2%98%E5%A4%96%E8%AF%9D
Learnt:（1）!!同样的错误犯两次了 （2）由于栈结构的特殊性，非常适合做对称匹配类的题目，对于这种题目一定要先分析有哪几种不匹配的情况，再动手分析（3）这道题的不匹配情况只有三种 a.左边有个多余的括号 b.没有多余的括号但是括号类型没有匹配上 c.右面多了右括号 (!!!need help)
```Python
#不能这样
# str object is not iterable
for i in str： #错误->好奇怪其实是可以的 所以我又是哪里做错了呢  
```
Difficulty during Implementation: 不知道为什么代码有错 需要理解 而且哈希表方法也想会（need help!!!）  
Logic of Solution:  
AC Code:  

## Day 10
Question: [1047 remove all adjacent duplicates in string](https://leetcode.com/problems/remove-all-adjacent-duplicates-in-string/)  
Outcome with Date: 12-03:X  
First Impression: 用stack是想到了 但是代码得到的结果为空 按照答案的顺序写才有解 不知道为什么  
Good Video/Blog: https://programmercarl.com/1047.%E5%88%A0%E9%99%A4%E5%AD%97%E7%AC%A6%E4%B8%B2%E4%B8%AD%E7%9A%84%E6%89%80%E6%9C%89%E7%9B%B8%E9%82%BB%E9%87%8D%E5%A4%8D%E9%A1%B9.html#%E5%85%B6%E4%BB%96%E8%AF%AD%E8%A8%80%E7%89%88%E6%9C%AC
Learnt: （1）不是要pop出来一个来看吗 不用pop的因为这样做了还要塞回去 直接下标访问就行  
Difficulty during Implementation:
Logic of Solution:
AC Code:
```Python
# 我当时一样的想法 但是没编译出来
class Solution:
    def removeDuplicates(self, s: str) -> str:
        # use stack, when the top of the stack is the same as the incoming item, pop the stack
        # final: return what is left in stack 
        stack = list()
        for i in s:
            if len(stack) != 0 and stack[-1] == i:
                stack.pop()
            else:
                stack.append(i)
        return "".join(stack)
        
# 如果不让使用栈，可以使用双指针模拟栈 #需要之后看
class Solution:
    def removeDuplicates(self, s: str) -> str:
        res = list(s)
        slow = fast = 0
        length = len(res)
        
        while fast < length:
            #如果一样就直接换，不一样会把后面的填在slow的位置
            res[slow] = re[fast]
            #如果发现和前一个一样，九退一格指针
            if slow > 0 and res[slow] == res[slow - 1]:
                slow -= 1
            else:
                slow += 1
            fast += 1
         return ''.join(res[0:slow])
```

Question: [150 Evaluate Reverse Polish Notation](https://leetcode.com/problems/evaluate-reverse-polish-notation/)  
Outcome with Date: 12-03: X  
First Impression:知道也是用stack而且知道怎么用 但是不知道怎么把str的操作符 变成正在的操作符  
Good Video/Blog: https://leetcode.com/problems/evaluate-reverse-polish-notation/solutions/509590/evaluate-reverse-polish-notation/  
Learnt:(1)dict可以帮忙apply不同情况的操作符号，lambda放在value里面可以起到操作 （2)普通的python操作不能连起来 比如stack.pop().pop()会报错  
Difficulty during Implementation:
Logic of Solution:
AC Code:
```Python
class Solution:
    def evalRPN(self, tokens: List[str]) -> int:
        # use stack, keep poping the nums untill meets an operation, 
        # caculate the first two nums use that operation then push back to stack
        # keep the process untill nothing is waiting to push
        # return the final number at stack 
        stack = list()
        oper_list = ["+", "-", "*", "/"]
        operations = {
        "+": lambda a, b: a + b,
        "-": lambda a, b: a - b,
        "/": lambda a, b: int(a / b),
        "*": lambda a, b: a * b }

        for item in tokens:
            if len(stack) > 1 and item in oper_list:
                num1, num2 = stack[-2], stack[-1]
                res = operations[item](int(num1), int(num2))
                stack.pop()
                stack.pop()
                stack.append(res)
            else:
                stack.append(item)
        return stack[0]
```
## Day 11

Question: [239 slidingg window maximum](https://leetcode.com/problems/sliding-window-maximum/)
Outcome with Date: 12-04:X  
First Impression:这题是hard说是一刷至少要理解思路 后面回来补充; 有点想法，感觉滑动窗口o(n)就能实现，但是一开始就败下了场，不知道怎么找k里面的最大值,找到了也会有问题，因为我需要记住下标，如果之前的最大值被移掉了 又要从头计算吗
Good Video/Blog: https://www.bilibili.com/video/BV1XS4y1p7qj/?vd_source=8b4794944ae27d265c752edb598636de https://programmercarl.com/0239.%E6%BB%91%E5%8A%A8%E7%AA%97%E5%8F%A3%E6%9C%80%E5%A4%A7%E5%80%BC.html
Learnt:（0）暴力解法是o(n*k) (1)去实现一个monotonic queue单调队列：维护队列里面单调递增或者单调递减 (2)流程 做一个长度为k的单调递减队列，保证当前窗口最大值在出口，如果新加的值比之前的值大，全都pop掉因为维护也没意义，最大值肯定都是当前值直到移动到当前值，他需要被pop的时候  
Difficulty during Implementation:
Logic of Solution:
AC Code:

## 优先队列priorityqueue, 堆heap
1. 图灵星球 https://www.youtube.com/watch?v=wTAoOhytiQs（不是python看看就好）https://www.geeksforgeeks.org/heap-and-priority-queue-using-heapq-module-in-python/
2. 对于之前的常见数据结构数组，链表，堆，栈，如果我们想找最大或者最小值，一般都要挨个寻找花费o(n);使用priorityqueue或者heap能节省很多，达o(1)
3. priorityqueue

Question: [347 Top K Frequent Element](https://leetcode.com/problems/top-k-frequent-elements/)
Outcome with Date:12-04:O
First Impression: 自己一开始想用defaultdict,但发现most_common是Counter里的函数，而且我不知道这个most_common操作的时间复杂度是多少->o(nlogn)
Good Video/Blog: https://www.bilibili.com/video/BV1Xg41167Lz/ 
Learnt:
Difficulty during Implementation: （1）不能import Counter from collections一定是from...import...  
Logic of Solution: (小唐说只要知道用heapq就行 之后看！！！)
AC Code:
```Python
from collections import Counter

class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        # hash table: nums: count
        # most_common(k)
        res = Counter()
        for num in nums:
            res[num] += 1
        return [i for i, j in res.most_common(k)]
```
## Stack & Queue Summarization

## Binary Tree二叉树
0. A full Binary tree is a special type of binary tree in which every parent node/internal node has either two or no children. It is also known as a proper binary tree; A Binary tree is a Perfect Binary Tree in which all the internal nodes have two children and all leaf nodes are at the same level. 
1. https://www.bilibili.com/video/BV1Hy4y1t7ij/?vd_source=8b4794944ae27d265c752edb598636de
2. 种类：(a)perfect binary tree满二叉树(节点数量2……k-1）,(b)complete binary tree完全二叉树（除了底层全满的，并且底层是从左到右节点连续的,(c)binary search tree二叉搜索树（搜索一个节点的时间复杂度是logn级别的，全部左子树都小于中间节点，全部右子树都大雨中间节点，其对树的结构没有要求，对元素又要求），(d)balanced binary search tree平衡二叉搜索树（全部左子树和右子树的高度绝对值的差不能大于1，很多广泛应用，使用这个数据结构，插入节点，查询元素都是o(logn)级别，需要有意识的了解python中的容器的底层实现，这样才能清楚的知乎到他们的数据是否有序，插入等操作消耗）
3. 二叉树的存储方式：链性存储(就是我们常见的，用指针链接节点)和线性存储（用一个数组来保存二叉树）
4. 链式存储：
5. 线性存储：given母节点，左孩子是2*i-1, 右孩子是2*i+1
6. leetcode里面的input都给你了，但是实际面试会让你传入一个二叉树。需要自己会构造，一般我们都用链式结构（单向双链）传入根节点
7. 二叉树的遍历：即是图论中的两种遍历方式：(a)深度优先遍历和(b)广度优先遍历
8. 深度优先搜索，我们一般用递归的方式实现，前序，中序，后序；也可以用对应的迭代法（非递归的方式 会考的给一个简单的二叉树问题，让你用迭代法实现）实现前中后序
9. 根节点的顺序，前：中左右，中：左中右，后：左右中；比如前：现就是中节点，然后在左子树中继续前序搜索，在右子树中继续前序搜索
10. 广度优先搜索，一层一层的遍历/一圈一圈的遍历，层序遍历（就是迭代法 用一个队列 先进先出 进行实现）
11. 二叉树的定义/构造，leetcode都给好了，但是面试会考，其实很简单就是一种链表
```Python
class TreeNode: 
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
```
12. iteration迭代还树内某段代码实现循环， recursion递归重复调用函数自身实现循环； 迭代与普通循环的区别：循环代码中参与运算的变量同时时保存结果的变量，当前保存的结果作为下一次循环计算的初始值；递归循环中，遇到满足终止条件的情况时逐层返回来结束。迭代则使用计数器结束循环。很多具体情况会采用多种循环结合

#### python数据结构的底层实现
1. c++中map, set, multimap, multiset的底层实现都是红黑树（平衡二叉搜索树），是一个排序的结构，所以map，set的插入，查找和删除的时间复杂度是o(logn)
2. c++中还提供了unordered_map, unorder_set,他们的底层实现是哈希表，是没有排序的，插入，查找和删除都可以在o(1)的时间复杂度内完成 
3. python
4. list列表，是长度可变的数组，细节上是由对其它对象的引用组成的连续数组。需要注意的是，一些在普通链表上代价很小的操作在python list的时间复杂度代价相对过高：(a)利用list.insert方法在任意位置插入一个元素-》o(n),(b)利用list.delete获得del删除一个元素->复杂度o(n)
5. tuple元组，tuple是不可变的，一旦创建就不能修改
6. dict字典，使用伪随机探测（pesudo-random prbing)的散列表（hash table)作为字典的底层数据结构。由于这个实现细节，只有可哈希的对象才能作为字典的键。python中所有不可变的内置类型都是可哈希的。可变的类型如列表，字典和集合就是不可哈希的，因此不能作为字典的键。dict的获取，修改和删除平均复杂度都是O（1）[平均最坏复杂度都是O(n)]比list的查找和杉树都要快，但是使用hashtable内存的开销更大。为了保证较少的冲突，hashtable的装载因子，一般要小与0.75，在python当中当装载因子达到2/3的时候就会自动进行扩容。使用字典的常见缺点：不会按照键的添加术训来保存元素的顺序。如果需要保存添加顺序怎么办：python标准困的collections模块提供了ordereddict的有序字典
7. set集合，集合是一种鲁棒性很好的数据结构，当元素顺序的重要性不如元素的唯一性和测试元素是否包含在集合中的效率的时候，大部分情况下这种数据结构及其有用。set和dict十分类似。事实上，集合被实现为带有空值的字典，只有键才是实际的集合元素。此外，集合还利用这种没有值的映射做了其他的优化。由于这一点，和dict一样，set可以快速的向集合中添加元素，删除元素，检查元素是否存在。平均时间复杂度为 o(1），最坏的时间复杂度是o（n）

## Day 11

### 递归遍历
1. https://www.bilibili.com/video/BV1Wh411S7xt/?vd_source=8b4794944ae27d265c752edb598636de
2. https://programmercarl.com/%E4%BA%8C%E5%8F%89%E6%A0%91%E7%9A%84%E9%80%92%E5%BD%92%E9%81%8D%E5%8E%86.html
3. 递归三部曲 （a）确定递归函数的参数和返回值（不需要一开始确定，需要什么参数加什么参数就行，大部分二叉树题目，只要根节点和一个数组作为参数，数组是用来放我们遍历的结果 （回溯算法的参数就多了）返回值一般来说都是void 因为我们把想要的结果直接放在参数里了）（b）确定终止条件（溢出出bug一般都是因为终止条件没有定义好，深度优先搜索会往一个方向一直搜再返回，那一定是遇到null空节点的时候）（c）确定单层递归的逻辑（比如一层中 中序遍历的顺序 中遍历直接放重脚步进入保存数组，再对左，右递归调用）
4. （a）确定递归函数的参数和返回值：确定哪些参数是递归的过程中需要处理的，那么就在递归函数里加上这个参数，并且还要明确每次递归的返回值是什么进而确定递归函数的返回类型 （b）确定终止条件：写完了递归算法，运行的时候，经常会遇到栈溢出的错误。如果递归没有终止，操作系统的内存栈必然就会溢出（c）确定单层递归的逻辑：确定每一次层递归需要处理的信息。在这里也就会重复调用自己来实现递归的过程

Question: [144 binary tree preorder traversal](https://leetcode.com/problems/binary-tree-preorder-traversal/)  
Outcome with Date: 12-04:X  
First Impression:第一次自己写还是不太会  
Good Video/Blog:https://programmercarl.com/%E4%BA%8C%E5%8F%89%E6%A0%91%E7%9A%84%E9%80%92%E5%BD%92%E9%81%8D%E5%8E%86.html  
Learnt:  
Difficulty during Implementation:  
Logic of Solution:
AC Code:
```Python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        result = []
        def traversal(root: TreeNode):
            if root == None:
                return 
            result.append(root.val)
            traversal(root.left)
            traversal(root.right)
        traversal(root)
        return result
```
Question: [145 binary tree postorder traversal](https://leetcode.com/problems/binary-tree-postorder-traversal/)  
Outcome with Date: 12-04:X  
First Impression: 知道思路了 结果返回集，定义递归函数，调用递归函数，返回结果  
Good Video/Blog:https://programmercarl.com/%E4%BA%8C%E5%8F%89%E6%A0%91%E7%9A%84%E9%80%92%E5%BD%92%E9%81%8D%E5%8E%86.html  
Learnt:  
Difficulty during Implementation: 根节点append的是val 记得不是root本身哦  
Logic of Solution:  
AC Code:
```Python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def postorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        result = []
        def traversal(root: TreeNode):
            if root == None:
                return 
            traversal(root.left)
            traversal(root.right)
            result.append(root.val)
        traversal(root)
        return result
```

Question: [94 binary tree inorder traversal](https://leetcode.com/problems/binary-tree-inorder-traversal/)  
Outcome with Date: 12-04: Y  
First Impression:  
Good Video/Blog:  
Learnt:  
Difficulty during Implementation: 才发现root的类型是之前我们怎么定义这个数的class的，奇怪的一点为什么basic case判断的是root是否为空而不是root.val（need help!!!）  
Logic of Solution:  
AC Code:
```Python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        result = []
        def traversal(root: TreeNode):
            if root == None:
                return 
            traversal(root.left)
            result.append(root.val)
            traversal(root.right)
            return result 
        return traversal(root)
         
```

### 迭代遍历
1. https://programmercarl.com/%E4%BA%8C%E5%8F%89%E6%A0%91%E7%9A%84%E8%BF%AD%E4%BB%A3%E9%81%8D%E5%8E%86.html 写出二叉树的非递归遍历很难么？前序和后序：https://www.bilibili.com/video/BV15f4y1W7i2/ 中序：
2. 遍历：given a tree,把节点放在数组里
3. 编程里使用栈（先进后出 所以前序的话：中节点入栈 出栈存入数组（数组是我们的遍历顺序） 右节点入栈 左节点入栈 左节点（中）出栈存入数组 左的右节点入栈...）来实现递归，那么这里迭代法也是用栈来模拟递归
4. 前序：中左右-》后序：中右左，reversed（不能用built-in操作的话 就用后序迭代法）-〉左右中
5. 中序：左中右：遍历的顺序（访问节点：-个个访问），和他处理的顺序是不一样的（处理节点：放在数组中），不能在简洁的前序代码中直接进行修改; 左孩子为空弹出自己，右孩子为空弹出栈中的一个元素；需要一个指针来遍历节点，栈来记录遍历过的元素

Question: [144 binary tree preorder traversal](https://leetcode.com/problems/binary-tree-preorder-traversal/)  
Outcome with Date: 12-05:X  
First Impression: 看完书视频后，看了代码细节，尝试后序的遍历  
Good Video/Blog: https://www.youtube.com/watch?v=xIS5oGZfaS4
Learnt:  
Difficulty during Implementation: (1)stack=[root]加的就是头节点 那个指针 不是整个数 （2）while stack那里不能写成stack != None因为None是一个特殊的类别Nonetype stack为空是也不是nonetype而是一个空栈，判断他为空len()==0 . 
Logic of Solution:
AC Code:
```Python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        if root == None:
            return []
        stack = [root] # stack to process tree node #是头节点的意思 是个数
        # TreeNode{val: 1, left: None, right: TreeNode{val: 2, left: TreeNode{val: 3, left: None, right: None}, right: None}}
        result = [] # list of int to save traversal results
        while stack:
            # mid point
            node = stack.pop()
            result.append(node.val)
            # right
            if node.right:
                stack.append(node.right)
            if node.left:
                stack.append(node.left)
        return result
```
Question: [145 binary tree postorder traversal](https://leetcode.com/problems/binary-tree-postorder-traversal/)  
Outcome with Date: 12-05:Y  
First Impression:  记得怎么写 经后需要多练习  
Good Video/Blog:
Learnt:  
Difficulty during Implementation: 
Logic of Solution:  
AC Code:
```Python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def postorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        if root == None:
            return []
        stack = [root]
        result = []
        while stack:
            node = stack.pop()
            result.append(node.val)
            if node.left:
                stack.append(node.left)
            if node.right:
                stack.append(node.right)
        return reversed(result)
```

Question: [94 binary tree inorder traversal](https://leetcode.com/problems/binary-tree-inorder-traversal/)  
Outcome with Date:  
First Impression:  
Good Video/Blog: https://www.bilibili.com/video/BV1Zf4y1a77g/?spm_id_from=333.788&vd_source=8b4794944ae27d265c752edb598636de  
Learnt: #左孩子为空弹出自己，右孩子为空弹出栈里元素  
Difficulty during Implementation:  
Logic of Solution:（！！了解了视频 之后再自己写）  
AC Code:
```Python

         
```

### 统一遍历
1. 这是统一迭代法的写法， 如果学有余力，可以掌握一下 https://programmercarl.com/%E4%BA%8C%E5%8F%89%E6%A0%91%E7%9A%84%E7%BB%9F%E4%B8%80%E8%BF%AD%E4%BB%A3%E6%B3%95.html

## Day 12  
### 层序遍历
1. 二叉树的层序遍历，相当于图论里的广度优先搜索 
2. 看完本篇可以一口气刷十道题，试一试， 层序遍历并不难，大家可以很快刷了十道题。https://programmercarl.com/0102.%E4%BA%8C%E5%8F%89%E6%A0%91%E7%9A%84%E5%B1%82%E5%BA%8F%E9%81%8D%E5%8E%86.html
3. 很明显按照二叉树本身结构是无法直接层序遍历，我们需要借助一个队列queue去保存每一层的元素，额外的一个size去记录正在遍历的二叉树这一层中有几个元素（也是队列的大小）；为什么要记录这个size：因为队列里的元素个数是不断变化的，如果不提前记录下来，我都不知道这个队列里面要弹出多少个元素（为什么会变化呢：因为弹出一个元素的时候就会加入他的左右孩子），怎么去记录这个size呢：，最后的结果是一个二维数组（【【第一层的元素】【第二层的元素】，。。】）
4. 代码逻辑：（a）把头节点放入队列中（2）开始循环 当队列不为空时 先快照其size （3）再size--循环 探出node 放入其左右节点 记录这一层的节点（4）append到result list中
5. deque相比list的好处是，list的pop(0)是O(n)复杂度，deque的popleft()是O(1)复杂度(!!!基本操作的时间复杂度确实是个大头！！！)
6. 复杂度. 时间：O(n), 空间: O(n)

```Python
class Solution:
    """二叉树层序遍历迭代解法"""

    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        results = []
        if not root:
            return results
        
        from collections import deque
        que = deque([root]) 
        # que = deque()
        # que.append(root)
        
        while que:#值得是一层
            size = len(que)
            result = []
            for _ in range(size):
                cur = que.popleft()
                result.append(cur.val)
                if cur.left:
                    que.append(cur.left)
                if cur.right:
                    que.append(cur.right)
            results.append(result)

        return results     
```

```Python
# 递归法
# # 这个leetcode官方recursion图解写的很好 https://leetcode.com/problems/binary-tree-level-order-traversal/solutions/255802/binary-tree-level-order-traversal/
class Solution:
    def levelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        levels = []
        if not root:
            return levels
        
        def helper(node, level):
            # start the current level
            if len(levels) == level:
                levels.append([])

            # append the current node value
            levels[level].append(node.val)

            # process child nodes for the next level
            if node.left:
                helper(node.left, level + 1)
            if node.right:
                helper(node.right, level + 1)
            
        helper(root, 0)
        return levels
```
Question: [102 binary tree level traversal](https://leetcode.com/problems/binary-tree-level-order-traversal/)
Outcome with Date: 12-05:X 递归和迭代法都要练习 这次的递归法看答案都理解不了   
First Impression:(1)queue和deque分不清楚 而且不知道里面对应的python操作 (2)遍历逻辑忘记了 看了答案 
Good Video/Blog: https://programmercarl.com/0102.%E4%BA%8C%E5%8F%89%E6%A0%91%E7%9A%84%E5%B1%82%E5%BA%8F%E9%81%8D%E5%8E%86.html  
Learnt: (1) 翻了第三次错误了 是from collection import deque 而且这道题是deque(!!有时间看一下queue和deque的区别！！！)——》建队d=deque(),从队尾入队d.append(1),从对头入队d.appendleft(2),从队尾删除d.pop(),从对头删除d.popleft()
Difficulty during Implementation:  (1)看完答案迭代法差不多写出来了 但是temp的位置放错了
Logic of Solution: 
AC Code:
```Python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
from collections import deque
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        result = []
        if root == None:
            return result 
        d = deque()
        d.append(root)
        while d:
            layerSize = len(d)
            temp = []
            while layerSize:
                node = d.popleft()
                layerSize -= 1
                temp.append(node.val)
                if node.left:
                    d.append(node.left)
                if node.right:
                    d.append(node.right)
            result.append(temp)
        return result
         
```
```Python
# 看了一开始深度优先所有的递归法 发现基本逻辑是一样的 但还是不会写 尝试自己在理解一下
# 这个leetcode官方recursion图解写的很好 https://leetcode.com/problems/binary-tree-level-order-traversal/solutions/255802/binary-tree-level-order-traversal/
# 12/05看完了还是不太会 之后要过来回顾！！！不太理解 现在就背 反正是固定写法,其实这个要理解一下 比如这里没有用deque 也没有用size, 而且为什么有node。
# 12/05为什么 left和node.right时候level会被加两次 加左边的数的时候len(results) = level的 右边的时候因为左边加了一个list他们的值是相差1的; 终止条件在if node.right/left中实现了
# 小唐说递归肯定是以深度优先遍历的，但是存储的时候是按一层层走的 讨巧的是if len(results) == levels 和后面results[level].append(node.val) 传入depth的目的是为了记住是哪一层

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

from collections import deque
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        results = []
        if root == None:
            return results 

        def helper(node, level):
            if len(results) == level:
                results.append([]) 
            # start the current level
            # append the current val to responding layer
            results[level].append(node.val)
            # process child nodes for the next level
            if node.left:
                helper(node.left, level+1)
            if node.right:
                helper(node.right, level+1)
        helper(root, 0)
        return results
```

Question: [107 binary tree level order traversal ii](https://leetcode.com/problems/binary-tree-level-order-traversal-ii/)  
题目： 广度优先 但是从子节点输出到头节点
Outcome with Date: 12-05: X  
First Impression: 觉得一个栈可能就管用 但是不知道怎么判断循环截止（当数全都append到栈里的时候） 
Good Video/Blog: https://programmercarl.com/0102.%E4%BA%8C%E5%8F%89%E6%A0%91%E7%9A%84%E5%B1%82%E5%BA%8F%E9%81%8D%E5%8E%86.html#_107-%E4%BA%8C%E5%8F%89%E6%A0%91%E7%9A%84%E5%B1%82%E6%AC%A1%E9%81%8D%E5%8E%86-ii  
Learnt: 按照广度优先搜索走 最后再reversed就行 但是我自己没想到；而且广度优先搜索的模版我还需要熟练
Difficulty during Implementation:  
Logic of Solution: 
AC Code:
```Python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def levelOrderBottom(self, root: Optional[TreeNode]) -> List[List[int]]:
        results = []
        if root == None:
            return results

        def helper(node, level):
            if len(results) == level:
                results.append([])
            results[level].append(node.val)
            if node.left:
                helper(node.left, level+1)
            if node.right:
                helper(node.right, level+1)
            return results 
        helper(root, 0)
        return reversed(results)   
```

Question: [199 binary tree right side view](https://leetcode.com/problems/binary-tree-right-side-view/)
Outcome with Date: 12-05: X  
First Impression:  有想法但是题目就读错了 像这种问题我就可以clarify是不是只能看见右边每一层离我最近的元素
Good Video/Blog: 
Learnt: 
Difficulty during Implementation:  
Logic of Solution: 
AC Code:
```Python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        # basic BFS in recursion
        # first right then left 
        results = []
        if root == None:
            return results
        def helper(node, level):
            if len(results) == level:
                results.append([])
            results[level].append(node.val)
            if node.right:
                helper(node.right, level + 1)
            if node.left:
                helper(node.left, level + 1)
            return results
        helper(root, 0)
        return [lst[0] for lst in results]      
```
```Python
# 代码分享录的解法 更好 我上面是o（2n）
```


Question: [637 average of levels in binary tree](https://leetcode.com/problems/average-of-levels-in-binary-tree/)
Outcome with Date: 12-05:X   
First Impression:  写了几次递归的写法就忘记了迭代的怎么写 哎 
Good Video/Blog: 看了之前迭代的code
Learnt: 
Difficulty during Implementation:  
Logic of Solution: 
AC Code:
```Python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

from collections import deque

class Solution:
    def averageOfLevels(self, root: Optional[TreeNode]) -> List[float]:
        results = []
        if root == None:
            return results
        d = deque()
        d.append(root) 
        while d:
            size = len(d)
            result = []          
            while size:
                node = d.popleft()
                result.append(node.val)  
                size -= 1
                if node.left:
                    d.append(node.left)
                if node.right:
                    d.append(node.right)
            results.append(mean(result))
        return results     
```

Question: [429 n-aray tree level order traversal](https://leetcode.com/problems/n-ary-tree-level-order-traversal/)
Outcome with Date: 12-05: X  
First Impression: 我觉得题目不清晰啊 这个null节点连在哪里 跟着哪一层？为什么头一层没有？那怎么区分 -> 问了小唐才知道 这是一种树的写法 用null隔开告诉你什么时候进入下一层 但是树节点本身没有null  
Good Video/Blog: https://programmercarl.com/0102.%E4%BA%8C%E5%8F%89%E6%A0%91%E7%9A%84%E5%B1%82%E5%BA%8F%E9%81%8D%E5%8E%86.html#_429-n%E5%8F%89%E6%A0%91%E7%9A%84%E5%B1%82%E5%BA%8F%E9%81%8D%E5%8E%86
Learnt: （1）提示给的是 这道题还是dfs的模板题，只不过一个节点有多个孩子 但是还没有很get到
Difficulty during Implementation:  
Logic of Solution: 
AC Code:
```Python
 """
# Definition for a Node.
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children
"""
from collections import deque

class Solution:
    def levelOrder(self, root: 'Node') -> List[List[int]]:
        results = []
        if root == None:
            return results 
        d = deque()
        d.append(root)
        while d:
            size = len(d)
            result = []
            while size:
                node = d.popleft()
                result.append(node.val)
                size -= 1
                if node.children: 
                    d.extend(node.children) # 我按照模版写的用append就运行不了，extend就可以为啥？->就是因为extend可以加入一个iterature的对性别
                    # append() adds a single element to the end of the list while . extend() can add multiple individual elements to the end of the list.
            results.append(result)
        return results     
```

Question: [515 find largest value in each tree row](https://leetcode.com/problems/find-largest-value-in-each-tree-row/)  
Outcome with Date: 12-05:Y   
First Impression:  
Good Video/Blog: 
Learnt: 
Difficulty during Implementation:  
Logic of Solution: mean的那道题换成max就行 
AC Code:
```Python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

from collections import deque 
class Solution:
    def largestValues(self, root: Optional[TreeNode]) -> List[int]:
        results = []
        if root == None:
            return results
        d = deque()
        d.append(root) 
        while d:
            size = len(d)
            result = []          
            while size:
                node = d.popleft()
                result.append(node.val)  
                size -= 1
                if node.left:
                    d.append(node.left)
                if node.right:
                    d.append(node.right)
            results.append(max(result))
        return results           
```

Question: [116 populating next right pointers in each node](https://leetcode.com/problems/populating-next-right-pointers-in-each-node/)
Outcome with Date: 12-05:X  
First Impression:没有想法  
Good Video/Blog: https://programmercarl.com/0102.%E4%BA%8C%E5%8F%89%E6%A0%91%E7%9A%84%E5%B1%82%E5%BA%8F%E9%81%8D%E5%8E%86.html#_116-%E5%A1%AB%E5%85%85%E6%AF%8F%E4%B8%AA%E8%8A%82%E7%82%B9%E7%9A%84%E4%B8%8B%E4%B8%80%E4%B8%AA%E5%8F%B3%E4%BE%A7%E8%8A%82%E7%82%B9%E6%8C%87%E9%92%88  
Learnt: 文章中写的是本题依然是层序遍历，只不过在单层遍历的时候记录一下本层的头部节点，然后在遍历的时候让前一个节点指向本节点就可以了 （！！！need help）
Difficulty during Implementation:  
Logic of Solution： 
AC Code:
```Python
      
```

Question: [117 populating next right pointers in each node ii](https://leetcode.com/problems/populating-next-right-pointers-in-each-node-ii/)
Outcome with Date: 12-05:
First Impression: 
Good Video/Blog: 
Learnt: 
Difficulty during Implementation:  
Logic of Solution： （！！！need help）
AC Code:

Question: [104 maximum depth of binary tree](https://leetcode.com/problems/maximum-depth-of-binary-tree/)
Outcome with Date: 12-05:Y      
First Impression: 我就用的层序遍历模版最后len一下 感觉肯定有更简单的写法 -》答案也是这么写的 
Good Video/Blog: 
Learnt: 
Difficulty during Implementation:  
Logic of Solution： 
AC Code:
```Python
# 我的
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
from collections import deque 

class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        results = []
        if root == None:
            return 0
        d = deque()
        d.append(root) 
        while d:
            size = len(d)
            result = []          
            while size:
                node = d.popleft()
                result.append(node.val)  
                size -= 1
                if node.left:
                    d.append(node.left)
                if node.right:
                    d.append(node.right)
            results.append(result)
        return len(results)
         
```
```Python
#答案
class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        if root == None:
            return 0
        
        queue_ = [root]
        result = []
        while queue_:
            length = len(queue_)
            sub = []
            for i in range(length):
                cur = queue_.pop(0)
                sub.append(cur.val)
                #子节点入队列
                if cur.left: queue_.append(cur.left)
                if cur.right: queue_.append(cur.right)
            result.append(sub)
            

        return len(result)
```

Question: [111 minimum depth of binary tree](https://leetcode.com/problems/minimum-depth-of-binary-tree/)
Outcome with Date: 12-05:X .    
First Impression:不知道怎么写 但是知道第一次没有left或者没有right的时候就时最短长度 但是不知道怎么拿到这个长度 -》需要注意的是，只有当左右孩子都为空的时候，才说明遍历的最低点了。如果其中一个孩子为空则不是最低点！！！
Good Video/Blog: https://programmercarl.com/0102.%E4%BA%8C%E5%8F%89%E6%A0%91%E7%9A%84%E5%B1%82%E5%BA%8F%E9%81%8D%E5%8E%86.html#_111-%E4%BA%8C%E5%8F%89%E6%A0%91%E7%9A%84%E6%9C%80%E5%B0%8F%E6%B7%B1%E5%BA%A6 
Learnt: 
Difficulty during Implementation:  
Logic of Solution： 
AC Code: （！！！need help 如下不是递归的写法吗 怎么做到这个效果的！！！）
```Python
class Solution:
    def minDepth(self, root: TreeNode) -> int:
        if root == None:
            return 0

        #根节点的深度为1
        queue_ = [(root,1)]
        while queue_:
            cur, depth = queue_.pop(0)
            
            if cur.left == None and cur.right == None:
                return depth
            #先左子节点，由于左子节点没有孩子，则就是这一层了
            if cur.left:
                queue_.append((cur.left,depth + 1))
            if cur.right:
                queue_.append((cur.right,depth + 1))

        return 0
         
```
------------------
Question: [226
Outcome with Date: 12-05:   
First Impression:  
Good Video/Blog: 
Learnt: 
Difficulty during Implementation:  
Logic of Solution： 
AC Code:
```Python
      
```

Question: [101
Outcome with Date: 12-05:   
First Impression:  
Good Video/Blog: 
Learnt: 
Difficulty during Implementation:  
Logic of Solution:
AC Code:
```Python

         
```

Question: [104
Outcome with Date: 12-05:   
First Impression:  
Good Video/Blog: 
Learnt: 
Difficulty during Implementation:  
Logic of Solution: 
AC Code:
```Python
      
```

Question: [559
Outcome with Date: 12-05:   
First Impression:  
Good Video/Blog: 
Learnt: 
Difficulty during Implementation:  
Logic of Solution: 
AC Code:
```Python

         
```

Question: [111
Outcome with Date: 12-05:   
First Impression:  
Good Video/Blog: 
Learnt: 
Difficulty during Implementation:  
Logic of Solution: 
AC Code:
```Python
      
```

Question: [222
Outcome with Date: 12-05:   
First Impression:  
Good Video/Blog: 
Learnt: 
Difficulty during Implementation:  
Logic of Solution: 
AC Code:
```Python

         
```