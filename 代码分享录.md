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

# Array
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
First Impression at 2nd: 有想法去遍历 但是不知道对应应该做什么操作 看了答案后自己写出来了   
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
Outcome with Date: 11-25:O, 12-06:O
First Impression: have idea that I can apply two pointers in left and right to compare each time to find the bigger one, but has some error  
First Impression at 2nd: 小错误 循环不变量 更新的时候left,right的位置放错了-》不是每时每刻都更新，只有当满足条件的时候才更新 
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
Outcome with Date: 11-25:X, 12-07: X
First Impression: don't know how to do  
First Impression at 2nd: 用了sort但是subarry应该是连续的 不能sort 然后也一点想法也没了 小唐说subarray是必须连续的 subsequence是可以不连续的-》暴力解法起码要知道两个for循环一个是开头的index 一个是结尾的index->滑动窗口的思路 用一个滑动窗口（一个for循环） 来做两个for循环的事情 那么这个for循环中的i 是开头指针 还是结尾指针呢 如果是开头 那么 【】还是一个个遍历结尾 和两层for循环就没区别了 所以这个for循环的i时结尾 当我们找到满足大于目标值得结尾index 我们再移动开头的位置 缩小这个长度-》时间复杂度：O(n) 空间复杂度：O(1) 一些录友会疑惑为什么时间复杂度是O(n)。不要以为for里放一个while就以为是O(n^2)啊， 主要是看每一个元素被操作的次数，每个元素在滑动窗后进来操作一次，出去操作一次，每个元素都是被操作两次，所以时间复杂度是 2 × n 也就是O(n)
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
Outcome with Date: 11-25:X, 12-07: X(还是很难写 下次还要看！！)  
First Impression: no idea  
First Impression at 2nd: 记住了最外一圈怎么做 但是不知道后面圈怎么走-》本质是个模拟题 很常考-》这题要用上循环不变量，这题是在一圈一圈的循环 一共填充n*n个数 count必须小于这个 ；这里的不变量定义成对于每条边的处理规则->解决了几圈的问题 就进入了循环 因为每一圈的位置都不是固定了 所以起始位置是一个变量 一直要变的 类似的终止位置 也是一直在变的 所以也要用变量 ;往里缩的时候也要注意方向 方向不一样可能要用减号
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

# Summarization for array
https://programmercarl.com/%E6%95%B0%E7%BB%84%E6%80%BB%E7%BB%93%E7%AF%87.html  
1. 经典题型： 二分法，双指针，滑动窗口，模拟行为  
2. 二分法：o(logn) 循环不变量loop invariant(只有这个条件为假，我们才跳出这个循环)注意区间的定义，保持这个定义，保持区间合法性  
3. 双指针：o(n)通过一个快指针和慢指针在一个for循环下完成两个for循环的工作  
4. 滑动窗口：o(n)主要要理解滑动窗口如何移动 窗口起始位置，达到动态更新窗口大小的，从而得出长度最小的符合条件的长度  
5. 模拟行为：相信大家有遇到过这种情况： 感觉题目的边界调节超多，一波接着一波的判断，找边界，拆了东墙补西墙，好不容易运行通过了，代码写的十分冗余，毫无章法，其实真正解决题目的代码都是简洁的，或者有原则性的，大家可以在这道题目中体会到这一点  
6. while 循环不变量  
7. backward iterate in loop: range(end, start-1, -1)  

# linked list
https://programmercarl.com/%E9%93%BE%E8%A1%A8%E7%90%86%E8%AE%BA%E5%9F%BA%E7%A1%80.html  
链表是一种通过指针串联在一起的线性结构，每一个节点由两部分组成，一个是数据域一个是指针域（存放指向下一个节点的指针），最后一个节点的指针域指向null（空指针的意思）。链表的入口节点称为链表的头结点也就是head。存储方式：上一章节的数组在内存中是连续分布的，但是链表在内存中不是连续分布的，链表是通过指针域的指针链接在内存中的各个节点，所以链表的节点在内存中不是连续分布的，而是散乱的分布在内存中的某地址上，分配机制取决于餐做系统的内存管理。

链表的定义  
```Python
class LinkNode:
    def __init__(self, value=0, next=None):
        self.value = value
        self.next = next
```
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
Outcome with Date: 11-27: X; 12-08: x -》〉 need do agian
First Impression: 总体上看上去不难，但是init那里不知道怎么写，看了下答案 -》循环不变量现在对我来说不难了 但是临界值很混乱 需要靠edge case；而且忘记了如果加减的是头节点那可能很不一样->不写dummy head很混乱 经常会空指针异常
First Impression at 2nd:不太记得了 add head那里就忘记了 记得要加dummy head这样有帮助 但是不知道怎么引用到头节点了-》答案是init一个额外的node class，再自己linked list调用的时候 也init一下得到头节点  
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
Outcome with Date:12-04:O， 12-10:x
First Impression: 自己一开始想用defaultdict,但发现most_common是Counter里的函数，而且我不知道这个most_common操作的时间复杂度是多少->o(nlogn)
2nd:学heap时重新看, 和后面1046对比 不知道为什么一个建立min heap一个建立max heap
Good Video/Blog: https://www.bilibili.com/video/BV1Xg41167Lz/ 
Learnt:(1)只求前k个高频之类的结果 -》sort n(logn)
Difficulty during Implementation: （1）不能import Counter from collections一定是from...import...  
Logic of Solution: 
(1) 先算counter (2) 一个个tuple加入 建立min heap 大于k了 就pop (3)剩下的k个在min-heap的就是1-kth maximum number 
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
```Python
# min-heap/priority heap ->现在看来不知道为什么要用min heap明明max heap更直接还是同样的效果
# 可能卡尔的核心是能保留kth下来
from collections import defaultdict 
import heapq as pq
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        num_freq = defaultdict(int)
        for num in nums:
            num_freq[num] += 1

        min_heap = [] #min_heap
        #用固定大小为k的小顶堆，扫面所有频率的数值
        for (num, freq) in num_freq.items():
            pq.heappush(min_heap, (freq, num))
            if len(min_heap) > k:#如果堆的大小大于了K，则队列弹出，保证堆的大小一直为k
                pq.heappop(min_heap)
        
        #小顶堆里剩下来的时最大的k个数 但是次序颠倒的
        result = [0]*k
        for i in range(k-1, -1, -1):
            result[i] = pq.heappop(min_heap)[1] # val
        return result
```
```Python
# leetcode官方解答是 直接max-heap 用nlargest的method
from collections import Counter
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]: 
        # O(1) time 
        if k == len(nums):
            return nums
        
        # 1. build hash map : character and how often it appears
        # O(N) time
        count = Counter(nums)   
        # 2-3. build heap of top k frequent elements and
        # convert it into an output array
        # O(N log k) time
        return heapq.nlargest(k, count.keys(), key=count.get) 
```
```Python
# max-heap
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        counter = Counter(nums)
        
        heap = [(-v, k) for k, v in counter.items()]
        heapify(heap)
    
        return [heappop(heap)[1] for i in range(k)]
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
4. （a）确定递归函数的参数和返回值：确定哪些参数是递归的过程中需要处理的，那么就在递归函数里加上这个参数，并且还要明确每次递归的返回值是什么进而确定递归函数的返回类型 （b）确定终止条件：写完了递归算法，运行的时候，经常会遇到栈溢出的错误。如果递归没有终止，操作系统的内存栈必然就会溢出（c）确定单层递归的逻辑：确定每一次层递归需要处理的信息。在这里也就会重复调用自己来实现递归的过程 (内含：超级操作：递归调用；微操作：递归到当前层)

### 递归遍历总结性笔记
1. 1209 记录在ting_backtracking里

Question: [144 binary tree preorder traversal](https://leetcode.com/problems/binary-tree-preorder-traversal/)  
Outcome with Date: 12-04:X， 12-09:X  
First Impression:第一次自己写还是不太会
First Impression at 2nd: 还是不会写递归遍历 （1）我好像是递归的参数和返回值不懂 仔细看了下python的pass by assignment机制 指的是immutable的object 在call的时候 外部的一直不会被改(这类似于与pass by value 但是如果穿入的是mutable object他就会被改 when we change the mutable object we didn't change the indentiy but only the content->推荐递归不要用mutable object 要用的话 记得copy成另一个东西) mutable object 在call (2)确认我想要递归的超级函数返回什么 他和主递归函数是在做同一件事前 只是解决小一点的问题再回溯回去 所以他们的返回值要是统一的！
Good Video/Blog:每次写递归都要靠直觉？ 这次带你学透二叉树的递归遍历 https://www.bilibili.com/video/BV1Wh411S7xt/?vd_source=8b4794944ae27d265c752edb598636de https://mathspp.com/blog/pydonts/pass-by-value-reference-and-assignment
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
```Python
# 12-09
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        self.path = []
        if root is None: return self.path 
        self.dfs(root)
        return self.path
    
    def dfs(self, node: TreeNode) -> None:
        # traverse the nodes, if valid add to solutions
        if node is not None:
            self.path.append(node.val)
        if node.left:
            self.dfs(node.left)
        if node.right:
            self.dfs(node.right)
        return
```
Question: [145 binary tree postorder traversal](https://leetcode.com/problems/binary-tree-postorder-traversal/)  
Outcome with Date: 12-04:X, 12-09:y  
First Impression: 知道思路了 结果返回集，定义递归函数，调用递归函数，返回结果
First Impression at 2nd time: 还是  
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
```Python
# 12-09
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def postorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        self.path = []
        if root is None: return self.path
        self.dfs(root)
        return self.path
        
    def dfs(self, node: TreeNode) -> None:
        # traverse all the node, store the valid answer to solution
        if node.left:
            self.dfs(node.left)
        if node.right:
            self.dfs(node.right)
        if node: 
            self.path.append(node.val)
```

Question: [94 binary tree inorder traversal](https://leetcode.com/problems/binary-tree-inorder-traversal/)  
Outcome with Date: 12-04: Y, 12-09: Y  
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
## Day 13

Question: [116 populating next right pointers in each node](https://leetcode.com/problems/populating-next-right-pointers-in-each-node/)
Outcome with Date: 12-06:X
First Impression: 我以为是层讯遍历然后再每次节点展开的时候做一个next指针赋值的操作
Good Video/Blog: 官方的解答更清楚
Learnt: (1)在每一层第一记录size的时候做操作，后面子树就展开了 操作就会乱 刚被pop的next指向队列中的头节点（此时头节点就是他右边的数）当没有头节点的时候 正好指向的就是none(2)这个#号表达 不是要自己实现的 自己实现的是整个数的编译，这个井号只是告诉你哪一层是哪一层 -》看来这道题可以问的是你想要怎样的输出，我应该输出头节点 还是你想要知道我的遍历顺序
Difficulty during Implementation:  当root为空时返回的也不是之前模板的results了 是这个空root
Logic of Solution： 
AC Code:
```Python
"""
# Definition for a Node.
class Node:
    def __init__(self, val: int = 0, left: 'Node' = None, right: 'Node' = None, next: 'Node' = None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next
"""
class Solution:
    def connect(self, root: 'Optional[Node]') -> 'Optional[Node]':
        if root == None:
            return root 
        from collections import deque
        d = deque()
        d.append(root)
        depth = 0
        while d:
            size = len(d) 
            for i in range(size):
                node = d.popleft()
                if i < size-1:
                    node.next = d[0]
                if node.left:
                    d.append(node.left)
                if node.right:
                    d.append(node.right)
        return root
```


Question: [117 populating next right pointers in each node ii](https://leetcode.com/problems/populating-next-right-pointers-in-each-node-ii/)
Outcome with Date: 12-06: Y
First Impression: 我感觉解题思路和perfect binary tree一样 就没有变-》对的
Good Video/Blog:  
Learnt: 
Difficulty during Implementation:  
Logic of Solution： 
AC Code:
```Python
 # same as above     
```



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
#### END OF bfs 

## Day 13 
###  第二次学习树dfs, bfs
1. 昨天感觉刷的不好 理解不好 今天打算换一个视频看基础（确实dfs适合递归，bfs适合迭代） + 再看一遍随想录视频
2. https://turingplanet.org/2020/07/12/%e6%a0%91tree%e6%b7%b1%e5%ba%a6%e4%bc%98%e5%85%88%e6%90%9c%e7%b4%a2dfs%e8%a7%a3%e9%a2%98%e5%a5%97%e8%b7%af%e3%80%90leetcode%e5%88%b7%e9%a2%98%e5%a5%97%e8%b7%af%e6%95%99%e7%a8%8b9%e3%80%91/ DFS/depth-first tree
3. tree是类似linked list的概念，内存中不一定连续的数据，由各个节点的reference串起来组成。节点可以分成parent和child两类，只有一个入口root，可以看作一个特殊的无环无向图; dfs是一种递归形势的搜索方式，偏向垂直的概念，向下扎；dfs递归三部曲模板：(a)base case (b)do somehting (c)recurse for subproblems要相信自己的递归是对的 后两部顺序可以遍 就是看你的遍历顺序；top down dfs: 把值通过参数的形势从上往下传，一般dfs本身不返回值，bottom up dfs更难也更常见，把值从下(subproblem)往上传，当前递归层利用subproblem传上来的值计算当前层的新值并返回，一定会有返回值；bottom up dfs (对于递归问题，建议大家找一个中间状态建立自己的思维，并且要相信自己的递归)general step(a)base case(b)向子问题要答案return value(c)利用子问题的答案构建当前问题(当前递归层)的答案(d)若有必要，做一些额外操作(e)返回答案（给父问题）
4. https://turingplanet.org/2020/07/12/%e6%a0%91tree%e9%a2%98%e5%9e%8b%e5%b9%bf%e5%ba%a6%e4%bc%98%e5%85%88%e6%90%9c%e7%b4%a2bfs%e5%a5%97%e8%b7%af%e3%80%90leetcode%e5%88%b7%e9%a2%98%e5%a5%97%e8%b7%af%e6%95%99%e7%a8%8b8%e3%80%91/ BFS/breath-first tree
5. BFS是层的概念进行的搜索算法，对于图来说就是一圈一圈的，适合解决与层数相关的题目。bfs十分简单，理由这个queue的模版就行了，bfs迭代模版(a)利用queue记录所有的遍历元素要被展开的节点 先进先出 initialize queue with ann entry points对于二叉树只有一个入口，对于图就有多个 basecase的多个都要写进去(b)while queue is not empty(b.1)for each node in the queue (currently)当前层有多少个就记录下来 因为第三步会展开数 所以这里的记录很重要 (b.2)poll out the element(add to result)(b.3)expand it, offer children to the queue in order对于树来说拿到一个节点 展开他的意思就是展开他的子树们(b.4) increase level

Question: [226 invert binary tree](https://leetcode.com/problems/invert-binary-tree/)  
Outcome with Date: 12-06: x（题目说要优先掌握递归）, 12-07: X 
First Impression: 题目看起来简单 但是我不知道怎么去交换左右节点 直觉是要用层序迭代遍历 但要是递归的话 又怎么写？对！好像是这个问题 之前的遍历都没有用到指针 这里要变化指针 那么怎么高呢（!!need help 递归方法， 特别是我要知道中序遍历的顺序，这对理解很重要->博主不建议写中旬 会给自己挖坑）  
First impression at 2nd: 我知道了就是对于每个节点交换他们的左右节点 但是对于递归的模版理解错了 这里不是普通的搜索 是自己的例子 所以要按照base case, 遍历顺序这个步骤走，由于这题返回的是根节点 所以对内部具体遍历顺序没有要求 用前序或者后续 都可以 中序的话 比较麻烦 已经前面已经改动了 比如前序例子 对中节点先操作交换节点 再向左遍历 向右遍历 什么时候是终止条件呢 当遍历到叶子节点 就要返回 再去好好理解几题 急不得
Good Video/Blog: https://www.bilibili.com/video/BV1sP4y1f7q7/?vd_source=8b4794944ae27d265c752edb598636de https://programmercarl.com/0226.%E7%BF%BB%E8%BD%AC%E4%BA%8C%E5%8F%89%E6%A0%91.html#%E9%80%92%E5%BD%92%E6%B3%95  
Learnt: (1)视频讲的是dfs递归方法这题适合前序和后序，用中序会比较绕 用递归 想起递归三部曲 确定返回值参数 确定终止条件 处理逻辑（前 中 后序遍历之前就要确定一个）（2）bfs的解法在代码随想录pdf中,解法其实很简单 只要交换一下就好了 我是不是对这个栈的存储形势没有好好理解 一开始栈里面安排的是这一层从左到右的节点 pop出来的时候 给他们交换下节点 就可以了？？（！！！need help 还需要理解下）
Difficulty during Implementation:  
Logic of Solution： 
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
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if root == None:
            return root 
        d = deque()
        d.append(root)
        while d:
            size = len(d)
            for i in range(size):
                node = d.popleft()
                node.left, node.right = node.right, node.left
                if node.left:
                    d.append(node.left)
                if node.right:
                    d.append(node.right)
        return root    
```

Question: [101 symmetric tree](https://leetcode.com/problems/symmetric-tree/)  
Outcome with Date: 12-06: x （题目说要优先掌握递归) 12-08:X
First Impression: 层序遍历popleft和pop比较 但是自己不能两个都pop因为模板只pop一个 那么我是不是还要塞回去？或者有没有peek方法？这个不行 因为我要保证每一层都能匹配上 但是我每次pop的时候只有一个数 就会不匹配左右-》其实和上一题是类似的 反转下左右子树 是否一样 这题考查的就是对于两个二叉树遍历比较的情况  
First Impression at 2nd: 遍历顺序是选对了是后序遍历 但他其实不是严格的后续遍历 返回值是bool 基础判断的条件是对的 但是基础判断要做的事 考虑的事如果我们只有这些简单的cases 返回值是什么 他的返回值也要和递归问题同源（同种类型）单层递归逻辑里面 做的是个判断 左数的从外到内 和右树的从外到内是否一致
Good Video/Blog: https://www.bilibili.com/video/BV1ue4y1Y7Mf/?vd_source=8b4794944ae27d265c752edb598636de  
Learnt: （1）递归dfs的话 要确定遍历顺序 一定要好好想 这确定了理解的深度-》此题只能用后序(左右中)bottom up收集完左右孩子的信息 上一级才能有这个消息 （什么题目一定要求后序：要收集子孩子信息 才能向上一级返回）这里的递归参数就是左子树和右子树的节点 终止条件 判断什么时候return false, true 就要分类（2）后序可以理解成一个回溯  
Difficulty during Implementation:  
Logic of Solution:
AC Code: （!!!need help层序遍历不会写）
```Python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        if root is None: return True
        if root.left is None and root.right is None: return True 
        if root.left is not None and root.right is None: return False
        if root.right is not None and root.left is None: return False 
        return self.dfs(root.left, root.right)
    
    def dfs(self, left: TreeNode, right: TreeNode) -> bool:
        # base case: 一个根节点 只有左右两个叶子节点
        if left is None and right is not None: return False
        elif left is not None and right is None: return False
        elif left is None and right is None: return True 
        else: 
            if left.val != right.val: return False 
            else:
                # 左右中 才能返回给母节点
                res_left = self.dfs(left.left, right.right)
                res_right = self.dfs(left.right, right.left)
                # aggregate to middle point
                if res_left is True and res_right is True:
                    return True 
                else:
                    return False
        
```

Question: [104 maximum depth of binary tree](https://leetcode.com/problems/maximum-depth-of-binary-tree/)  
Outcome with Date: 12-06: X， 12-09: O
First Impression:按卡尔的建议 在用递归的写法写下 但是不知道怎么写停止条件 我总感觉这里我需要指针-》如果用球高度的方法 从下往上 左右中 后序遍历， 终止条件就是碰到了空间点了 空节点的告诉就是0 中节点就是处理节点 返回的是左子树和右子树的最大值+1（加一：当前父节点的高度）
First Impression at 2nd: 基本写出来 但是忘记考虑left_depth和右边为零的情况
Good Video/Blog: https://www.bilibili.com/video/BV1Gd4y1V75u/?vd_source=8b4794944ae27d265c752edb598636de    
Learnt:(1)什么是深度？高度？深度： 二叉树中任意一个节点到根节点的距离 要用前序遍历（中左右） 从上往下 中是我们的处理过程 我们往下遍历一个就加一 这符合我们求深度的过程；高度：二叉树任意一个节点到叶子节点的距离 要用后序遍历（左右中） 从下往上 中放在最后也是处理逻辑在做后，我们就可以把叶子节点的高度返回给其父节点 父节点知道他的叶子节点后 我的父节点直接加个一就可以了-》求深度：也可以用根节点的高度（后序 往上长 发芽的过程）
Difficulty during Implementation: (1)为什么要把helper function写成类里的一个程序？->小唐说这随便 选一个习惯的就好 类里的一个函数叫method （我还是不要写成method 我有时会忘记self给自己挖坑）（2）注意这个getheight的逻辑 下一个节点不为空 里面的递归使用helper function就不用做这个判断为不为空？  
Logic of Solution: （！！need help）
AC Code:
```Python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        return self.getHeight(root, 0)

    def getHeight(self, node: TreeNode, height: int):
        if node == None:
            return 0
        leftheight = self.getHeight(node.left, height+1)
        rightheight = self.getHeight(node.right, height+1)
        return 1+max(leftheight, rightheight)
      
```

Question: [559 maximum depth of nary tree](https://leetcode.com/problems/maximum-depth-of-n-ary-tree/)
Outcome with Date: 12-05: X， 12-09: X  
First Impression: 我直接用了上面的code 我感觉没区别 但是报错maximum recursion depth exceeded in comparison->这里变成了children没有左右孩子了
First Impression at 2nd: 忘记了怎么处理children node
Good Video/Blog: 
Learnt: 
Difficulty during Implementation:这个需要理解一下 不太理解为什么不需要height作为参数也可以 可能是都可以的写法（！！need help）
Logic of Solution: 逻辑是 既然不分左右子树 那么没关系 对每个child 算max
AC Code:
```Python
"""
# Definition for a Node.
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children
"""

class Solution:
    def maxDepth(self, root: 'Node') -> int:
        return self.getHeight(root)

    def getHeight(self, node: 'Node') -> int:
        if node == None:
            return 0
        height = 0
        for i in range(len(node.children)):
            height = max(height, self.getHeight(node.children[i]))
        return 1+height         
```

Question: [111 minimum depth of binary tree](https://leetcode.com/problems/minimum-depth-of-binary-tree/) 卡尔建议 用递归写法  
Outcome with Date: 12-06:X， 12-09: X    
First Impression: 感觉逻辑和上面也没啥区别 换成min就行 不知道为什么 也是报错 -》先看视频讲解，和最大深度 看似差不多，其实 差距还挺大，有坑 -> 是定义有错 
First Impression at 2nd: 还是定义上踩了坑 如果他的一遍就根本没有叶子节点 不能用这边算最小长度; 不知道怎么写
Good Video/Blog: https://www.bilibili.com/video/BV1QD4y1B7e2/?vd_source=8b4794944ae27d265c752edb598636de  同学的python code更符合我的书写逻辑 https://blog.csdn.net/weixin_47617361/article/details/128141286?csdn_share_tail=%7B%22type%22%3A%22blog%22%2C%22rType%22%3A%22article%22%2C%22rId%22%3A%22128141286%22%2C%22source%22%3A%22weixin_47617361%22%7D  
Learnt: (1)题目中给的minimun depth: the number of nodes along the shortest path from the root node down to the nearest leaf node根节点到最近的叶子节点的距离->一定要从叶子节点开始 所以是算叶子节点的高度！所以要做个判断 根节点如果有一方没有子节点 那么要去计算另外一方的最小高度(2)递归的终止条件 我们这里写的是 当遇节点遍历到叶子节点的再下一个即是空节点的时候 他的高度是0（高！度！）-》但是也可以写成其他的终止条件 见下 
Difficulty during Implementation:  (！！！need help编译错误 而且不知道什么时候加height的参数什么时候不加->感觉不用加 因为底层的下一个为0嘛 之后都在叠加一)(1)不写成class method的话 一定要先定义 才能调用 (2)注意错过好多次！！之前代码里应该也有问题！！！ node.left == None (一般不会写== None 因为none是个特殊的值 所以一般用is None 不会去判断值是否相等 而是判断是否就是nonetype) 和 if not node.left (3) if ! node.left 和 if not node.left-> not的是作用在if上 这个node.left是没有！操作符
Logic of Solution: 
AC Code:
```Python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
# 小唐帮忙debug的
class Solution:
    def minDepth(self, root: Optional[TreeNode]) -> int:     
        def getHeight(node: TreeNode) -> int:
            """Find the minimum depth start from node"""
            if node is None:
                return 0
            leftDepth = getHeight(node.left)
            rightDepth = getHeight(node.right)
            if leftDepth and rightDepth:
                return 1 + min(leftDepth, rightDepth)
            else:
                return leftDepth + rightDepth + 1 #如果leftdepth is none， 那么这个值也是0 同一个概念 

        return getHeight(root)    
```
### 树的递归dfs写法总结
1. 对于树的高度：要不要传入depth(当调用的时候就是helper（root, 0）)，这个与下面的终止条件的定义也是相关的
2. 递归里面终止条件 也有两种写法（a）判断传入none的时候 即是叶子节点在下一个 这个时候就要终止了 因为叶子节点的高度为1 所以此处的高度为0（b）如果没有判断这个none那么就要判断是否有if node.left, if node.right 在做对应的左右子树的调用
3. 第一点要不要穿入树的高度 在（a）就不用 因为知道了base case是0，在(b)就一定需要
4. 递归写法可以写在class里的method 但是我会容易忘记self.也可以内部调用 但是就要注意 先定义 才能调用
5. 卡尔推荐的是所有题目用递归写 因为写法简单 就是一个递推公式 第二次刷的时候再用迭代

Question: [222 count complete tree nodes](https://leetcode.com/problems/count-complete-tree-nodes/)  
Outcome with Date: 12-06: X, 12-09:X 
First Impression: 按照老师说的去理解递归写法 但是写出来的逻辑 比要求的数大很多 怀疑自己哪里有重复计算 但是也不知道具体在哪-》一个小错误 但也很重要 left那边不用加一 因为是只有计算到parent时才加一
First Impression at 2nd: 知道用满二叉树性质 但是不知道怎么dfs算出后面缺少几个节点-》满二叉树的逻辑还是不会写 放弃, 尝试写了iteration写法
Good Video/Blog: https://www.bilibili.com/video/BV1eW4y1B7pD/?vd_source=8b4794944ae27d265c752edb598636de  
Learnt: （1）complete binary tree: except leaf node all full and it starts with left 
Difficulty during Implementation:  
Logic of Solution: （这个剪枝的方法之后也要会！！！先跳过bit manipulation）
AC Code:
```Python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def countNodes(self, root: Optional[TreeNode]) -> int:
        if root is None:
            return 0
        def count(node: TreeNode) -> int:
            if node == None:
                return 0
            # recursive + postorder(left,right -> middle)
            left_count = count(node.left) #这里一开始+1了所以报错 但不是这个逻辑 只有middle的时候 我们才要加自己本身
            right_count = count(node.right)
            return left_count + right_count + 1
        return count(root)         
```
```Python
# 上面是time o(n)
# full binary tree满二叉树的话 节点数量：(2^depth)-1
# 利用完全二叉树的特性: 子树是满二叉树的话 直接用公式计算 =满左子树（只遍历了两侧的节点 如果相等的话就是满二叉树 没有遍历所有节点）+满右子树+1 -》判断是不是满二叉树也是写在终止条件里
# 没有自己写 直接复制的代码
class Solution:
    def countNodes(self, root: TreeNode) -> int:
        if not root:
            return 0
        return 1 + self.countNodes(root.left) + self.countNodes(root.right)     
```
```Python
# 迭代法
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def countNodes(self, root: Optional[TreeNode]) -> int:
        if root is None: return 0
        from collections import deque 
        d = deque()
        d.append(root)
        result = 0
        while d:
            size = len(d)
            while size:
                node = d.popleft()
                result += 1
                if node.left:
                    d.append(node.left)
                if node.right:
                    d.append(node.right)
                size -= 1
        return result
```

Question:[110 balanced binary tree](https://leetcode.com/problems/balanced-binary-tree/)  
Outcome with Date: 12-06:X , 12-09:X  
First Impression: 这个balanced的定义我都不知道：balanced binary search tree平衡二叉搜索树（全部左子树和右子树的高度绝对值的差不能大于1）-》意思是maximun depth和子节点的minimun depth《=1对吧 我先按这个逻辑试着写下代码 不是这个意思 是所有的 我看是直接看下解答吧->自己还是写了 不会递归函数的返回值 不知道是depth呢还是bool  
First Impression at 2nd: 1）balanced binary tree 任何一个节点 左右子树的高度不超过1 (2)知道了概念也不会写 不知道怎么算高度差 怎么遍历-》看了答案觉得很妙 用后序遍历 算书的高度 返回给中节点 如果左右数之差大于1 直接返回-1
Good Video/Blog: https://www.bilibili.com/video/BV1Ug411S7my/  
Learnt: （1）balanced binary tree 任何一个节点 左右子树的高度不超过1 （！不是深度！）(2)卡尔递归写的返回数是int 节点的高度 他中间会做个判断 一旦不平衡就返回-1 这样就知道不满足了 （2）终止条件：看是不是空节点 
Difficulty during Implementation:  
Logic of Solution: 
1. 递归终止条件还是遇到空间点
2. 按后序遍历，左右子树分别计算的时候 发现已经不是平衡二叉树 就可以终止 所以这里调用的可以判断是不是-1 然后操作中节点的时候 也是判断左右相差是否《1 如果小于1返回的就是此时的高度max+1!!!
AC Code:
```Python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        # dfs, recursive, postorder(left, right ->  mid)/bottom up
        # require every maximun depth - minmun depth <= 1
        def countChildren(node: TreeNode) -> int:
            '''given a node, return the depths of its left tree and its right tree'''
            if node is None:
                return 0 
            left_height = countChildren(node.left)
            if left_height == -1:
                return -1
            right_height = countChildren(node.right)
            if right_height == -1:
                return -1
            if abs(right_height - left_height) > 1:
                return -1
            else:
                return 1 + max(right_height, left_height)
        if countChildren(root) != -1:
            return True
        return False
```
## Day 14

Question:[257 binary tree paths](https://leetcode.com/problems/binary-tree-paths/)  
Outcome with Date: 12-07: X, 12-09: X   
First Impression: 不知道怎么记录走过的道路-》“这是大家第一次接触到回溯的过程， 我在视频里重点讲解了 本题为什么要有回溯，已经回溯的过程。 如果对回溯 似懂非懂，没关系， 可以先有个印象” 其实递归和回溯是相辅相成的 只要有递归就一定有回溯-》back tracking是dfs的暴力解法
Frist Impression at 2nd: 这是dfs search模版 最基础的那个 不是base case而是答案valid时 加入到solution 
Good Video/Blog: https://www.bilibili.com/video/BV1ZG411G7Dh/ 
Learnt:(1)这里的递归法 要用前序遍历 让父节点指向孩子节点 这样才能把这个路径 按照top down输出出来（2）回溯：用一个容器记录从头到尾的一条路径 再一个个弹出子节点（此处就是回溯过程）回到根节点 再一个个弹入其他路径的节点 （3）递归的三部曲：参数 节点和pass用来记录单条路径的和result数组放的是所有路径结果 终止：收集路径的过程 到叶子节点就行 没必要到空节点（node.left and node.right都为空 代表遍历到叶子节点）然后把pass放进results就行, 操作：前序 中左右 中进行的操作就是把每个节点 放进pass里-》我是不是可以换个角度理解 因为是dfs所以一定所有边都会遍历 （想象那个遍历的动态图）而且也只会每个节点遍历1次 dfs就是一个扫描所有路径的过程
Difficulty during Implementation:  (1)用递归又忘记了需要先定义 在调用（！！！！错了三遍）
Logic of Solution: (!!!need help)
AC Code:
```Python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def binaryTreePaths(self, root: Optional[TreeNode]) -> List[str]:
        def dfsFindPath(node: TreeNode, path: str, result: List[str]) -> list[str]:
            # 中：
            path += str(node.val) #这里写错了 要累加 所以也是+=
            #终止条件 不用到空字符串 到空字符串的前一个就行
            #if (node.left and node.right) is None: #这样居然不可以！！这样的话 是left或者right是none可能又个优先级的问题
            if node.left is None and node.right is None:
                #做的操作是append路径到结果集
                result.append(path)
            # preorder: 中左右
            if node.left:
                dfsFindPath(node.left, path + '->', result)
            if node.right:
                dfsFindPath(node.right, path + '->', result)
            return result
        path = ''
        result = []
        if not root: return result
        results = dfsFindPath(root, path, result)
        return results
```

Question:[404 sum of left leaves](https://leetcode.com/problems/sum-of-left-leaves/)
Outcome with Date: 12-07: X, 12-09: X
First Impression: 我以为就是一个前序dfs记录左叶节点的值之和
First Impression at 2nd: learnt: 用dfs search模版 只用关valid case 
Good Video/Blog: https://www.bilibili.com/video/BV1GY4y1K7z8/?vd_source=8b4794944ae27d265c752edb598636de
Learnt: （0）这题定义的是 所有叶子节点左边的那个 所以是可以大于 两个的 需要向面试官确认（1）判断条件有两个 必须要是叶子节点 必须要是其母节点的左孩子（2）设计想要的元素 要注意 node.left is none and node.right is none 只能告诉是叶节点 并不能告诉你这是左叶子节点！！-》用他的母节点来帮助 左孩子要不为空 左孩子的左右孩子为空 （3）这道题用后序遍历 先收集左子树的左叶子节点之和 右子树的左叶子节点之和 再返回
Difficulty during Implementation:  （1）！！！错了第四次！！！递归函数 先定义 在调用
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
    def sumOfLeftLeaves(self, root: TreeNode) -> int:
        if not root: 
            return 0
        # 用后序 把左子树的左叶子节点之和 右子树的左叶子节点之和 返回去
        left_left_leaves_sum = self.sumOfLeftLeaves(root.left)  # 左
        right_left_leaves_sum = self.sumOfLeftLeaves(root.right) # 右
        # 中 判断想要收集的元素 是不是我们想要的左叶子节点
        cur_left_leaf_val = 0
        if root.left and not root.left.left and not root.left.right: 
            cur_left_leaf_val = root.left.val 
        # 递归函数本身的调用 有返回值 所以左边右边调用也会有返回值   
        return cur_left_leaf_val + left_left_leaves_sum + right_left_leaves_sum # 中
```

Question:[513 find bottom left tree value](https://leetcode.com/problems/find-bottom-left-tree-value/)  
Outcome with Date: 12-07:X  
First Impression: 知道了应该判断这些条件 在终止条件中 但是居然不知道给终止条件 要返回个什么值（这里其实要做个比较 到了叶子节点 判断当前深度（函数参数）有没有比最大深度（全局变量）大 是的话 就更新全局变量 同时记录节点 这样才能返回 题目需要的） （单层遍历逻辑 就是需要按照左右的方向进行递归 有个回溯的过程） （1）找到叶子节点们 （2）找到左叶子节点们 （3）找到最深的左叶子节点（1个或者多个）->同样深度 返回最左边的  发现自己原来动定义错了 要求是要在last row里面leftmost value  return the leftmost value in the last row of the tree ->所以可以是右节点的
Good Video/Blog:https://www.bilibili.com/video/BV1424y1Z7pn/?vd_source=8b4794944ae27d265c752edb598636de->视频说没有中的处理逻辑 那么前中后序都可以 讲的好 我第一次仔细听他怎么讲回溯的   
Learnt: （1）因为是找到要求层的某个值 所以用层序遍历很简单
Difficulty during Implementation: （1）汗两天不到层序的模版都忘了（！！！need rememeber again） （2）while循环不变量 需要自己更新值啊！要不然是永远满足的(3)这题前中后序都可以 因为中序位置在哪都不重要 不重要的话单层循环逻辑里面也不用写 这里只要强调左在右边就可以了
Logic of Solution: 视频里的（1）递归 什么顺序都可以 没有中的处理逻辑 参数返回值 参数是count（目标值的减减）返回值是bool 因为我们需要一有返回值就返回 终止条件：到子节点的时候count为0返回真 到子节点的时候count不为0返回假 单层逻辑（含回溯的写法）如果左边有left count减去left值 递归调用 用的还是count 之后再count加回left（不含回溯的写法）直接递归调用 参数是count减去left值 因为参数的减不会影响下面的coutn的本身的值 这里内涵了回溯的过程；因为我们需要 找到路径后继续向上报告我们得到了这个值 所以这里就要判断是否有true;左右子树都没return true我么就return false 
AC Code:
```Python
## 层序遍历
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def findBottomLeftValue(self, root: Optional[TreeNode]) -> int:
        # bfs 
        if root is None: 
            return 0
        if root.left is None and root.right is None: 
            return root.val 
        from collections import deque
        d = deque()
        d.append(root)
        results = []
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
        return results[-1][0]
```
```Python
#递归写法 没有自己写
class Solution:
    def findBottomLeftValue(self, root: TreeNode) -> int:
        max_depth = -float("INF")
        leftmost_val = 0

        def __traverse(root, cur_depth): 
            nonlocal max_depth, leftmost_val
            if not root.left and not root.right: 
                if cur_depth > max_depth: 
                    max_depth = cur_depth
                    leftmost_val = root.val  
            if root.left: 
                cur_depth += 1
                __traverse(root.left, cur_depth)
                cur_depth -= 1
            if root.right: 
                cur_depth += 1
                __traverse(root.right, cur_depth)
                cur_depth -= 1

        __traverse(root, 0)
        return leftmost_val
```

Question:[112 path sum](https://leetcode.com/problems/path-sum/submissions/856428149/)  
Outcome with Date: 12-07: X, 12-09: 0 
First Impression: (1)这里有个reference类别回溯里常碰到的问题 请看下面代码（2）找path的话 对中节点的操作 要放在终止条件前面！！小唐教的 递归写法要小心！（1）需要改动的参数 都放成递归的参数 要不然他有可能会被后续的参数给一起更新 因为是reference类型的数（2）推荐写成method的class不要分开写 要不然还是有调用时 复制参数 里外值得不是同一个东西的问题 (3)要返回一个list of list类型的话 是写成这个形式List[list]->其实是这样List[]这是一个函数？List[List[int]]
First Impression at 2nd: 差不多写出来了 但是忘记了 需要剪去root.val
Good Video/Blog: https://www.bilibili.com/video/BV19t4y1L7CR/?vd_source=8b4794944ae27d265c752edb598636de->视频说没有中的处理逻辑 那么前中后序都可以 录友的python解法：https://blog.csdn.net/wh1234546/article/details/128179751?csdn_share_tail=%7B%22type%22%3A%22blog%22%2C%22rType%22%3A%22article%22%2C%22rId%22%3A%22128179751%22%2C%22source%22%3A%22wh1234546%22%7D 
Learnt: 
Difficulty during Implementation:  (1)写成class method的话 是一模一样的 也需要回溯 （2）如果没有对中节点的操作的话 一开始外函数调用的时候 是都就需要先剪去root的值呢
Logic of Solution: 
AC Code:
```Python
###path回溯一个重要的code!!!!
###对后面回溯的题目很重要！！！
class Solution:
    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> list:
        # dfs, preorder, path_sum
        def dfs(node: TreeNode, path: list, path_list: list) -> list:
            path.append(node.val) #这里写错了要注意 先中 再append 要不然漏节点了
            if node.left is None and node.right is None:
                path_list.append([x for x in path]) #reference类别（除了int,str,floot,bool）如果有后续操作修改 它本身也会跟着修改（他是一个pointer）!!
            if node.left:#所以这里要复制一遍
                dfs(node.left, path, path_list)
                path.pop()#这里需要回溯一下
            if node.right:
                dfs(node.right, path, path_list)
                path.pop()#这里需要回溯一下
            return path_list
        if root is None:
            return False 
        path = []
        path_list= []
        print(dfs(root, path, path_list))#257那题没错 不用回溯 也不用复制 是因为他是一个str非reference类别
```
```Python
## 递归写法要小心！
## （1）需要改动的参数 都放成递归的参数 要不然他有可能会被后续的参数给一起更新 因为是reference类型的数
## （2）推荐写成method的class不要分开写 要不然还是有调用时 复制参数 里外值得不是同一个东西的问题
class Solution:
    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> list:
        # dfs, preorder, path_sum
        def dfs(node: TreeNode, path: list, path_list: list, path_sum: int) -> bool:
            nonlocal flag #把他写外面
            path.append(node.val)
            if node.left is None and node.right is None:
                path_list.append([x for x in path]) 
                path_sum = sum(path)
                if path_sum == targetSum:
                    flag = True #boolen值这些东西直接return不了 本来我想的逻辑时当发现了sum是等于target sum就return True
                    #小唐说bool只是传到上一层 不能直接传到最外面 不能这么简单的写 所以放在参数里更简单
                    print("True")
            if node.left:
                dfs(node.left, path, path_list, path_sum)
                path.pop()
            if node.right:
                dfs(node.right, path, path_list, path_sum)
                path.pop()
        if root is None:
            return False 
        path = []
        path_list = []
        path_sum = 0
        flag = False
        dfs(root, path, path_list, path_sum) #调用传参的时候 都会复制一遍 所以在里面改bool值外面也接受不到 但reference类型的参数的时候 复制一遍 外面会变 写成method的class
        #定义了用的就是本身self.什么 就不会有这个问题
        return flag
```
```Python
# 我来写一个干净的class method来总结这道题的题解
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        if root is None:
            return False 
        path, path_list, path_sum, self.flag = [], [], 0, False
        self.dfs(root, path, path_list, path_sum, targetSum)
        return self.flag
    
    def dfs(self, node: TreeNode, path: list, path_list: List[list], path_sum: int, targetSum): #小唐说path_list可以删掉
        path.append(node.val) #这个问题是函数条件不成立没有返回值-》这里是dfs不需要返回值 条件成立我就吧外面值修改了
        if node.left is None and node.right is None:
            path_list.append([x for x in path]) 
            if sum(path) == targetSum:
                self.flag = True #要是把他放入参数的话 其实是新定义一个z=flag 吧z当flag用 那么修改z flag不会变-》这是封装的特性（参数都是这样）不希望函数影响外面的东西
                # 哪怕是穿入的变量也不想要改参数的本身，然后int哪些不是reference类别的话 是他们数据结构的特性 等到了函数里的参数 也是要符合封装性
                # self.的实现是一个指针 把他变成了reference类型（address）类似的吧flag变成长度为1的list也行 即要flag[0] == True这样里面的也会改
        if node.left:
            leftNode = self.dfs(node.left, path, path_list, path_sum, targetSum)
            path.pop() #还是同一个理由 reference的list类型 手动需要回溯过去
        if node.right:
            rightNode = self.dfs(node.right, path, path_list, path_sum, targetSum)
            path.pop()
```
```Python
# 录友里答案的写法
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        if root is None: return False
        return self.findPath(root, targetSum-root.val) #穿入的substract应该不包含root.val
    # 写成class的method方便之后去引用里面的参数 用self就好了
    def findPath(self, node: TreeNode, substract: int) -> bool:
        # stopping criteria
        if node.left is None and node.right is None and substract == 0: 
            return True
        if node.right is None and node.left is None and substract != 0: 
            return False 
        # 左
        if node.left:
            substract -= node.left.val
            if self.findPath(node.left, substract) == True:
                return True
            substract += node.left.val
        # 右
        if node.right:
            substract -= node.right.val
            if self.findPath(node.right, substract) == True:
                return True
            substract += node.right.val
        return False 
```
## Day 16

Question:[113 path sum ii](https://leetcode.com/problems/path-sum-ii/)
Outcome with Date: 12-09: X 
First Impression: 尝试了不知道为什么没有任何返回值
Good Video/Blog: 卡尔说的是这题不要返回值https://programmercarl.com/0112.%E8%B7%AF%E5%BE%84%E6%80%BB%E5%92%8C.html#python 但是我和这位录友的想法是一样的 这题是112+257的结合版本 需要返回值 https://zhuanlan.zhihu.com/p/588970571 我感觉我还是没有好好理解reference pointer以及要不要返回值 所以又看了257这道第一次讲回溯的题目 https://www.bilibili.com/video/BV1ZG411G7Dh/?vd_source=8b4794944ae27d265c752edb598636de
Learnt: (1)卡尔想要我们做个比较 112是要返回值 113是不要返回值 你有感受到理由吗 因为112找到一条满足条件的变就可以终止所有的操作 这边找到了 就可以return true那么程序运行就停止了 而113要返回所有满足条件的值;;257 note;; 遍历顺序选择前序的原因：只有前序 才会按照（父节点指向孩子节点）要求的顺序输出
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
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> List[List[int]]:
        self.path_list = []
        if root is None: return []
        path = [root.val]
        self.findPaths(root, path, targetSum-root.val)
        return self.path_list

    def findPaths(self, node, path, substract)-> None:
        if node.left is None and node.right is None and substract == 0:
            self.path_list.append(path[:])
        if node.left:
            path.append(node.left.val)
            self.findPaths(node.left, path, substract-node.left.val)
            path.pop()
        if node.right:
            path.append(node.right.val)
            self.findPaths(node.right, path, substract-node.right.val)
            path.pop()
        
```

Question:[106 construct binary tree from inorder and postorder traversal](https://leetcode.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/)
Outcome with Date: 12-09: X 
First Impression: 没有想法 直接看视频了
Good Video/Blog: https://www.bilibili.com/video/BV1vW4y1i7dn/
Learnt: 通过后序找到中是那个元素 然后对应到中旬即可分割做区间和右区间 用这个对应左右区间 match到后序左右区间位置 然后递归再用后序找到中元素
Difficulty during Implementation:  
Logic of Solution: 
AC Code:
```Python

```

Question:[105 
Outcome with Date:  
First Impression: 
Good Video/Blog: 
Learnt: 
Difficulty during Implementation:  
Logic of Solution: 
AC Code:
```Python

```

Question:[654  
Outcome with Date: 12-07:  
First Impression: 
Good Video/Blog: 
Learnt: 
Difficulty during Implementation:  
Logic of Solution: 
AC Code:
```Python

```

Question:[617 
Outcome with Date: 12-07:  
First Impression: 
Good Video/Blog: 
Learnt: 
Difficulty during Implementation:  
Logic of Solution: 
AC Code:
```Python

```

Question:[700  
Outcome with Date: 12-07:  
First Impression: 
Good Video/Blog: 
Learnt: 
Difficulty during Implementation:  
Logic of Solution: 
AC Code:
```Python

```

Question:[98 
Outcome with Date:  
First Impression: 
Good Video/Blog: 
Learnt: 
Difficulty during Implementation:  
Logic of Solution: 
AC Code:
```Python

```
Question:[530  
Outcome with Date: 12-07:  
First Impression: 
Good Video/Blog: 
Learnt: 
Difficulty during Implementation:  
Logic of Solution: 
AC Code:
```Python

```

Question:[501 
Outcome with Date:  
First Impression: 
Good Video/Blog: 
Learnt: 
Difficulty during Implementation:  
Logic of Solution: 
AC Code:
```Python

```