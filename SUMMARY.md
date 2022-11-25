# Table of contents

## 随想录day1
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
## 随想录day2
Question: [34 Find First and Last Position of Element in Sorted Array](https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/)  
Outcome with Date: 11-24:X  
First Impression:know that I can apply binary search twice by finding the target-1, and target+1 -> can't work, also the same thing will happen in target-1 and target+1->wrong  
Good Video/Blog:https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/solutions/1136731/find-first-and-last-position-of-element-in-sorted-array/  
Learnt:
Difficulty during Implementation:  
Logic of Solution: (need help)  
AC Code:  

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

## 随想录day2
Question: [977 Squares of a Sorted Array](https://leetcode.com/problems/squares-of-a-sorted-array/)
Outcome with Date: MM-DD:X|Y|O
First Impression:
Good Video/Blog:
Learnt:
Difficulty during Implementation:
Logic of Solution:
AC Code:

Question: [209 Minimum Size Subarray Sum](https://leetcode.com/problems/minimum-size-subarray-sum/)
Outcome with Date: MM-DD:X|Y|O
First Impression:
Good Video/Blog:
Learnt:
Difficulty during Implementation:
Logic of Solution:
AC Code:

Question: [59 Spiral Matrix II](https://leetcode.com/problems/spiral-matrix-ii/)
Outcome with Date: MM-DD:X|Y|O
First Impression:
Good Video/Blog:
Learnt:
Difficulty during Implementation:
Logic of Solution:
AC Code:

Question: [203
Outcome with Date: MM-DD:X|Y|O
First Impression:
Good Video/Blog:
Learnt:
Difficulty during Implementation:
Logic of Solution:
AC Code:

Question: [707
Outcome with Date: MM-DD:X|Y|O
First Impression:
Good Video/Blog:
Learnt:
Difficulty during Implementation:
Logic of Solution:
AC Code:

Question: [206
Outcome with Date: MM-DD:X|Y|O
First Impression:
Good Video/Blog:
Learnt:
Difficulty during Implementation:
Logic of Solution:
AC Code:

Question: [24
Outcome with Date: MM-DD:X|Y|O
First Impression:
Good Video/Blog:
Learnt:
Difficulty during Implementation:
Logic of Solution:
AC Code:

Question: [19
Outcome with Date: MM-DD:X|Y|O
First Impression:
Good Video/Blog:
Learnt:
Difficulty during Implementation:
Logic of Solution:
AC Code:

Question: [142
Outcome with Date: MM-DD:X|Y|O
First Impression:
Good Video/Blog:
Learnt:
Difficulty during Implementation:
Logic of Solution:
AC Code:

Question: [242
Outcome with Date: MM-DD:X|Y|O
First Impression:
Good Video/Blog:
Learnt:
Difficulty during Implementation:
Logic of Solution:
AC Code:

Question: [349
Outcome with Date: MM-DD:X|Y|O
First Impression:
Good Video/Blog:
Learnt:
Difficulty during Implementation:
Logic of Solution:
AC Code:

Question: [202
Outcome with Date: MM-DD:X|Y|O
First Impression:
Good Video/Blog:
Learnt:
Difficulty during Implementation:
Logic of Solution:
AC Code:

Question: [1
Outcome with Date: MM-DD:X|Y|O
First Impression:
Good Video/Blog:
Learnt:
Difficulty during Implementation:
Logic of Solution:
AC Code:
