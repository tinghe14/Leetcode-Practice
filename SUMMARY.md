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
