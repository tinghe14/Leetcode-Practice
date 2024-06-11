# 347. top k frequent elements
# https://leetcode.com/problems/top-k-frequent-elements/description/
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        # k min-heap, T: O(n log k), S:O(n + k): n for counter, k for final heap
        import heapq as pq #priority heap
        from collections import Counter
        counter = Counter(nums) #O(n)
        heap = [] # heapq的initialize就是个list啊！！
        # 1. to add first k elements in to k-min heap: k log k 
        # 2. pop every item after n-k: n-k * log k (logk: time complexity for heap pop/push )
        # 3. conver the final min-heap into array: k log k 
        for num, c in counter.items(): #得到counter的pair需要用items
            pq.heappush(heap, (c, num)) #定义一个小顶堆 大小为k hq.heappush(pri_que, (freq, val))
            if len(heap) > k:
                pq.heappop(heap) #priority heap pop也是要有参数的 heap 本身
        res = [0]*k
        for i in range(k-1, -1, -1):
            res[i] = pq.heappop(heap)[1] #先pop最小的
        return res 