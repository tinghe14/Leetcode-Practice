# NEETCODE 150

contains duplicate: 
1. 12.10 Y 只有一个小错误 忘记了defaultdict的使用方法 并不是defauldict(0)而是defauldict(int)

## Arrays & Hashing

## Two Pointers

## Sliding Window

## Stack 

## Binary Search

[74 Search a 2D Matrix](https://leetcode.com/problems/search-a-2d-matrix/)
Log: 
FE:  
Video:
Learnt: 
Difficuly whem Implement:
Logic of Code:
AC Code:


## Linked List

## Trees

### Day 17
## Tries
1. it is a tree data structure (but not a binary) used for storing and locating keys from a sets. the kyes are usally
strings that are stored character by character-each node of a trie corresponds to a single character
rather than the entire key; the order of the characters in a string is represented by edges between 
the adjacent nodes.
2. This way, additional space is not required for strong strings with commonm prefiexes. 
We can keep moving down the tree until a new character, that is not present in the node's children, 
is encounrered and add it as a new node. Similarly, searches can also be performed using depth first 
search by following the edges between the nodes. Essentially, in a trie, words with the same prefix 
or stem share the memory area that corresponds to that prefix.
3. To understand how tries are more efficient for storing and searching string, consider a binary tree.
The time complexity of a binary tree is O(logn), where we talk in terms of log base 2. Instead, think of 
a quaternary tree, where every node has a fan-out of four, so each node can have four children. The time 
complexity of this tree is still o(logn). However, now we're talking in terms of log with base 4. That's 
an improvement in the performance even if it's by a contant factor. As our trees become wider and shorter, 
the operations become more efficient. This is because we don't have to traverse as deep.
4. This is exactly the motivation behind a trie. What if we had a n-ary tree with the fan-out equal to the 
number of unique values in the given dataset? For example, if we're considering strings in English, the fan-
out would be 26, corresponding to the number of letters in English language. This makes tree wider and shorter.
The maximum depth of the tire would be maximum length of a word or string
5. when my problem match this pattern? Yes, if either of these conditions is fulfilled(1) we need to compare two strings to detect partial matches, based on the initial 
characters of one or both string. (2) we wish to optimize the space used to store a dictionary of words. strong shared prefixes once allows for significant saving. No, if
either of these conidtions is fulfilled: (1) the problem statement restricts us from 
breaking down the strings into individual characters. (2) partial matches between pairs of strings are not significant to solving the problem
6. real-world example: (1) autocomplete system: trie prompts the search engine to give us some suggestions to complete our query when we start typing something in the 
search bar. These suggestions are given based on common queries that users have searched already that match the prefix we have typed (2) orhographic corrector: pop-up
suggestions or red-lines under a word while you're typing a message. This is orhographic corrector making suggestions and pointing out spelling mistakes by searching through a dictionary. It uses a trie data structure for efficient searches
and retrievals from the available database

[208 Implement Trie(Prefix Tree)](https://leetcode.com/problems/implement-trie-prefix-tree/)
Log:12/10 F
FE: 
Video: https://www.youtube.com/watch?v=oobqoCJlHA0
Learnt: (1) instead of hash map, it can do very efficient to find a word start with at o(1) (2) to implement a trie we can't do it with a a class of trie node by creating an constructor(init) which contains memeber variables （3）constructor里面参数只用写 self
Difficuly whem Implement:(1)trie node哪里要有什么memeber variables我很疑惑
AC Code:
```Python
class TrieNode:
    # constructor parameter only has self
    def __init__(self):
        # each trienode will have children 
        self.children = {}
        self.end_of_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()
 
    def insert(self, word: str) -> None:
        cur = self.root 
        for c in word:
            if c not in cur.children:
                # 如果不在的话 需要付什么值
                cur.children[c] = TrieNode()
            cur = cur.children[c]
        cur.end_of_word = True

    def search(self, word: str) -> bool:
        cur = self.root
        for c in word:
            if c not in cur.children:
                return False
            cur = cur.children[c]
        return cur.end_of_word   

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
[211 Design Add and Search Words Data Structure](https://leetcode.com/problems/design-add-and-search-words-data-structure/)
Log: 12/10: x
FE: 没有意识到.应该怎么办 (need help!!!dfs那里不知道怎么写 输入是什么)
Video:
Learnt:
Difficuly whem Implement:
AC Code:

[79 Word Search](https://leetcode.com/problems/word-search/)
Log: 12/10: x
FE:  
Video:
Learnt: 
Difficuly whem Implement:
AC Code:

[212 Word Search II](https://leetcode.com/problems/word-search-ii/)
Log: 12/10: x
FE: 第一次遇到二维数组的题目就直接看视频了 
Video:https://www.youtube.com/watch?v=asbcE9mZz_U
Learnt: 推荐先解决79
Difficuly whem Implement:
AC Code:


## Heap/Priority Queue
### Heap
1. heap are advanced data structures that are useful in applications such as sorting and implementing priority queues. They are regular binary trees with two special properties (1.1) heaps must ne complete binary trees (complete binary tree: each node has at most two children and the nodes at all levels are full, except for the leaf nodes which can be empty but are filled from left to right)
2. min-heap: all the parent node keys are less than or equal to their child node keys. so the root node, in this case, will always contain the smallest element present in the heap 
3. max-heap:all the parent node keys must be greater than or equal to their child node keys. so the root node will always contain the largest element in the heap. 
4. where are heaps used? the primary purpose of heaps is to return the smallest or largest element. This is because the time complexity of getting the minimum/maximum value from a min/max heap is o(1). This way, algorithms that requires retrieving the maximum/minimum value can be optimized. Heaps are also used to design priority queues. Some famous algorithms which are implemented using heaps are prim's algorithm, dijkstra'algorithm and heap sort algorithm.
5. 
### Top K elements
1. it helps find some specific k number of elements from the given data with optimum
time complexity. many questions ask us to find the top, the smallest, or the most/least frequent k elements in an unsorted list of elements. Brue Furce: sorting the list takes o(nlogn) time, then finding the k elements takes o(k) times. However, the top k element pattern can allow us to solve the problen using o(n log k) time without sorting the list first
2. the best data structure to keep track the smallest or largest k elements is heap. For this question, we can either use a max-heap to find the k smallest element or a min-heap to find the largest k elements,.
3. steps: (1)insert the first k elements from the given set of elements to the min-heap or max-heap (2) iteratre through the rest of the element (2.1) for min-heap, if you find the larger element, remove the top (smallest number) of the min-heap and insert the new larger element (2.2) for max-heap, if you find the smaller element, remove the top (largest number) of the max-heap and insert the new smallest element. -> Iterating the complete list takes o(n) time, and the heap takes o(log k) time for inseration. And we get the o(1) access to the k elements usingt the heap
4. examples of questions (1) sort characters by frequency: caculate the frequency and 
store them in the hash map. Then use the frequency to maintain the max-heap and sort character by frequency. (2) connect n ropes with minimum cost: first of all, insert all n ropes into the min-heap. Then, extract the minimum and second minimum from the min-heap. And the extracted values and insert the calculated value back to the min-heap. Maintain the total cost variable to keep track of the minimum cost while calculating the cost
5. does my problem match this pattern? (1) yes, if both of these condfitions are fulfilled: (1.1) we need to find the largest, smallest, most frequent, or least frequent subset of elements in an unsorted lsit. (1.2) This may be the requirement of the final solution, or it may be necessary as an intermediate step toward the final solution (2) no, if any of these conditions is fulfilled (2.1) the input data structure doesn't support random access (2.2) the input data is already sorted according to the criteria relevant to solving the problem (2.3)if only 1 extreme value is required, that is, k=1, as that problem can be solved in o(n) with a simple scan through the input array
6. real-world problems (1) uber: select at least the n nearest drivers within the user's vicnity, avoiding the drivers that are too far away (2) stocks: given the set of IDs of brokers, determine the top k broker's performance with the frquently repeated IDs in the given dataset

[703 Kth Largest Element in a Stream](https://leetcode.com/problems/kth-largest-element-in-a-stream/)
Log: 12/10: X
FE:  
Video: https://www.youtube.com/watch?v=hOjcdrqMoQ8
Learnt: (1)min heap of size k (1.1) add and pop element from min-heap 保持原来的心智at o(logn) 因为是complete binary treetime ->因为要一直把他pop上去(1.2) get the minimum value at o(1) time
Difficuly whem Implement:
Logic of Code: (1)list convert to min heap (2) narrow them down into k-min heap (contains 1-kth maximum)(3) every time add a value, pop the minimum and matina the k-min heap (4) return the element we need ->小唐说这就是必须要min-heap的 都没有所有的data我们怎么heapify呢
AC Code:
```Python
import heapq
class KthLargest:
    def __init__(self, k: int, nums: List[int]):
        self.minHeap, self.k = nums, k
        # maintain the heapq from List
        # 存成min heap 
        heapq.heapify(self.minHeap) #o(n)
        # maintain the k-th min-heap
        # 变化成kth min heap
        while len(self.minHeap) > self.k:
            heapq.heappop(self.minHeap)

    def add(self, val: int) -> int: # return kth largest
        # 更新保存成k-th min heap
        # push a heapq的语法
        heapq.heappush(self.minHeap, val)
        if len(self.minHeap) > self.k:
            heapq.heappop(self.minHeap) # get minmun from k-th heap equals to return kth largest at o(1)
        return self.minHeap[0]

# Your KthLargest object will be instantiated and called as such:
# obj = KthLargest(k, nums)
# param_1 = obj.add(val)
```
[1046 Last Stone Weight](https://leetcode.com/problems/last-stone-weight/)
Log: 12/10: X
FE: 对于heap的用法还是要熟练下（1）heapq.heapfiy(lst)没有返回值，是修改本身，（2）heapy.nlargest(k, lst)只是输出k大的数 没有pop里面的k个数也没有 要维持max heap还是要用 -1
Video: https://www.youtube.com/watch?v=B-QCq79-Vfw
Learnt: 
Difficuly whem Implement:（1）请注意一个概念一个mmutable object, apply with method是没有返回值的 [].append(0),‘hello’.upper()是有返回值的 所以！heapy.heapify(lst)是没有返回值的 会直接在lst上修改 （2）很多edge case 靠debug才能发现 对题目理解不够深刻
Logic of code:
AC Code:
```Python
import heapq

class Solution:
    def lastStoneWeight(self, stones: List[int]) -> int:
        size = len(stones)
        if size == 1:
            return stones[0]
        #heapq.heapify(-1*stones)会报错的
        #这不是numpy不能向量化操作
        stones = [-1*i for i in stones]
        heapq.heapify(stones)
        while size >= 2:
            first = heapq.heappop(stones)
            second = heapq.heappop(stones)
            # 忘记把东西pop出去了
            new_stone = -1*first - -1*second
            if new_stone > 0:
                heapq.heappush(stones, -1*new_stone)
                size -= 1
            if new_stone == 0:
                size -= 2
        if len(stones) == 0:
            return 0
        # 还要做判断 如果里面一个数也没有了 因为全部都抵消了 
        result = heapq.heappop(stones)
        return -1*result
```
[973 K closest points to origin](https://leetcode.com/problems/k-closest-points-to-origin/description/)
Log: 12/11: x
FE:  不懂为什么我的直接min-heap直接转换不行 大家都要用maxheap而且是一个个加->明白了 因为这样是对所有的数都进行了维护 是不表要的 维护长度为k的max-heap就行->不是的看了347官方解法 这不是对所有数进行维护 ->我之前用的是nlargest确实不用对所有数进行维护 但是我这里有重复的数据 不能用dict导致直接用heappush to min heap确实是和sort的效率一样就没有必要 以下链接有更优解法 还没有看 可以到NLOGK甚至官方解答有N的
Video: https://leetcode.com/problems/k-closest-points-to-origin/solutions/348171/python3-sort-o-nlogn-minimum-heap-o-nlogn-and-maximum-heap-o-nlogk/ https://stackoverflow.com/questions/3954530/how-to-make-heapq-evaluate-the-heap-off-of-a-specific-attribute
Learnt: (1)heapq.heapify(lst)里面只能放list (1)heappush by tuple, it will sort by first element
Difficuly whem Implement: (1)会有相同的坐标 输出的时候 也需要输出所有的值-》不能用 dict要s
Logic of Code:
AC Code:
```Python
import heapq

class Solution:
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        # caculate the distance to the origin
        # set key: coordianate->tuple; value: distance 
        # return kth items from min heap
        # 新建一个min-heap用的就是list
        min_heap, dist = [], {}
        # 哈希表的key不能是mutable的-》不能用哈希表有重复数据
        for coors in points:
            #when you heappush by a tuple, it will sort by first element
            heapq.heappush(min_heap, (coors[0]**2+coors[1]**2, coors))
        return[heapq.heappop(min_heap)[1] for i in range(k)]
```
[215](https://leetcode.com/problems/kth-largest-element-in-an-array/)
Log: 12/11: X
FE:  maintain一个kth的max heap不就行了 但是答案有错-》写法不对不是只push k个 是每一个都要push 当长度大于k 再pop 自己发现了 然后就改成整了 但官方答案有最优解 第二次刷的时候要看一下
Video:
Learnt: 
Difficuly whem Implement:
Logic of Code:
AC Code:
```Python
import heapq

class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        min_heap = []
        size = len(min_heap)
        for num in nums:
            heapq.heappush(min_heap, -1*num)
            while size:
                heapq.heappop(min_heap)
        return [heapq.heappop(min_heap) for i in range(k)][-1]*(-1)
```

## Backtracking

## Graphs

## Advanced Graphs 

## 1-D Dynamic Programming 

## 2-D Dynamic Programming 

## Greedy 

## Intervals

## Math & Geometry

## Bit Manipulation

[]()
Log: 
FE:  
Video:
Learnt: 
Difficuly whem Implement:
Logic of Code:
AC Code:


class Solution:
    def deleteGreatestValue(self, grid: List[List[int]]) -> int:
        # convert row from max into min for all rows 
        # find max on the columns 
        # sum the max on the columns up
        numRow = len(grid)
        numCol = len(grid[0])
        sortMat = [[0]*numCol for i in range(numRow)]
        maxCol = []
        for i in range(numRow):
            sortMat[i] = sorted(grid[i], reverse=True)
        # t：n*mlogm
        for j in range(numCol):
            maxCol.append(max(sortMat[x][j] for x in range(numRow)))
        return sum(maxCol)
        # t: n*m
        # total: n*mlogm
            
    # numcol
    # [i,:]是numpy语法
    # [i][:]->[i]
    # sorted()才有返回值,reverse才从大到小
    # 第 i行grid[i]
    # 第j行 sortMat[x][j] for x in range(numRow)
    # log m趋于正无穷 所以nlogn 比n慢

    class Solution:
    def longestSquareStreak(self, nums: List[int]) -> int:
        hashmap = set(nums) #n
        tempmax = 1
        for num in hashmap:
            times = 1
            while num**2 in hashmap: #n log log k
                times += 1
                num = num**2
            tempmax = max(tempmax, times)
        if tempmax != 1:
            return tempmax
        else:
            return -1
        
    # Time: n log log k -> logn趋紧正无穷 是大于1

    # 解读题目 不要盲目follow步骤 这个问题你用什么可以解
    # 什么类型的题 可以用什么以前的模版来解

    # 常见的写法 需要有自己的套路