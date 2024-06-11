# Tree 
- 二叉树的定义 
```Python
class TreeNode:
    def __init__(self, val):
        self.val = val 
        self.left = None 
        self.right = None 
```
- 递归和迭代： 递归：重复调用自身函数实现循环，迭代：使用计数器结束循环
- 递归三部曲 (1)确定递归函数的参数和返回值，一般二叉树题只需要根节点和一个数组作为参数，数组用来存储我们遍历的结果，一般没有返回值 （2）确定终止条件 （3）确定单层递归逻辑

## DFS
### Pre-oder Traversal/Recursion
- [144 Binary Tree Preorder Traversal](https://leetcode.com/problems/binary-tree-preorder-traversal/description/)
```Python
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

### In-order Recursion

### Post-order Recursion
[145 Binary Tree Postorder Traversal](https://leetcode.com/problems/binary-tree-postorder-traversal/description/)
```Python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def postorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        self.path = []
        if root is None:
            return 
        self.dfs(root)
        return self.path 
    def dfs(self, node):
        if node is None:
            return 
        self.dfs(node.left)
        self.dfs(node.right)
        self.path.append(node.val)
```

### Pre-order Iteration/Post-order Iteration
- 迭代遍历/非递归遍历： 用栈来模拟递归
- 前序： 中左右 后序：左右中 （中右左好实现 reversed一下 就是左右中）
- 中序： 左中右，遍历的顺序（一个个访问）和处理顺序（放在数组中）不一样的 

[Binary Tree Preorder Traversal]
```Python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        stack = []
        result = []
        if root is None: return result 
        stack.append(root)
        while stack:
            cur = stack.pop()
            result.append(cur.val)
            if cur.right: #栈是先进后出 所以要先append右边
                stack.append(cur.right)
            if cur.left:
                stack.append(cur.left)
        return result
```

[Binary Tree Postorder Traversal]
```Python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def postorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        stack = []
        result = []
        if root is None: return result 
        stack.append(root)
        while stack:
            cur = stack.pop()
            result.append(cur.val)
            if cur.left: #栈是先进后出 所以要先append右边
                stack.append(cur.left)
            if cur.right:
                stack.append(cur.right)
        return result[::-1]
```

## BFS
- 按照二叉树本身结构也是无法直接层序遍历，我们需要借助一个队列去报讯每一层的元素
- deque比list的好处是，list pop是o(n)复杂度，deque的popleft是o（1）
- 迭代的下次再写

[102 Binary Tree Level Order Traversal]()
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
        if root is None: return []
        queue = deque()
        queue.append(root)
        paths = []
        while queue:
            size = len(queue)
            path = []
            for i in range(size):
                cur = queue.popleft()
                path.append(cur.val) 
                if cur.left:
                    queue.append(cur.left)
                if cur.right:
                    queue.append(cur.right)
            paths.append(path)
        return paths
```
### BFS VS DFS
- BFS在子节点数比较多的情况，消耗内存十分严重
- DFS难以寻找最优解

# Backtracking 
- 二叉树和 backtracking的区别：二叉树是回溯的一种
- 递归的过程中就有回溯
- 回溯本身不是高效的算法，因为本质就是穷举，穷举出所有可能，如果想要回溯法高效一点，我们可以加一些剪枝的操作
- 虽然不高效，但是如下问题只有回溯这种暴力解法：（1）组合（2）切割（3）子集（4）排列（5）棋盘
- 如何取理解回溯：递归都可以理解成树状结构，回溯解决的都是在集合中递归查找子集，集合的大小就构成了树的宽度，递归的深度就构成了树的深度
- backtracking三部曲：（1）决定回溯参数和返回值（一般没有返回值，回溯法的参数没有二叉树那么简单，一般是看需要记录比较什么参数就加）（2）valid case递归终止条件（if(终止条件){存放结果，return；})（3）回溯的遍历过程 for循环是遍历的集合区间，可以理解成一个节点有几个子孩子for循环就执行几次，backtracking调用子集，实现递归

[39 Combination Sum](https://leetcode.com/problems/combination-sum/description/)
```Python
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        # 要sort一下才好去重
        self.target = target
        self.res = []
        self.candidates = sorted(candidates) #这道题可以不用sort
        self.backtracking(0, [])
        return self.res 

    def backtracking(self, start_ind, path):
        if sum(path) == self.target:
            self.res.append(path[:])
            return #这时就满足条件 可以不用搜 除非有负数
        if sum(path) >= self.target:
            return
        for num in range(start_ind, len(self.candidates)):
            path.append(self.candidates[num])
            self.backtracking(num, path) #因为可以用自己 所以这里是1 #注意经常写错 这里backtrack是从i的下标开始 否则会有重复的结果
            path.pop()
```
[46 Permutations](https://leetcode.com/problems/permutations/description/)
```Python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        self.nums = nums
        self.res = []
        self.backtracking([], [])
        return self.res 
        
    def backtracking(self, cur_path, path):
        if len(path) == len(self.nums):
            self.res.append(path[:])
            return 
        for i in range(len(self.nums)):
            if i in cur_path:
                continue #跳出下一个循环是continue不是break 
            path.append(self.nums[i])
            cur_path.append(i)
            self.backtracking(cur_path, path)
            cur_path.pop()
            path.pop()
```
[Subset](https://leetcode.com/problems/subsets/)
```Python
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        self.nums = nums
        self.res = []
        self.dfs(0, [])
        return self.res

    def dfs(self, cur_ind, path): #记录当前的状态
        self.res.append(path[:]) #没有判断条件 因为每个条件都要放 最开始也是空集 所以空集就在里面 #主要在这
        if cur_ind == len(self.nums):
            return
        for i in range(cur_ind, len(self.nums)):
            path.append(self.nums[i])
            self.dfs(i+1, path)
            path.pop()
```

[Word Search](https://leetcode.com/problems/word-search/)
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
        for dx, dy in [(1,0), (0,1), (-1,0), (0,-1)]: #写好所有方向在判断在不在visited过的 更直接
            newx, newy = dx + curx, dy + cury
            #可以往这一个方向走的条件1.不在visit 2.没有出借 3.满足下一个字母的要求
            if (newx, newy) not in visited and newx >= 0 and newx < self.m and newy >= 0 and newy < self.n \
                and self.board[newx][newy] == self.target[-remain]:
                visited.add((newx, newy))
                self.backtracking(remain-1, newx, newy, visited)
                visited.remove((newx, newy)) #set不用pop用remove因为没有序 一定要有参数 
```
# Divide and Conquer:
- divide and conquer和backtracking的区别：
- 思想上比上面两个都广泛，dfs和回溯都是很套路的题目，做的都是盲目搜索，然后返回想要的东西
- 但是dc每一题的代码都不太一样，有分和治两个步骤，分解成小问题，然后再喝起来
- 想想有一个神奇的函数可以解决我们的问题 大部分的时候 这个神奇的函数就是 直接回答问题的 需要定义这个神奇的函数的参数和输出是什么 之后写出合并的公式conquer 最后想一想base cas
- divide and conquer三部曲（1）base case看哪些小问题不需要继续往下划 （2）参数和输出都是看递推公式需要哪些东西（3）有个分和治的步骤 分就是干super function的步骤 治就是干合起来的步骤 dfs一般不会又返回值也不会想把两个东西aggregate起来 （bfs不是一层层往下的写法 跟分治法不太大变）

# Union Find 
[695 Max Area of Island](https://leetcode.com/problems/max-area-of-island/)
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
# Toplogical Sort

[207 Course Schedule](https://leetcode.com/problems/course-schedule/)
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