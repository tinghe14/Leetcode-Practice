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