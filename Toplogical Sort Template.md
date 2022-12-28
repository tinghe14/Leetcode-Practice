### Topological Sort
[207 Course Schedule]()
- Tag: DFS, BFS, Graph, Topological Sort
- Time: 12-26
- Logic of Solution: 
- 一题多解:
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
[Parallel Courses](https://leetcode.com/problems/parallel-courses/description/)
- Tag: Graph, Topological Sort
- Time: 12-26
- Logic of Solution: 
- 一题多解:
```Python
class Solution:
    def minimumSemesters(self, n: int, relations: List[List[int]]) -> int:
        from collections import defaultdict, deque
        in_degree, out_edges = defaultdict(int), defaultdict(list)
        for item in relations:
            in_degree[item[1]] += 1 
            out_edges[item[0]].append(item[1])
        queue = deque()
        for course in range(1, n):
            if in_degree[course] == 0:
                queue.append(course)
        res_order = []
        res = 0
        while queue:
            size = len(queue)
            res += 1
            for i in range(size):
                cur = queue.popleft()
                res_order.append(cur)
                for pointed in out_edges[cur]:
                    in_degree[pointed] -= 1
                    if in_degree[pointed] == 0: #topo的判断
                        queue.append(pointed)
        if len(res_order) < n:
            return -1
        return res
```