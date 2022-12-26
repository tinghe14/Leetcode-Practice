12.09.2022
唐博浩真传

"""return an array of all leaves (in any order)
   All functions did not consider edge conditions
"""

class Solution:
    """void, only difference in template is from the definition of the valid case(if this is answer, append; if not, do nothing)"""
    #dfs search本身是个通用解法 就是遍历所有可能的路径 让他通用就是没有返回值 根据题目要求 在他遍历的时候 就可以把满足条件的append到solution里 所以此处叫valid case
    #因为append到solution里面 solution会在外面定义 我们这里用self来帮助 就是一个全局被影响的变量
    #此时不用修改__init__因为我们先跑的是主函数 只要主函数里有初始化 self.solutions=[] 他们就会自动在__init__里面初始化
    #正常__init__写法是因为我们要先init这个class 比如solution() 所以他会先跑__init__ 
    def dfs(self, node):
        if node is leaf: # different
            self.solutions.append(node) 
        # 12.09疑惑：遍历顺序怎么体现？->没有 切记 没有返回值的正常dfs是在遍历所有路径他的if在做的是当遍历的时候满足条件 就要把结果加进solution中
        # 19-20行是固定的->bfs!!的基本search模版不用考虑中间的节点 就是找其他方向！
        dfs(node.left)
        dfs(node.right)
        ## more backtracking sense
        node = node.left
        dfs(node)
        node = node.parent
        node = node.right
        dfs(node)
        node = node.parent
    # 核心（1）如果你知道子问题的答案怎么 aggregate 到母问题（2）你要返回啥（base case,子问题都该回答原问题问的东西）
    # 没有返回值就很容易写 因为只要改动valid case 
    # 有返回值要考虑的就多些 因为要让他们都return同一个东西 然后在agggreate起来 
    # 但是大部分的二叉树都是第二种情况
    def rec(self, node) -> List[Node]:
        """return value, differences a lot: base case(since need to aggregte), what we do in base case, return what from base case, what is recursion funciton(how to agg)"""
        if node is leaf:    # different
            return [node]   # different-》base case也要return list因为要是同一个类型的
        # leaves = left tree leaves + right tree leaves 递归函数可以是不一样的 但是都可以理解冲左子树和右子树的东西向上传
        left = rec(node.left)
        right = rec(node.right)
        agg = left + right # different
        return agg
    
    def main(self, root):
        self.solutions = []


"""
    If the question is asking to find all solutions / whether this is a solution, DFS is typical (find whether there is a subset satisfying conditions like "sum to zero")
    Otherwise, may think about general recursive solution (flatten a binary tree to a linked list in specific order)
"""

# cur_state = [...] past path visited (most general case, remember all path, in some simple case, only need to remeber some summary: sums)

def dfs(self, cur_state):
    if valid(cur_state): # different
        self.solutions.append(copy(cur_state)) # mutable need to deep copy
    for direction in next_directions: # need to self determin what are next directions, for example in binary tree, it should be like left right, if maze, next possible directions
        cur_state.append(direction)
        dfs(cur) 
        cur_stat.pop()

[39 Combination Sum](https://leetcode.com/problems/combination-sum/description/)
- Tag: Array, Backtracking
- Time: 12-26
- Logic of Solution: given an array of distinct integers, return a list of all unique combinations of cancdidates where the chosen numbers sum to target. The same number may be chosen from cnadidates an unlimited number of times. Two combinations are unique if the frequency of at least one of the chosen numbers is different
- 一题多解:
```Python
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        # bracktracking 
        # 1.要inistiate的结果的list 
        # 2.搜索的起点是什么
        # 常见trick:把输入的东西sort一下 更好去重
        self.sums = []
        self.candidates = sorted(candidates) #这个才有返回值
        self.target = target
        self.backtracking([], 0)
        return self.sums
        #回溯要有两个参数：一个时当前的path,一个ind代表之后能走的方向（去重）
    def backtracking(self, cur, ind):
        # 1.解是否满足条件，满足时，记住
        if sum(cur) == self.target: 
            self.sums.append(cur[:])
        # 2.（optional）是否可以提前结束
        if sum(cur) >= self.target:
            return 
        # 3.搜索所有的可能方向
        for i in range(ind, len(self.candidates)):
            cur.append(self.candidates[i])
            self.backtracking(cur, i) # 这里是i因为可以重复自己
            cur.pop()
```