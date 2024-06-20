"""左神讲过的题目都是很重要的题型，值得反复擦了重写，额外加了几道之外刷的题目我感觉可以帮助加深对重要题型的理解"""


# P4.2认识O(nlogn)的排序
## merge sort
"""
递归典型：把问题分解成子问题,一直划分到base case,然后把子问题的解答结合起来来得到母问题的答案
"""
def merge_sort(arr, l, r):
    if r == l+1: #注意区间的定义会影响base case的写法
        return #当是左闭右开时如果base case写成r==l,程序运行时永远不会碰到这个base case导致栈溢出
    mid = l + (r - l) // 2
    merge_sort(arr, l, mid) # 划分成问题本身的子问题
    merge_sort(arr, mid, r)
    merge(arr, l, mid, r) #此处时arr自身的修改写法
def merge(arr, l, mid, r):
    p1, p2 = l, mid
    help = []
    while (p1 < mid) and (p2 < r):
        if arr[p1] <= arr[p2]:
            help.append(arr[p1])
            p1 += 1
        else:
            help.append(arr[p2])
            p2 += 1
    help.extend(arr[p1:mid]) # 注意：list
    help.extend(arr[p2:r])
    arr[l:r] = help[:]
# ex1 = [1, 3, 4, 2, 5]
# merge_sort(ex1, 0, len(ex1))
# print(ex1)

## 求小和问题
"""
关键点：在归并排序的merge那步做小和运算
"""
def merge_sort_add(arr, l, r):
    if r == l+1:
        return 0
    mid = l + (r - l) // 2
    left_sum = merge_sort_add(arr, l, mid)
    right_sum = merge_sort_add(arr, mid, r)
    return left_sum + merge_sum(arr, l, mid, r) + right_sum
def merge_sum(arr, l, mid, r):
    p1, p2 = l, mid
    help = []
    res = 0
    while (p1 < mid) and (p2 < r):
        if arr[p1] < arr[p2]:
            help.append(arr[p1])
            res += arr[p1] * (r-p2)
            p1 += 1
        else:
            help.append(arr[p2])
            p2 += 1
    help.extend(arr[p1:mid]) # 注意：list
    help.extend(arr[p2:r])
    arr[l:r] = help[:]
    return res
# ex1 = [1, 3, 4, 2, 5]
# ans = merge_sort_add(ex1, 0, len(ex1))
# print(ans)

## 求逆序对
def find_inverse_pair(arr):
    if len(arr) == 1:
        return 0
    mid = len(arr) // 2
    left_sum = find_inverse_pair(arr[:mid])
    right_sum = find_inverse_pair(arr[mid:])
    return left_sum + right_sum + compare(arr[:mid], arr[mid:])
def compare(left, right):
    res = 0
    for j, r in enumerate(right):
        for i, l in enumerate(left):
             if left[i] > right[j]:
                 res += 1
    return res
# ex1 = [3, 2, 4, 5, 0]
# ans = find_inverse_pair(ex1)
# print(ans)

## 快排实现 arr[p..r], p, i, j, r
import random
def random_quick_sort(arr, p, r): #这里加上p,r的原因,不用每次直接调用浪费空间的开销; 以及提供更灵活的接口
    if p < r: #这也是为什么选取pivot_ind是用p和r-1而不是0,len(arr)-1
        pivot_ind = random.randint(p, r-1) # 随机性质让他平均情况下成为O(n logn)算法
        # 且random.randint(i, j)是左闭右闭的函数
        arr[pivot_ind], arr[r-1] = arr[r-1], arr[pivot_ind] #要inplace操作,所以要直接对index进行交换,不要写成pivot=arr[pivot_ind]再交换
        i, j = partition(arr, p, r) #i,j 等于区域的左和右节点（左开右闭）,这就是循环不变量,需要在每一个时刻都成立
        random_quick_sort(arr, p, i) #比如最开始时,是空集[i,i)
        random_quick_sort(arr, j, r)
    return arr #在函数内部进行了排序操作,然后返回排序后的数组

def partition(arr, p, r): # 排序,荷兰旗问题的实现
    pivot = arr[r-1]
    i, j  = p, p
    for ind in range(p, r-1): # [p：i) <; [i：j） ==; [j：r-1) >, arr[r-1] pivot
        if arr[ind] < pivot:
            arr[ind], arr[i] = arr[i], arr[ind]
            i += 1
            j += 1
        elif arr[ind] == pivot:
            arr[ind], arr[j] = arr[j], arr[ind]
            j += 1
        else:
            pass
    arr[r-1], arr[j] = arr[j], arr[r-1]
    j += 1
    return i, j
"""
def partition_while_loop(arr, p, r):
    pivot = arr[r-1]
    i, j = p, p
    while j < r-1:
        if arr[j] < pivot:
            arr[i], arr[j] = arr[j], arr[i]
            i += 1
        j += 1
    arr[i], arr[r-1] = arr[r-1], arr[i]
    return i+1, i+2
i 表示小于 pivot 的元素的右边界,而 j 则表示当前遍历的元素.
循环不变式是 arr[p:i] 中的元素都小于 pivot,arr[i:j] 中的元素等于
pivot,而 arr[j:r-1] 中的元素都大于 pivot
"""
# ex1 = [3, 2, 4, 5, 0]
# ans = random_quick_sort(ex1, 0, len(ex1))
# print("ans: ", ans)

# P5.3详解桶排序以及排序内容大总结
import heapq as hq
## 堆操作之heap insert, T O(logn), S O(1)
def swap(arr, ori, after):
    arr[ori], arr[after] = arr[after], arr[ori]
def heap_insert(arr, num):
    # python中叫heappush()
    arr.append(num)
    ind = len(arr) - 1
    parent = (ind-1)//2
    while (parent >= 0) and (arr[ind] > arr[parent]):
        swap(arr, ind, parent)
        ind = parent
        parent = (ind-1)//2
    return arr
# ex1, num = [5, 4, 2], 10
# ans = heap_insert(ex1, num)
# print(ans)

## 堆操作之heapify
def heapify(arr, ind, heap_size):
    # 建立大根堆, T O(logn), S O(1)
    left, right = ind*2+1, ind*2+2
    largest = ind
    while left < heap_size:
        if (right < heap_size) and (arr[right] > arr[left]):
            if arr[right] > arr[ind]:
                largest = right
        else:
            if arr[left] > arr[ind]:
                largest = left
        if largest == ind:
            break
        swap(arr, largest, ind)
        ind = largest #往下移动加检查
        left = ind*2+1
    return arr
# ex1 = [1,7,8,6,5,3,2]
# heap_size = len(ex1)
# ans = heapify(ex1, 0, heap_size)
# print(ans)

## 堆排序的实现,T O(nlogn), S O(1)
def heap_sort(arr):
    heap_size = len(arr)
    # 从后向前直到index为0
    for i in range(heap_size-1, -1, -1):
        heapify(arr, i, heap_size) #heap_size是不会变的因为从最后一个节点到0
    """heapify(arr, 0, heap_size)"""
    for i in range(heap_size-1, 0, -1):
        swap(arr, 0, i) #最值换到最后
        # 随着交换,heap_size的大小就减了一
        heapify(arr, 0, i) #heap_size控制堆的范围
    return arr
# ex1 = [3, 5, 9, 4, 6, 0]
# ans = heap_sort(ex1)
# print(ans)

## 堆排序扩展：几乎有序的数组来排序
def extra_heap_sort(arr: list, k: int) -> list:
    output = []
    hq.heapify(arr[:k+2])
    # 0的位置拍好了序,接下来让1位置也排好...通过加一个弹一个的操作
    for i in range(k+2,len(arr)-k-1):
        output.append(hq.heapify.pop(arr))
        hq.heapify.push(output, arr[i])
        heap_insert(arr[:k+1], arr[k+1+i])
        heapify(arr, i, k+1)
    # for i in range(1, len(arr)-k-1):
    #     heapify(arr,i,k+1)
    # heap_sort(arr[len(arr)-k-1:len(arr)])
    return arr
# ex1 = [2, 4, 3, 5, 1]
# k = 5
# print(extra_heap_sort(ex1, k))

## 桶（基数）排序radix sort的实现
## 希尔排序的实现

# P6.4链表
from typing import List
class SingleListNode:
    def __init__(self, val, next=None):
        self.val = val
        self.next = next
def print_single_list(head: SingleListNode) -> List:
    output = []
    while head:
        output.append(head.val)
        head = head.next
    return output
node1 = SingleListNode(val=3)
node2 = SingleListNode(val=2)
node3 = SingleListNode(val=1)
node1.next = node2
node2.next = node3
lst = print_single_list(node1)

## 反转单向链表TO(N),SO(1)迭代
def reverseList(head):
    prev, curr = None, head
    while curr:
        next_temp = curr.next
        curr.next = prev
        prev = curr
        curr = next_temp
    return prev
# new_head = reverseList(node1)
# def reverseLinkList2(head):
#     if head is None:
#         return head 
#     if head.next is None:
#         return head

#     p = reverseLinkList2(head.next)
# print(f"before {lst}; after {print_single_list(new_head)}")

## 反转单向链表TO(N),SO(N)递归
def reverseList(head):
    if head is None or head.next is None: 
        return head
    p = reverseList(head.next)
    head.next.next = head
    head.next = None
    return p
# new_head = reverseList(node1)
# print(f"before {lst}; after {print_single_list(new_head)}")

# class DoubleLinkList:
#     def __init__(self, val, prev=None, next=None):
#         self.val = val
#         self.prev = prev
#         self.next = next
# node1 = DoubleLinkList(val=3)
# node2 = DoubleLinkList(val=2)
# node3 = DoubleLinkList(val=1)
# node1.next = node2
# node2.prev = node1
# node2.next = node3
# node3.prev = node2
# lst = print_single_list(node1)
# print(lst)

## 反转双向链表TO(N),SO(1)迭代
def reverseDouble(head):
    prev, curr = None, head
    while curr:
        next_temp = curr.next
        curr.next = prev
        curr.prev = next_temp
        prev = curr
        curr = next_temp
    return prev
# new_head = reverseDouble(node1)
# print(print_single_list(new_head))

## 反转双向链表TO(N),SO(N)递归
def reverseDouble(head):
    if head is None or head.next is None:
        return head
    new_head = reverseDouble(head.next)
    head.next.next = head
    head.next = None
    new_head.prev = head
    return new_head
# new_head = reverseDouble(node1)
# print(print_single_list(new_head))

# node1 = SingleListNode(val=1)
# node2 = SingleListNode(val=2)
# node3 = SingleListNode(val=3)
# node1.next = node2
# node2.next = node3
# lst1 = print_single_list(node1)
# node21 = SingleListNode(val=0)
# node22 = SingleListNode(val=1)
# node23 = SingleListNode(val=3)
# node21.next = node22
# node22.next = node23
# lst2 = print_single_list(node21)
# print(lst1, lst2)
## 链表公共区域TO(N),SO(1)自身有序
def common_node(head1, head2):
    output = []
    while head1!= None and head2!=None:
        if head1.val < head2.val:
            head1 = head1.next
        elif head1.val > head2.val:
            head2 = head2.next
        else:
            output.append(head1.val)
            head1 = head1.next
            head2 = head2.next
    return output
# output = common_node(node1, node21)
# print(output)

node1 = SingleListNode(val=1)
node2 = SingleListNode(val=2)
node3 = SingleListNode(val=3)
node0 = SingleListNode(val=0)
node4 = SingleListNode(val=3)
node5 = SingleListNode(val=2)
node6 = SingleListNode(val=1)
node1.next = node2
node2.next = node3
node3.next = node0
node0.next = node4
node4.next = node5
node5.next = node6
lst1 = print_single_list(node1)
## 判断链表是否为回文结构,TO(N), SO(N/2)
def isPalindrome(head):
    slow, fast = head, head
    right_half_stack = []
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    while slow:
        right_half_stack.append(slow.val)
        slow = slow.next
    while right_half_stack:
        if right_half_stack.pop() != head.val:
            return False
        head = head.next
    return True
# print(lst1)
# res = isPalindrome(node1)
# print(res)

## 判断链表是否为回文结构,TO(N), SO(1)，[完美解]
def isPalindrome(head):
    # edge case, 0和1时无论什么数字返回都是true, 2开始就要比值
    if head is None or head.next is None:
        return True
    slow, fast = head, head
    while fast.next and fast.next.next :
        """
        slow停在中点,循环不变量的选择：（中点时,左右的next指针都指向它）
        退出循环时 是不再满足循环不变量的时候
        fast.next is None = 代表奇数时fast走到最后一位(ind5),
                            slow走到正中间(ind2) n=6
        fast.next.next is None = 代表偶数时fast走到倒数第二位(ind3),
                            slow走到中间偏向左边的中间(ind1) n=5
        """
        slow = slow.next # mid of ori
        fast = fast.next.next # end or the end before last end
    prev, second_half_start = None, slow.next
    current = second_half_start
    while current:
        """
        右边逆序,循环不变量的选择：
        slow is None: 因为左右两边要做一一对比,而slow代表的是右边正在
                遍历的值,那么slow必须要遍历到最后一位,下一位就
                不满足条件了
        """
        next_node = current.next
        current.next = prev # 每次循环操作,修改的都是正在遍历的node指向
        prev = current
        current = next_node #最后一次满足循环不变量时next_temp没法指向prev
    second_half_reversed = prev # end of right-part
    first_half_pointer, second_half_pointer = head, second_half_reversed
    res = True
    while second_half_pointer:
        """
        左右一一对比完,循环不变量的选择：
        左边剩下中点没比较,右边已经走完:奇数时,左边顺序的数组比右边逆序的数组多一个
        （2的时候ind0归于左边，ind1归于右边）
        左边右边都遍历完:偶数时,数组长度相等
        """
        if second_half_pointer.val != first_half_pointer.val:
            res = False
            break
        first_half_pointer = first_half_pointer.next
        second_half_pointer = second_half_pointer.next
    # 最后把右边再逆序回来
    prev, current = None, second_half_reversed
    while current:
        next_node = current.next
        current.next = prev
        prev = current
        current = next_node
    slow.next = prev
    return res
# print(lst1)
# res = isPalindrome(node1)
# lst2 = print_single_list(node1)
# print(res)
# print(lst2)


## 链表的荷兰旗问题,左边小中间等右边大, T&S O(n),[完美解]
# 新建一个single node类型的数组
def parition3parts(head, k):
    if head is None: 
        return head
    # dummy heads
    less_head = SingleListNode(0)
    equal_head = SingleListNode(0)
    large_head = SingleListNode(0)
    # pointers
    less, equal, large = less_head, equal_head, large_head
    current = head 
    while current:
        print(current.val) 
        next_node = current.next
        current.next = None # disconnect the current node
        if current.val < k:
            less.next = current 
            less = less.next 
        elif current.val == k:
            equal.next = current
            equal = equal.next 
        else:
            large.next = current 
            large = large.next 
        current = next_node 
    # connect 3 lists together
    large.next = None # Important to avoid cycle in linked list
    less.next = equal_head.next 
    equal.next = large_head.next
    return less_head.next
# print(lst1)
# head2 = parition3parts(node1, k=1)
# lst2 = print_single_list(head2)
# print(lst2)

## 额外：merged two sorted linkedlist (要还是不要 看前面有没有包括)

## 额外：remove nth node from the end of linked list
# 这题的while循环不变量的选择和找mid point始终slow走一步fast走两步这类题型不一样

class SingleListNodeR:
    def __init__(self, val, next=None, random=None):
        self.val = val
        self.next = next
        self.random = random
def print_single_listR(head: SingleListNodeR) -> List[List]:
    output = []
    rand_output = []
    rand_head = head
    while head:
        output.append(head.val)
        rand_head = head.random
        if rand_head is None: 
            rand_output.append(None)
        else:
            rand_output.append(rand_head.val)
        head = head.next
    output.append(None)
    return output, rand_output

node1 = SingleListNodeR(val=1)
node2 = SingleListNodeR(val=2)
node3 = SingleListNodeR(val=3)
node4 = SingleListNodeR(val=0)
node0 = SingleListNodeR(val=3)
node5 = SingleListNodeR(val=2)
node6 = SingleListNodeR(val=1)
node0.next = node4
node0.random = node4
node1.next = node2
node1.random = node3
node2.next = node3
node2.random = node0
node3.next = node0
node3.random = node6
node4.next = node5
node4.random = node2
node5.next = node6
node5.random = None
node6.next = None
node6.random = node5
# 3, 3, 1, 0, 2, None, 2
lst1, lst1R = print_single_listR(node1)
# print("before: ", lst1, lst1R)

## 复制有random pointer的链表，T(n)S(n), LT138, [完美解]
def copyRandomList(head):
    if head is None:
        return head 
    # 1. map old node to a new node with next and random pointers as None 
    old2new = {} 
    current = head
    while current:
        old2new[current] = SingleListNodeR(current.val, None, None)
        current = current.next 
    # 2. assign next pointer using the map, same to the random pointers
    #new_head = old2new[head] # attention!
    #new_head_reserved = new_head
    current = head
    while current:
        if current.next:
            old2new[current].next = old2new[current.next]
        if current.random:
            old2new[current].random = old2new[current.random]
        current = current.next
    return old2new[head]
# output_head = copyRandomList(node1)
# lst2, lst2R = print_single_listR(output_head)
# print("after: ", lst2, lst2R)

## 复制有random pointer的链表，T(n)S(1), LT138, [完美解]
def copyRandomList(head):
    if head is None:
        return head
    # 1. add new node behind the old node, assign the next pointer of new node to the next old node
    current = head 
    while current: 
        next_node = current.next 
        current.next = SingleListNodeR(current.val, None, None)
        current.next.next = next_node 
        current = next_node
    # 2. second pass: assign random pointer
    current = head 
    while current: # 每一个node都要复制
        if current.random: #因为要复制random.next所以他要先判断是否存在
            current.next.random = current.random.next
        current = current.next.next
    # 3. thrid pass: assign next pointer
    current = head 
    new_current = head.next # 要形成两个链表，所以需要这俩个头
    new_node = new_current
    while current:
        if current.next:
            current.next = current.next.next 
        if new_current.next:
            new_current.next = new_current.next.next 
        current = current.next # 先改变了结构，所以下一个操作只移动一步
        new_current = new_current.next 
    return new_node
# output_head = copyRandomList(node1)
# lst2, lst2R = print_single_listR(output_head)
# print("after: ", lst2, lst2R)

## 复制有random pointer的链表，T(n)S(n), LT138, 递归法

# P7.5二叉树
from typing import Optional

class TreeNode:
    def __init__(self, val, left=None, right=None):
        self.val = val 
        self.left = left 
        self.right = right 
def printTree(head: Optional[TreeNode]) -> List[int]:
    res = []
    def traverse(node: Optional[TreeNode]):
        if node is None: 
            return
        res.append(node.val)
        traverse(node.left)
        traverse(node.right)
    traverse(head)
    return res
tree_node4 = TreeNode(4)
tree_node5 = TreeNode(5)
tree_node6 = TreeNode(6)
tree_node7 = TreeNode(7)
tree_node2 = TreeNode(2, tree_node4, tree_node5)
tree_node3 = TreeNode(3, tree_node6, tree_node7)
tree_node1 = TreeNode(1, tree_node2, tree_node3)

print("before: ", printTree(tree_node1))

## 先序遍历，recursion
def preOrderRec(tree_node):
    if tree_node is None:  
        return tree_node
    print(tree_node.val)
    preOrder(tree_node.left)
    preOrder(tree_node.right)
    return tree_node
# print(preOrderRec(tree_node1))
# iteration
def preOrderUnRec(tree_node):
    if tree_node is None:
        return 
    stack = [tree_node] #初始化栈，先将根节点压入栈中
    while stack is not None:
        current = stack.pop() # 弹出栈顶节点
        print(current.val) # 访问当前节点
        # 注意：先将右子节点压入栈中，后将左子节点压入栈中
        # 这样在下一个循环中，左子节点会先被处理
        if current.right is not None:
            stack.append(current.right)
        if current.left is not None:
            stack.append(current.left)
# print(preOrderUnRec(tree_node1))

## 后序遍历,iteration (有更节省space的方法)
def postOrder(tree_node): # true再次访问的时候
    if tree_node is None:# 第一次遇到节点的时候，将其标记为true然后重新压入栈，然后依次
        return []#将右子节点，和左子节点压入栈。这保证了在再次访问该节点时，其子节点已经被处理
    stack = [tree_node] #可以用一个栈实现
    collect = []
    res = []
    while stack:
        current = stack.pop()
        collect.append(current)
        if current.left:
            stack.append(current.left)
        if current.right:
            stack.append(current.right)
    for i in range(len(collect)-1, -1, -1):
        res.append(collect[i].val)
    return res 
# print(postOrder(tree_node1))
# Space better, 第一次到append进去(node, false)，第二次回到pop出来改成(node, true)，第三次回到true时pop
def postOrderB(tree_node):
    if tree_node is None:
        return []
    res = []
    visited = False
    stack = [(tree_node, visited)] # or here False, when pop, assign as visited
    while stack: # whether stack is empty, not whether is None X while stack is not None:
        current, visited = stack.pop()
        if visited == True:
            res.append(current.val)
        else:
            stack.append((current, True))
            if current.right:
                stack.append((current.right, False))
            if current.left:
                stack.append((current.left, False))

    return res 
# print(postOrderB(tree_node1))

##中序遍历，非递归
def inOrder(treeNode):
    curr = treeNode
    stack = []
    res = []
    while curr is not None or stack:
        # reach the leftmost treenode
        while curr is not None:
            stack.append(curr)
            curr = curr.left
        curr = stack.pop()
        res.append(curr.val)
        # visit right tree
        curr = curr.right
    return res
print(inOrder(tree_node1))


