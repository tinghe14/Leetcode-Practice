# 2.基础数据结构（上）: 链表，堆栈，队列
homepage: https://github.com/datawhalechina/leetcode-notes/blob/main/docs/ch02/index.md

## 2.1 链表（第1～4天）
### 链表基础知识
- linked list: 一种线性表数据结构。它使用一组任意的存储单元（随机分配的物理地址：可以连续或者不连续的一串内存地址），来存储一组具有相同类型的数据。每个链节点不仅要存放一个数据元素的值，还要存放一个后继指针next(指出这个数据元素在逻辑关系上的直接后继元素所在链表节点的地址)
- 优点：不需要提前分配存储空间，在需要的时候可以临时申请，不会造成空间的浪费；插入，移动，删除元素的时间效率比数组高
- 缺点：因为要存储数据本身信息还要存储next指针，因此链表结构比数组结构的空间开销大
- 除了单链表以外，还有duobly linked list, circular link list
  - doubly linked list: 每个链节点有指向前驱和后继的两个指针
  - circular linked list: 它的最后一个链节点指向头节点，形成一个环
- 链表的基本操作（单链表为例）
  - 结构定义：链表是由链节点通过next链接而构成的。我们可以先定义一个简单的listNode，再来定义完整的linkedList类
    - listNode：使用成员变量val表示数据元素的值，使用指针变量next表示后继指针
    - linkedNode:使用一个链节点变量head来表示链表的头节点
    ~~~
    # listNode
    class ListNode:
      def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
    # linkedList
    class LinkedList:
      def __init__(self):
        self.head = None
    ~~~
  1. 创建空链表，只需要把相应的链表头节点变量设置为空链接即可，python中用none
  2. 建立线性链表 (O(n),n为线性表长度) 根据线性表的数据元素动态生成链节点，并依次将其连接到链表中
  ~~~
  def create(self, data):
    self.head = ListNode(0)
    cur = self.head
    for i in range(len(data)):
      node = ListNode(data[i])
      cur.next = node
      cur = cur.next
  ~~~
  3. 求线性链表的长度 (O(n),n为线性表长度)
  ~~~
  def length(self):
    count = 0
    cur = self.head
    while cur:
      count += 1
      cur = cur.next
    return count
  ~~~
  4. 查找元素 (O(n),n为线性表长度) 从头节点head开始，沿着链表接电逐一进行查找
  ~~~
  def find(self, val):
    cur = self.head
    while cur:
      if val == cur.val:
        return cur
      cur = cur.next
    return None
  ~~~
  5. 插入元素：头部(与链表的长度无关，O(1))，尾部(要将cur从链表头部移到尾部, O(n))，中间(将cur从链表头部移动到第i个链节点之前，操作的平均时间复杂度是O(n)因此,链表中间插入元素的时间复杂度是O(n)
  ~~~
  # 在链表第1个链节点之前插入值为val的链节点
  # 1.建立值为val的链节点node
  # 2.将node的next指针指向链表的头节点head
  # 3.将链表的头节点head指向node
  def inserFront(self, val):
    node = ListNode(val)
    node.next = self.head
    self.head = node

  # 在链表最后1个链节点之后插入值为val的链节点
  #1.建立值为val的链节点node
  #2.使用指针cur指向链表的头节点head
  #3.通过链接点的next指针移动cur指针，从而遍历链表，直到cur.next为None
  #4.令cur.next指向新链节点node
  def insertRear(self, val):
    node = ListNode(val)
    cur = self.head
    while cur.next:
      cur = cur.next
    cur.next = node

  #在链表第i个链节点之前插入值为val的链接点
  #1.使用指针变量cur和一个计数器count。令cur指向链表的头节点,count初始值赋值为0
  #2.沿着链节点的next指针遍历链表，指针变量cur每指向一个链节点，计数器就做一次计数
  #3.当遍历到第index-1个链节点时停止遍历
  #4.创建一个值为val的链节点node
  #5.将node.next指向cur.next
  #6.将cur.next指向node
  def insertInside(self, index, val):
    count = 0
    cur = self.head
    while cur and count < index -1:
      count += 1
      cur = cur.next
    if not cur:
      return 'Error'
    node = ListNode(val)
    node.next = cur.next
    cur.next = node
  ~~~
  6.改变元素o(n)将链表中第i个元素改为val,和插入的code对比，判断的是while cur and count < index然后直接更改cur的值
### 练习题目01
### 练习题目02
### 链表基础题目
### 链表排序03
### 练习题目03
### 链表排序题目
### 链表双指针04
### 练习题目04
### 链表双指针题目

