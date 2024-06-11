# 208 Implement Trie (Prefix Tree)
class TrieNode:
    # constructor parameter only has self
    def __init__(self):
        # each trienode will have children 
        self.children = {} # trie node: tire node list 
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

# 14. Longest Common Prefix
class Solution:
    def __init__(self):
        self.root = TrieNode() # root是一个空节点
    
    def insert(self, word):
        cur = self.root # start at root
        for c in word:
            if c not in cur.children:
                cur.children[c] = TrieNode() # insert a new character
                cur.numofchildren += 1
            cur = cur.children[c] # if cur already exist, move to the next node
        cur.endofWord = True 

    def longestCommonPrefix(self, strs: List[str]) -> str:
        for word in strs:
            self.insert(word)
        cur = self.root
        res = ""
        while cur and cur.numofchildren == 1 and cur.endofWord != True:
            for c in cur.children:
                res += c
                cur = cur.children[c]
                break
        return res            
        
class TrieNode:
    def __init__(self):
        self.children = {}
        self.endofWord = False
        self.numofchildren = 0  
    
                