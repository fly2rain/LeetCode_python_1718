class TreeNode(object):
  def __init__(self, x):
    self.val = x
    self.left = None
    self.right = None

class ListNode(object):
  def __init__(self, x):
    self.val = x
    self.next = None


class Solution(object):
  def n100_isSameTree(self, p, q):
    if not p and not q: return True
    if not p or not q: return False

    return (
      p.val == q.val and
      self.n100_isSameTree(p.left, q.left) and 
      self.n100_isSameTree(p.right, q.right)
      )



  def n101_isSymmetricTree(self, node):
    if not node: return True
    return self.z101_helper(node.left, node.right)

  def z101_helper(self, p, q):
    if not p and not q: return True
    if not p or not q: return False

    return (
      p.val == q.val and
      self.z101_helper(p.left, q.right) and
      self.z101_helper(p.right, q.left)
      )



  def n101_isSymmetricTree2(self, root):
  	if not root: return True

  	stack = [(root.left, root.right)]
  	while stack:
  		x, y = stack.pop()
  		if not x and not y:
  			continue
  		elif not x or not y:
  			return False
  		elif x.val != y.val:
  			return False
  		stack.append((x.left, y.right))
  		stack.append((x.right, y.left))
  	return True



  def n104_maxDepth(self, node):
    if not node: return 0

    left = self.n104_maxDepth(node.left)
    right = self.n104_maxDepth(node.right)
    return 1 + max(left, right)

class Solution(object):
	def n108_sortedTree2BST_2(self, nums): # 👍👍 save space
		if not nums: return None
		return self.helper(nums, 0, len(nums)-1)

	def helper(nums, l, r):
		if l > r: return None
		m = l + ((r -l) >> 1)
		root = TreeNode(nums[m])
		root.left = self.helper(nums, l, m-1)
		root.right = self.helper(nums, m+1, r)
		return root


  def n108_sortedTree2BST(self, nums):
    if not nums: return None

    k = len(nums) >> 1
    node = TreeNode(nums[k])
    node.left = self.n108_sortedTree2BST(nums[:k])
    node.right = self.n108_sortedTree2BST(nums[k+1:])
    return node


class Solution(object): # 109: converted sorted link list to BST
  def sortedList2BST(self, head):
    if not head: return None
    if not head.next: return TreeNode(head.val)

    slow = head
    fast = fast.next.next
    while fast and fast.next:
      fast = fast.next.next
      slow = slow.next

    mid, slow.next = slow.next, None
    root = TreeNode(mid.val)
    root.left = self.sortedList2BST(head)
    root.right = self.sortedList2BST(mid.next)
    return root



class Solution(object): # 111.easy_Minimum Depth of Binary Tree¶
  def minDepth(self, root):
    if not root: return 0
    if not root.left:
      return 1 + self.minDepth(root.right)
    elif not root.right:
      return 1 + self.minDepth(root.left)
    else:
      return 1 + min(self.minDepth(root.left), self.minDepth(root.right))



class Solution(object): # 110.easy-Balanced Binary Tree
  # solution #1 --------------------------
  def isBalanced(self, root):
    if not root: return True
    depth, isBalanced = self.dfs(root)
    return isBalanced

  def dfs(self, root):
    if not root: return 0, True

    left, isBalancedLeft= self.dfs(root.left)
    right, isBalancedRight = self.dfs(root.right)
    return 1 + max(left, right), (isBalancedLeft  and isBalancedRight and abs(left-right) < 2)

    # solution #2 --------------------------
    def __init__(self):
      self.rst = True

    def isBalanced(self, root):
      if not root: return True
      return self.rst

    def helper(self, root):
      if not root: return 0

      left = self.helper(root.left)
      right = self.helper(root.right)
      self.rst = self.rst and (abs(left-right) < 2)
      return 1 + max(left, right)



class Solution(object): 
# 173.med-Binary Search Tree Iterator
  def next():

  def hasNext():



class Solution(object):
# 144.med-Binary Tree Preorder Traversal
# preorder: root -> left -> right
  def preorder_travesal(self, root):
    if not root: return []

    stack, rst = [root], []
    while stack:
      node = stack.pop()
      rst.append(node.val)

      if node.right: stack.append(node.right)
      if node.left: stack.append(node.left)
    return rst



class Solution(object): 
# 94.med-Binary Tree Inorder Traversal
# inorde: left -> root -> right
  def inOrder_iteration(self, root):
    if not root: return []

    stack, rst = [], []
    node = root
    while node or stack:
      while node:
        stack.append(node)
        node = node.left

      node = stack.pop()
      rst.append(node.val)
      node = node.right
  return rst



class Solution(object):
  # 145.hard-Binary Tree Postorder Traversal
  # postorder: left -> right -> root
  def postOrder_traversal(self, root):
    if not root: return []

    stack, rst = [root], []
    while stack:
      node = stack.pop()
      rst.append(node.val)
      if node.left: stack.append(node.left)
      if node.right: stack.append(node.right)
    return rst[::-1]



from collections import deque
class Solution(object):
# 102.med-Binary Tree Level Order Traversal
  def level_order_travesal(self, root):
    if not root: return []

    q, rst = deque([root]), []
    while q:
      n = len(q)
      tmp = []
      for i in range(n):
        node = q.popleft()
        tmp.append(node.val)
        if node.left: q.append(node.left)
        if node.right: q.append(node.right)
      rst.append(tmp)
    return rst


class Solution(object):
# 107.easy-Binary Tree Level Order Traversal II
# 题意, 包含三层意思:
# level order traversal.
# 先遍历 leaf nodes: from leaf to root.
# 从左到右遍历. level by level from left to right.
  def level_order_travesal3(self, root):
    if not root: return []

    q, rst = deque([root]), []
    while q:
      n, tmp = len(q), []
      for i in range(n):
        node = q.popleft()
        tmp.append(node.val)
        if node.left: q.append(node.left)
        if node.right: q.append(node.right)
      rst.append(tmp)
    return rst[::-1] 

class Solution(object):
# 103.med-Binary Tree Zigzag Level Order Traversal
# 题意: Given a binary tree, return the zigzag level order traversal of its nodes' values.
#  (ie, from left to right, then right to left for the next level and alternate between).
  def zigzag_level_travesal(self, root):
    if not root: return []

    q, rst, k = deque([root]), [], 1
    while q:
      tmp, n = [], len(q)
      for i in range(n):
        node = q.popleft()
        tmp.append(node.val)
        if node.left: tmp.append(node.left)
        if node.right: tmp.append(node.right)
      rst.append(tmp[::k])
      k = -k
    return rst

class Solution(object): # 👍👍👍
# 314.med-Binary Tree Vertical Order Traversal
# 题意: Given a binary tree, return the vertical order traversal of its nodes' values.
# (ie, from top to bottom, column by column).
	def verticalOrder(self, root):
		if not root: return []

		stack = [(root, 0)]
		minIdx = (1<<31) -1
		maxIdx = -(1>>31)
		while stack:
			node, idx = stack.pop()
			minIdx = min(minIdx, idx)
			maxIdx = max(maxIdx, idx)
			rst.append((idx, node.val))

			if node.right: stack.append((node.right, idx+1))
			if node.left: stack.append((node.left, idx-1))

		bucket = [[] for i in range(minIdx
			)]

















