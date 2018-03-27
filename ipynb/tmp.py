# Definition for a undirected graph node
# class UndirectedGraphNode:
#     def __init__(self, x):
#         self.label = x
#         self.neighbors = []

class Solution:
    # @param node, a undirected graph node
    # @return a undirected graph node
    def cloneGraph(self, node):
        if not node: return node
        
        root = UndirectedGraphNode(node.label)
        stack = [node] # ❤👍️👍 表示老的 graph, 遍历顺序
        visit = {root.label : root} # ❤️ 👍👍 新的 graph 已经遍历过的
        
        while stack:
            top = stack.pop()
            
            for n in top.neighbors:
                if n.label not in visit:
                    stack.append(n)
                    visit[n.label] = UndirectedGraphNode(n.label)
                # ❤️👍👍 所以不用用 top.neighbors.append() 代替.
                visit[top.label].neighbors.append(visit[n.label])
        return root