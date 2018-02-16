# ##### evaluate the two methods creating the 2-d lists #######
# import time 
# NN   = 1000
# m, n = 10, 20

# # Alg 1: [ [None] * m ]  * n 
# start = time.clock ()
# for i in xrange (NN):
#     tmp = [ [None]*n ] * m
# print time.clock () - start

# # Alg 2:  [ [None] * m ]  * n 
# start = time.clock ()
# for i in xrange (NN):
#     tmp = [ [None for j in xrange(n)] for k in xrange (m)  ]
# print time.clock () - start
# # Conclusion: Alg 2 is much better than Alg 1 in speed


# def longestCommonSubstring (S, T):
#     """
#     : type S: str
#     : type T: str
#     : rtype: str
#     """
#     m, n = len(S), len(T)
#     DP   = [ [0] * n  for i in xrange (m) ]
    
#     z   = 0
#     rst = ''
#     for i in xrange (m):
#         for j in xrange (n):
#             if S[i] == T[j]:
#                 # set the DP value 
#                 if i == 0 or j == 0:
#                     DP[i][j] = 1
#                 else:
#                     DP[i][j] = DP[i-1][j-1] + 1
                  
#                 # check whether DP[i][j] is the longest sub string
#                 if DP[i][j] > z:
#                     z   = DP[i][j]
#                     rst = S[i-z+1 : i+1]
#             else:
#                 DP[i][j] = 0                 
#     return rst



# def longestCommonSubstring_v2 (S, T):
#     """
#     : type S: str
#     : type T: str
#     : rtype: str
#     """
#     m, n = len(S), len(T)
#     DP   = [ [0] * (n+1) for i in xrange (m+1) ]
    
#     z   = 0
#     rst = ''
#     for i in xrange (1, m+1):
#         for j in xrange (1, n+1):
#             if S[i-1] == T[j-1]:
#                 DP[i][j] = DP[i-1][j-1] + 1

#                 # check whether DP[i][j] is the longest sub string
#                 if DP[i][j] > z:
#                     z   = DP[i][j]
#                     rst = S[i-z : i+1]  
#             else:
#                 DP[i][j] = 0 
#     print z, DP             
#     return rst 



#         for i in xrange (1, m+1):
#         for j in xrange (1, n+1):
            
#             if S[i-1] == T[j-1]:
#                 # if (i-1, j-1) in DP:
#                 #     DP[(i, j)] = DP[(i-1,j-1)] + 1
#                 # else:
#                 #     DP[(i,j)]  = 1
                
#                 ## 或者这样的代码更紧凑一些, 但更难看懂一些
#                 tmp = DP[(i-1,j-1)] if (i-1,j-1) in DP else 0
#                 DP[(i,j)] = tmp + 1

#                 # check whether DP[i][j] is the longest sub string
#                 if DP[(i,j)] > z:
#                     z   = DP[(i,j)]
#                     rst = S[i-z : i] # 原来和 _v3 的差别是, 在于绝对数值和 index 的区别.      
#     return rst, z

# ### testing ####################
# print longestCommonSubstring ('abc', 'zbcd')
# print longestCommonSubstring_v2 ('abc', 'zbcd')



class Solution(object):        
    def tree2str(self, t):
        """
        :type t: TreeNode
        :rtype: str
        """
        if not t: return ""
        
        if not t.left and not t.right:
            return str(t.val)

        if not t.right:
            return str(t.val) + "(" + self.tree2str(t.left) + ")"

        return str(t.val) + "(" self.tree2str(t.left) ")(" + self.tree2str(t.right) + ")" 


class Solution(object):
    def buildTree(self, inorder, postorder):
        """
        :type inorder: List[int]
        :type postorder: List[int]
        :rtype: TreeNode
        """
        if not len(inorder) == len(postorder): return None
        indic = {}
        for i, x in enumerate(inorder): indic[x] = i
        return self.helper(0, len(inorder)-1, 0, len(inorder)-1, indic, postorder)
    
    def helper(self, indic, postorder, inLow, inHigh, poLow, poHigh):
        if poLow > poHigh or inLow > inHigh: return None
        
        node = TreeNode(postorder[poHigh])
        inIdx = indic[node.val]

        left_length = i
        node.left = self.helper(poLow, poLow+left_length-1, inLow, inIdx-1, 
                               indic, postorder)
        node.right = self.helper(poLow+inIdx-inHigh, poHigh-1, inIdx+1, inHigh,
                                indic, postorder) 
        return node