"""
Author: Li Wang
Question: Delivery problem with mid-way checkpoints
Town Setting: In a 2-D Matrix
'G': goal
'X': checkpoint
'S': starting point
 0 : open block
 1 : closed block
Idea: BFS. Steps are as follow:
1. find locations of S, all checkpoints X (in a list), and goal G
2. use BFS to get shortest paths from S to each checkpoint X. Save in List.
Each checkpoint is matched under the same list index
Return -1 if not reachable
3. get shortest paths from each X to each other X
4. get shortest path from each X to the G
5. find total shortest distance from starting point, via all midway checkpoints to the ending point
Assumptions: has and only has one goal point and one starting point.
"""
from collections import deque
from Node import Node
import itertools
import sys


# Class Node that saves previous path in a list and the current point

class Node(object):
	def __init__(self, prev, cur):
		# type prev: List[Tuple]
		# type cur: Tuple
		self.prev = prev
		self.cur = cur

class Solution(object):
	# actions() helper method. Return possible actions from given location
	def actions(self, matrix, location):
		# rtype: List[Tuple]
		res = []
		row = location[0]
		col = location[1]
		if row != 0 and matrix[row-1][col] != 1: 
			res.append((row-1, col))
		if row != len(matrix)-1 and matrix[row+1][col] != 1: 
			res.append((row+1, col))
		if col != 0 and matrix[row][col-1] != 1: 
			res.append((row, col-1))
		if col != len(matrix[0])-1 and matrix[row][col+1] != 1: 
			res.append((row, col+1))
		return res

	def delivery(self, matrix):
		# Step 1: get the location of start_point, goal_point, all checkpoints
		start_point  = ()	
		check_points = []	# List of point tuples
		goal_point = ()
		row_nums = len(matrix)
		col_nums = len(matrix[0])
		for i in xrange(row_nums):
			for j in xrange(col_nums):
				if matrix[i][j] == 'S':
					start_point = (i,j)
				elif matrix[i][j] == 'G':
					goal_point = (i,j)
				elif matrix[i][j] == 'X':
					check_points.append((i,j))

		if len(check_points) == 0:
			print "No Checkpoint Exists"
			return -1

		# Step 2. Find shortest paths from 'S' -> Each 'X'
		S_X_paths = []
		for i in xrange( len(check_points) ):
			check_point = check_points[i]
			# BFS
			queue = deque([])
			queue.append( Node([], start_point) )
			visited = set()
			
			while queue:
				node     = queue.popleft()
				location = node.cur		# Tuple
				prev = node.prev		# List
				visited.add(location)

				# Checkpoint reached. Record this path.
				if location == check_point:
					prev.append(location)
					S_X_paths.append(prev)
					break
				
				for action in self.actions(matrix, location):
					if action in visited: 
						continue
					newPath = prev + [location]
					queue.append( Node(newPath, action) )

		# Some checkpoints are not reachable. Return -1
		if len(check_points) != len(S_X_paths): 
			print "Some Checkpoints Are Unreachable"
			return -1
		
		# Step 3. Find shortest paths between each 'X' <-> each other 'X'
		X_X_paths = []
		for i in range( len(check_points) ):
			start = check_points[i]
			for j in range( i+1, len(check_points) ):
				end = check_points[j]
				# BFS
				queue = deque( [] )
				queue.append( Node([], start) )
				visited = set()
				while queue:
					node = queue.popleft()
					location = node.cur		# Tuple
					prev = node.prev		# List
					visited.add(location)

					# Checkpoint reached. Record this path.
					if location == end:
						prev.append(location)
						X_X_paths.append(prev)
						break
					
					for action in self.actions(matrix, location):
						if action in visited: 
							continue
						newPath = prev + [location]
						queue.append(Node(newPath, action))
		
		# Step 4. Find shortest paths from each 'X' to 'G'
		X_G_paths = []
		for check_point in check_points:
			# BFS
			queue = deque([])
			queue.append(Node([],check_point))
			visited = set()
			
			while queue:
				node = queue.popleft()
				location = node.cur		# Tuple
				prev = node.prev		# List
				visited.add(location)

				# Goal reached. Record this path.
				if location == goal_point:
					prev.append(location)
					X_G_paths.append(prev)
					break
				
				for action in self.actions(matrix, location):
					if action in visited: 
						continue
					newPath = prev + [location]
					queue.append(Node(newPath, action))
		# Some checkpoints cannot reach goal. Return -1
		if len(check_points) != len(X_G_paths): 
			print "Some Checkpoints Cannot Reach The Goal"
			return -1

		# Step 5. Find minimum path from S to G via all X
		# 3 sets of dictionaries to store path costs
		S_X_dic = {}
		for path in S_X_paths:
			S_X_dic[path[-1]] = len(path) - 1
		X_dic = {}
		for path in X_X_paths:
			X_dic[(path[0], path[-1])] = len(path) - 1
		X_G_dic = {}
		for path in X_G_paths:
			X_G_dic[path[0]] = len(path) - 1

		# Generator to get different paths
		gen = itertools.permutations(check_points)
		res_path = []
		min_length = sys.maxint
		while True:
			try:
				path = [start_point] + list(gen.next()) + [goal_point]
				length = 0
				for i in range(len(path)-1):
					# S -> X
					if i == 0:
						x = path[i+1]
						length += S_X_dic[x]
					elif i == len(path)-2:
						x = path[i]
						length += X_G_dic[x]
					else:
						if (path[i], path[i+1]) in X_dic:
							length += X_dic[(path[i], path[i+1])]
						else:
							length += X_dic[(path[i+1], path[i])]
				if length < min_length:
					min_length = length
					res_path = path
			# End iteration
			except StopIteration:
				break
		
		# Display the path taken
		print "Shortest Path:"
		print res_path

		return min_length