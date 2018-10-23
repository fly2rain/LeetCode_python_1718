class Solution(object):
	def n26_deleteDuplicatedElement_from_sortedArray(self, nums):
		if not nums or len(nums) < 2: return nums

		k = 1
		n = len(nums)
		for i in range(1, n):
			if nums[i] != nums[i-1]:
				nums[k] = nums[i]
				k += 1
		return k
