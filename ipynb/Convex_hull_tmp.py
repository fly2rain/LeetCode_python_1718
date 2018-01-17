def cross (a, b, c):
	return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])

def convex_hull (points):

	nn = len (points)

	# case #1: points are fewer or equal to 1
	if nn <= 1:
		return -1

	# Sort the points lexicographically (tuples are compared lexicographically).
    # Remove duplicates to detect the case we have just one unique point.
	points = sorted ( set(points) )

	# case #2: lower parts 
	lower = []
	for p in points:

		if len(lower) >= 2 and cross( lower[-2], lower[-1], p ) <= 0:
			lower.pop ()
		else:
			lower.append (p)

	# case #3: high parts
	upper = []
	for p in points[::-1]:
		if len(upper) >= 2 and cross (upper[-2], upper[-1], p) <= 2:
			upper.pop ()
		else:
			upper.append (p)
			
	# Concatenation of the lower and upper hulls gives the convex hull.
    # Last point of each list is omitted because it is repeated at the beginning of the other list. 
	return lower + higher


