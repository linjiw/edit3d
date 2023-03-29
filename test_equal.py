import numpy as np

# List of points
points_list = np.array([(0, 0, 0), (1, 2, 3), (4, 5, 6), (7, 8, 9)])

# Point to check
point = np.array([1, 2, 4])

# Check if point is inside the list of points
# if np.all(np.equal(points_list, point).all()):
#     print("Point is inside the list of points")
# else:
#     print("Point is not inside the list of points")
print(f"{points_list[np.equal(points_list, point).all(axis=1)]}")
print(f"{np.any(np.equal(points_list, point).all(axis=1))}")