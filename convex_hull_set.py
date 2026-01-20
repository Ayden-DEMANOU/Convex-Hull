import numpy as np            # For scientific computing in python 
import pandas as pd             # For data analysis and manipulation
from scipy.spatial import ConvexHull        # For convex hull 
import matplotlib.pyplot as plt            # For static, animated, and interactive visualizations.
from typing import List, Tuple, Union        # Type Hints	The "labels" on your storage boxes.

"""
from scipy.spatial import ConvexHull 
The Function: A library for scientific and technical computing.
Why use it? Specifically, the ConvexHull module finds the "outer boundary" of a set of points.
Key Role: Imagine stretching a rubber band around a group of nails in a board; the shape the rubber band takes 
is the Convex Hull.
Common Use: Identifying the perimeter or "envelope" of a cloud of data points.
"""

def get_boundary_points(data_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Uses SciPy (Qhull) to find the outer boundary (Convex Hull) of points.
    Parameters:
        data_points : np.ndarray of shape (n, 2)
                      where column 0 = longitude (x),
                            column 1 = latitude (y)
                            
    Returns:
        hull_points_x, hull_points_y : coordinates of hull vertices (closed loop)
    """
    if data_points.shape[0] < 3:
        raise ValueError("At least 3 points are required to compute a convex hull")

    hull = ConvexHull(data_points)

    # Extract hull vertices
    hull_points_x = data_points[hull.vertices, 0]
    hull_points_y = data_points[hull.vertices, 1]

    # Close the polygon
    hull_points_x = np.append(hull_points_x, hull_points_x[0])
    hull_points_y = np.append(hull_points_y, hull_points_y[0])

    return hull_points_x, hull_points_y


# =========================
# LOAD AND PREPARE DATA
# =========================

# Load dataset
df = pd.read_csv("vec_dyn_download.csv")

# Remove rows with missing coordinates
df = df.dropna(subset=["sample_lat_dd", "sample_long_dd"])

# Extract data as NumPy array (longitude = x, latitude = y)
data_points = df[["sample_long_dd", "sample_lat_dd"]].to_numpy()


# =========================
# COMPUTE CONVEX HULL
# =========================

bx, by = get_boundary_points(data_points)


# =========================
# METRICS
# =========================

# Number of original points
num_points = len(data_points)

# Perimeter
perimeter = 0.0
for i in range(len(bx) - 1):
    perimeter += np.linalg.norm(
        [bx[i+1] - bx[i], by[i+1] - by[i]]
    )

# Area (from Qhull)
hull = ConvexHull(data_points)
area = hull.volume   # In 2D, volume = area

print("1. Number of original points:", num_points)
print("2. Convex hull perimeter:", perimeter)
print("3. Enclosed area:", area)


# =========================
# VISUALIZATION
# =========================

plt.figure(figsize=(8, 5))
plt.scatter(
    df["sample_long_dd"],
    df["sample_lat_dd"],
    color="blue",
    label="Data Points"
)

plt.plot(bx, by, "r--", lw=2, label="Convex Hull (Boundary)")
plt.fill(bx, by, "r", alpha=0.15)

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Convex Hull of Major US Cities")
plt.legend()
plt.show()





"""def get_boundary_points(data_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    hull = ConvexHull(data_points)
    # Get the x and y coordinates of the hull vertices to close the loop
    hull_points_x = data_points[hull.vertices, 0]
    hull_points_y = data_points[hull.vertices, 1]
    
    # Append the first point to the end to close the shape in the plot
    hull_points_x = np.append(hull_points_x, hull_points_x[0])
    hull_points_y = np.append(hull_points_y, hull_points_y[0])
    
    return hull_points_x, hull_points_y

# 1. Create random data using NumPy
raw_data = np.random.rand(30, 2) 

# 2. Put it into a pandas DataFrame for easy viewing/organization
df = pd.DataFrame(raw_data, columns=['Longitude', 'Latitude'])

# 3. Calculate the boundary using our function
bx, by = get_boundary_points(df.values)

# 4. Visualize with Matplotlib
plt.figure(figsize=(8, 5))
plt.scatter(df['Longitude'], df['Latitude'], color='blue', label='Data Points')
plt.plot(bx, by, 'r--', lw=2, label='Convex Hull (Boundary)')
plt.fill(bx, by, 'r', alpha=0.1)
plt.title("Visualizing the Boundary of Random Data")
plt.legend()
plt.show()"""