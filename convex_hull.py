"""
Convex Hull Calculation for Latitude/Longitude Coordinates

This notebook demonstrates how to calculate the convex hull from a set of 
latitude and longitude coordinates.

=============================================================================
PART 1: Understanding Convex Hull Concept
=============================================================================

What is a Convex Hull?
----------------------
The convex hull of a set of points is the smallest convex polygon that 
contains all the points. Think of it like stretching a rubber band around 
all the outermost points - the shape the rubber band forms is the convex hull.

Key Properties:
- Contains all points in the dataset (either inside or on the boundary)
- All interior angles are less than 180 degrees (convex property)
- Represents the minimal boundary that encloses all points
- Only the "outermost" points are vertices of the hull

Applications:
- Geographic boundary detection (e.g., service area coverage)
- Collision detection in computer graphics
- Pattern recognition and image processing
- Route optimization and spatial analysis

Algorithm:
----------
We'll use SciPy's implementation which uses the Qhull library. Qhull 
implements the QuickHull algorithm, which:
1. Finds points with extreme coordinates (leftmost, rightmost)
2. Divides remaining points into groups
3. Recursively finds the farthest point from each line segment
4. Builds the hull by connecting these extremal points

Time Complexity: O(n log n) on average, O(n²) worst case
Space Complexity: O(n)

=============================================================================
PART 2: Implementation
=============================================================================
"""

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from typing import List, Tuple, Union


def calculate_convex_hull(
    latitudes: Union[np.ndarray, pd.Series, List],
    longitudes: Union[np.ndarray, pd.Series, List]
) -> Tuple[np.ndarray, ConvexHull]:
    """
    Calculate the convex hull from latitude and longitude coordinates.
    
    Parameters:
    -----------
    latitudes : array-like
        Array or list of latitude values (y-coordinates)
    longitudes : array-like
        Array or list of longitude values (x-coordinates)
    
    Returns:
    --------
    hull_points : np.ndarray
        Array of shape (n_hull_points, 2) containing [longitude, latitude] 
        pairs of the convex hull vertices in counter-clockwise order
    hull : ConvexHull
        The SciPy ConvexHull object containing additional information
    
    Raises:
    -------
    ValueError : If inputs have different lengths or contain invalid values
    
    Example:
    --------
    >>> lats = [40.7128, 34.0522, 41.8781, 29.7604]
    >>> lons = [-74.0060, -118.2437, -87.6298, -95.3698]
    >>> hull_points, hull_obj = calculate_convex_hull(lats, lons)
    >>> print(f"Hull has {len(hull_points)} vertices")
    """
    
    # Convert inputs to numpy arrays
    lats = np.asarray(latitudes).flatten()
    lons = np.asarray(longitudes).flatten()
    
    # Validation
    if len(lats) != len(lons):
        raise ValueError(
            f"Latitudes and longitudes must have same length. "
            f"Got {len(lats)} latitudes and {len(lons)} longitudes."
        )
    
    if len(lats) < 3:
        raise ValueError(
            f"At least 3 points required for convex hull. Got {len(lats)}."
        )
    
    # Check for NaN values
    if np.any(np.isnan(lats)) or np.any(np.isnan(lons)):
        raise ValueError("Coordinates contain NaN values.")
    
    # Combine into points array [longitude, latitude]
    # Note: Using [lon, lat] order is standard for x-y plotting
    points = np.column_stack([lons, lats])
    
    # Calculate convex hull
    hull = ConvexHull(points)
    
    # Extract hull vertices in order
    # hull.vertices gives indices in counter-clockwise order for 2D
    hull_points = points[hull.vertices]
    
    return hull_points, hull


def plot_convex_hull(
    latitudes: Union[np.ndarray, pd.Series, List],
    longitudes: Union[np.ndarray, pd.Series, List],
    hull_points: np.ndarray,
    title: str = "Convex Hull of Geographic Points",
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Visualize the points and their convex hull.
    
    Parameters:
    -----------
    latitudes : array-like
        Original latitude values
    longitudes : array-like
        Original longitude values
    hull_points : np.ndarray
        Hull vertices from calculate_convex_hull()
    title : str
        Plot title
    figsize : tuple
        Figure size (width, height)
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The matplotlib figure object
    """
    
    lats = np.asarray(latitudes).flatten()
    lons = np.asarray(longitudes).flatten()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot all points
    ax.scatter(lons, lats, c='blue', s=50, alpha=0.6, 
               label=f'All Points (n={len(lats)})', zorder=2)
    
    # Plot hull vertices
    ax.scatter(hull_points[:, 0], hull_points[:, 1], 
               c='red', s=100, marker='^', 
               label=f'Hull Vertices (n={len(hull_points)})', zorder=3)
    
    # Draw hull polygon (close it by appending first point)
    hull_closed = np.vstack([hull_points, hull_points[0]])
    ax.plot(hull_closed[:, 0], hull_closed[:, 1], 
            'r-', linewidth=2, label='Convex Hull', zorder=4)
    ax.fill(hull_closed[:, 0], hull_closed[:, 1], 
            'red', alpha=0.1, zorder=1)
    
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Equal aspect ratio for geographic data
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    return fig


def get_hull_statistics(hull: ConvexHull, hull_points: np.ndarray) -> dict:
    """
    Calculate statistics about the convex hull.
    
    Parameters:
    -----------
    hull : ConvexHull
        The SciPy ConvexHull object
    hull_points : np.ndarray
        Hull vertices
    
    Returns:
    --------
    stats : dict
        Dictionary containing hull statistics
    """
    
    stats = {
        'n_vertices': len(hull_points),
        'perimeter': hull.area,  # In 2D, 'area' is actually perimeter
        'enclosed_area': hull.volume,  # In 2D, 'volume' is actually area
        'centroid_lon': np.mean(hull_points[:, 0]),
        'centroid_lat': np.mean(hull_points[:, 1]),
        'lon_range': (hull_points[:, 0].min(), hull_points[:, 0].max()),
        'lat_range': (hull_points[:, 1].min(), hull_points[:, 1].max()),
    }
    
    return stats


# =============================================================================
# EXAMPLE USAGE (Replace with your actual data)
# =============================================================================

if __name__ == "__main__":
    
    # Example 1: Sample data (US cities)
    print("=" * 70)
    print("EXAMPLE: Convex Hull of Major US Cities")
    print("=" * 70)
    
    cities_data = {
        'city': ['New York', 'Los Angeles', 'Chicago', 'Houston', 
                 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego',
                 'Dallas', 'San Jose', 'Miami', 'Seattle'],
        'latitude': [40.7128, 34.0522, 41.8781, 29.7604, 
                     33.4484, 39.9526, 29.4241, 32.7157,
                     32.7767, 37.3382, 25.7617, 47.6062],
        'longitude': [-74.0060, -118.2437, -87.6298, -95.3698,
                      -112.0740, -75.1652, -98.4936, -117.1611,
                      -96.7970, -121.8863, -80.1918, -122.3321]
    }
    
    df = pd.DataFrame(cities_data)
    
    # Calculate convex hull
    hull_points, hull_obj = calculate_convex_hull(
        df['latitude'], 
        df['longitude']
    )
    
    # Get statistics
    stats = get_hull_statistics(hull_obj, hull_points)
    
    print(f"\nNumber of original points: {len(df)}")
    print(f"Number of hull vertices: {stats['n_vertices']}")
    print(f"Perimeter: {stats['perimeter']:.2f} degrees")
    print(f"Enclosed area: {stats['enclosed_area']:.2f} square degrees")
    print(f"Centroid: ({stats['centroid_lat']:.4f}°, {stats['centroid_lon']:.4f}°)")
    print(f"\nLongitude range: {stats['lon_range']}")
    print(f"Latitude range: {stats['lat_range']}")
    
    print("\nHull vertices (in counter-clockwise order):")
    print("-" * 50)
    for i, (lon, lat) in enumerate(hull_points):
        city_idx = df[(df['longitude'] == lon) & (df['latitude'] == lat)].index
        if len(city_idx) > 0:
            city_name = df.loc[city_idx[0], 'city']
            print(f"{i+1}. {city_name:15s} ({lat:8.4f}°, {lon:9.4f}°)")
    
    # Visualize
    fig = plot_convex_hull(
        df['latitude'], 
        df['longitude'], 
        hull_points,
        title="Convex Hull of Major US Cities"
    )
    plt.show()
    
    
    # Example 2: VectorByte Dataset Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS: VectorByte Dataset Convex Hull")
    print("=" * 70)
    
    # Load the VectorByte data
    try:
        # Read the file using window.fs.readFile API
        import json
        file_content = await_window.fs.readFile('vec_dyn_download.csv', {'encoding': 'utf8'})
        from io import StringIO
        vb_df = pd.read_csv(StringIO(file_content))
        
        print(f"\nDataset loaded successfully!")
        print(f"Total records: {len(vb_df)}")
        print(f"Columns: {vb_df.shape[1]}")
        
        # Clean the data - remove rows with missing coordinates
        print(f"\nCleaning data...")
        original_count = len(vb_df)
        vb_clean = vb_df.dropna(subset=['sample_lat_dd', 'sample_long_dd'])
        cleaned_count = len(vb_clean)
        print(f"Records with valid coordinates: {cleaned_count}")
        print(f"Records removed: {original_count - cleaned_count}")
        
        if cleaned_count < 3:
            print("\nError: Need at least 3 points for convex hull!")
        else:
            # Calculate convex hull
            print(f"\nCalculating convex hull...")
            vb_hull_points, vb_hull_obj = calculate_convex_hull(
                vb_clean['sample_lat_dd'],
                vb_clean['sample_long_dd']
            )
            
            # Get statistics
            vb_stats = get_hull_statistics(vb_hull_obj, vb_hull_points)
            
            print(f"\n{'='*70}")
            print("CONVEX HULL RESULTS")
            print(f"{'='*70}")
            print(f"Total data points: {cleaned_count:,}")
            print(f"Hull vertices: {vb_stats['n_vertices']}")
            print(f"Perimeter: {vb_stats['perimeter']:.2f} degrees")
            print(f"Enclosed area: {vb_stats['enclosed_area']:.2f} square degrees")
            print(f"Centroid: ({vb_stats['centroid_lat']:.4f}°, {vb_stats['centroid_lon']:.4f}°)")
            print(f"\nGeographic extent:")
            print(f"  Longitude: {vb_stats['lon_range'][0]:.4f}° to {vb_stats['lon_range'][1]:.4f}°")
            print(f"  Latitude:  {vb_stats['lat_range'][0]:.4f}° to {vb_stats['lat_range'][1]:.4f}°")
            
            print(f"\n{'='*70}")
            print("HULL VERTICES (Counter-clockwise order)")
            print(f"{'='*70}")
            print(f"{'#':<4} {'Longitude':<12} {'Latitude':<12}")
            print("-" * 30)
            for i, (lon, lat) in enumerate(vb_hull_points):
                print(f"{i+1:<4} {lon:<12.6f} {lat:<12.6f}")
            
            # Export hull points as list
            print(f"\n{'='*70}")
            print("HULL POINTS AS PYTHON LIST")
            print(f"{'='*70}")
            hull_list = vb_hull_points.tolist()
            print("hull_vertices = [")
            for lon, lat in hull_list:
                print(f"    [{lon:.6f}, {lat:.6f}],")
            print("]")
            
            # Visualize
            print(f"\nGenerating visualization...")
            fig = plot_convex_hull(
                vb_clean['sample_lat_dd'],
                vb_clean['sample_long_dd'],
                vb_hull_points,
                title="Convex Hull of VectorByte Data (vec_dyn_download.csv)",
                figsize=(14, 10)
            )
            plt.show()
            
            # Save hull points to CSV
            hull_df = pd.DataFrame(vb_hull_points, columns=['longitude', 'latitude'])
            hull_df['vertex_number'] = range(1, len(hull_df) + 1)
            hull_df = hull_df[['vertex_number', 'longitude', 'latitude']]
            
            print(f"\n{'='*70}")
            print("EXPORT OPTIONS")
            print(f"{'='*70}")
            print("\nTo save hull vertices to CSV:")
            print("hull_df.to_csv('convex_hull_vertices.csv', index=False)")
            print("\nTo save hull vertices as JSON:")
            print("import json")
            print("with open('hull_vertices.json', 'w') as f:")
            print("    json.dump(hull_list, f, indent=2)")
            
    except Exception as e:
        print(f"\nError loading VectorByte data: {e}")
        print("\nTo use with your downloaded dataset:")
        print("""
# 1. Load your data
vb_df = pd.read_csv('vec_dyn_download.csv')

# 2. Clean data (remove missing coordinates)
vb_clean = vb_df.dropna(subset=['sample_lat_dd', 'sample_long_dd'])

# 3. Calculate convex hull
hull_points, hull_obj = calculate_convex_hull(
    vb_clean['sample_lat_dd'],
    vb_clean['sample_long_dd']
)

# 4. Get hull as list
hull_list = hull_points.tolist()

# 5. Visualize
fig = plot_convex_hull(
    vb_clean['sample_lat_dd'],
    vb_clean['sample_long_dd'],
    hull_points
)
plt.show()
""")