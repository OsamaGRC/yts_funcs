import io
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin, array_bounds
from rasterio.io import MemoryFile
from rasterio.warp import reproject, Resampling, calculate_default_transform
from pyproj import CRS
from shapely.geometry import Point
from sklearn.neighbors import KDTree
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error
from typing import Union, Tuple, Optional
from scipy.spatial import cKDTree
# from numba import njit, prange


def regress_rf(df1, df2, start=None, end=None, resample=None):
    """
    Performs linear regression with no intercept between matching columns
    of two datetime-indexed DataFrames. Supports resampling and slicing.

    Parameters:
    - df1, df2: pandas DataFrames with datetime index and matching columns
    - start, end: optional datetime strings or Timestamps to slice the index
    - resample: either a string (e.g., 'W') or a dict with keys:
                'rule', 'closed', 'label'

    Returns:
    - Transposed DataFrame with columns: coefficient, r2, rmse
    """

    # Handle resampling
    if resample:
        if isinstance(resample, str):
            rule = resample
            closed = 'right'
            label = 'right'
        elif isinstance(resample, dict):
            rule = resample.get('rule')
            closed = resample.get('closed', 'right')
            label = resample.get('label', 'right')
        else:
            raise ValueError("resample must be a string or a dictionary")

        if rule is None:
            raise ValueError("resample dictionary must include a 'rule' key")

        df1 = df1.resample(rule, closed=closed, label=label).sum()
        df2 = df2.resample(rule, closed=closed, label=label).sum()

    # Slice by datetime
    if start or end:
        df1 = df1.loc[start:end]
        df2 = df2.loc[start:end]

    # Align on index after slicing
    df1, df2 = df1.align(df2, join='inner')

    # Match columns
    common_columns = df1.columns.intersection(df2.columns)
    if len(common_columns) == 0:
        raise ValueError("No matching columns between the two dataframes.")

    model = LinearRegression(fit_intercept=False)
    results = {}

    for col in common_columns:
        x = df1[[col]].values
        y = df2[[col]].values

        # Remove NaNs
        mask = (~pd.isnull(x.flatten())) & (~pd.isnull(y.flatten()))
        if mask.sum() < 2:
            results[col] = [np.nan, np.nan, np.nan]
            continue

        x_clean = x[mask].reshape(-1, 1)
        y_clean = y[mask].reshape(-1, 1)

        model.fit(x_clean, y_clean)
        y_pred = model.predict(x_clean)

        coef = model.coef_[0][0]
        r2 = r2_score(y_clean, y_pred)
        rmse = root_mean_squared_error(y_clean, y_pred)

        results[col] = [coef, r2, rmse]

    return pd.DataFrame(results, index=["coefficient", "r2", "rmse"]).T


##### This function is intended to produce a grid data that covers Yass catchment from pluvial point data within the catchment extent
def rasterize_points(
    vector_data: Union[str, gpd.GeoDataFrame],
    attribute: str,
    cell_size: float = 0.15,        # ~2000m or 0.15 deg is enough
    method: str = 'voronoi',        # or 'idw' or 'scipy_idw'
    power: float = 2,
    projected_crs: Optional[Union[int, str, CRS]] = None,
    reproject_to_original: bool = False,
    output_raster: Optional[str] = None,
    return_result: bool = True,
    return_memoryfile: bool = False,
    k: Optional[int] = None
) -> Optional[Union[Tuple[np.ndarray, dict], MemoryFile]]:
    """
    Rasterize a point vector layer using Voronoi or IDW interpolation.

    Parameters:
        vector_data (str | GeoDataFrame): Path to vector layer or a GeoDataFrame.
        attribute (str): Name of attribute to interpolate.
        cell_size (float): Raster cell size (in meters if reprojected, or map units).
        method (str): 'voronoi', 'idw', or 'scipy_idw'.
        power (float): Power for IDW interpolation.
        projected_crs (int | str | CRS | None): CRS to reproject to for metric cell size.
        reproject_to_original (bool): Whether to reproject output raster back to original CRS.
        output_raster (str | None): File path to save the raster. If None, no file is saved.
        return_result (bool): If True, return raster and metadata.
        return_memoryfile (bool): If True, return raster as rasterio.io.MemoryFile.
        k (int | None): Number of nearest neighbors to use for IDW. Defaults to all points.

    Returns:
        - (raster_array, metadata) if return_result is True
        - MemoryFile if return_memoryfile is True
        - None otherwise
    """

    # Load data
    if isinstance(vector_data, str):
        gdf = gpd.read_file(vector_data)
    elif isinstance(vector_data, gpd.GeoDataFrame):
        gdf = vector_data.copy()
    else:
        raise TypeError("vector_data must be a file path or a GeoDataFrame.")

    if gdf.empty or not all(gdf.geometry.type == "Point"):
        raise ValueError("Input must be a non-empty point layer.")

    if attribute not in gdf.columns:
        raise ValueError(f"Attribute '{attribute}' not found in the input layer.")

    original_crs = gdf.crs
    if original_crs is None:
        raise ValueError("Input GeoDataFrame must have a CRS defined.")

    # Reproject if needed
    if projected_crs is not None:
        projected_crs = CRS.from_user_input(projected_crs)
        gdf = gdf.to_crs(projected_crs)
    else:
        projected_crs = original_crs

    # Define raster grid
    minx, miny, maxx, maxy = gdf.total_bounds
    width = int((maxx - minx) / cell_size)
    height = int((maxy - miny) / cell_size)
    transform = from_origin(minx, maxy, cell_size, cell_size)

    x = np.linspace(minx + cell_size / 2, maxx - cell_size / 2, width)
    y = np.linspace(miny + cell_size / 2, maxy - cell_size / 2, height)
    grid_x, grid_y = np.meshgrid(x, y[::-1])

    coords = np.array([[pt.x, pt.y] for pt in gdf.geometry], dtype=np.float64)
    values = gdf[attribute].to_numpy(dtype=np.float64)

    if k is None or k > len(coords):
        k = len(coords)

    if method == 'voronoi':
        tree = cKDTree(coords)
        _, idx = tree.query(np.c_[grid_x.ravel(), grid_y.ravel()], k=1)
        raster = values[idx].reshape((height, width))

    elif method == 'idw':
        tree = cKDTree(coords)
        points = np.c_[grid_x.ravel(), grid_y.ravel()]
        dists, idxs = tree.query(points, k=k)

        if k == 1:
            dists = dists[:, np.newaxis]
            idxs = idxs[:, np.newaxis]

        weights = 1.0 / (dists + 1e-10) ** power
        weighted_vals = np.sum(weights * values[idxs], axis=1) / np.sum(weights, axis=1)
        raster = weighted_vals.reshape((height, width))

    elif method == 'scipy_idw':
        from scipy.interpolate import griddata
        points = coords
        grid_points = np.c_[grid_x.ravel(), grid_y.ravel()]
        dists = np.linalg.norm(points[:, None, :] - grid_points[None, :, :], axis=2)
        weights = 1.0 / (dists + 1e-10) ** power
        weighted_vals = (weights.T @ values) / weights.sum(axis=1)
        raster = weighted_vals.reshape(grid_x.shape)

    else:
        raise ValueError(f"Invalid method '{method}'. Choose 'voronoi', 'idw', or 'scipy_idw'.")

    # Metadata
    meta = {
        'driver': 'GTiff',
        'height': raster.shape[0],
        'width': raster.shape[1],
        'count': 1,
        'dtype': raster.dtype,
        'crs': projected_crs,
        'transform': transform
    }

    # Optional reprojection
    if reproject_to_original and projected_crs != original_crs:
        bounds = array_bounds(raster.shape[0], raster.shape[1], transform)
        dst_transform, dst_width, dst_height = calculate_default_transform(
            projected_crs, original_crs,
            raster.shape[1], raster.shape[0],
            *bounds
        )
        reprojected = np.empty((dst_height, dst_width), dtype=raster.dtype)
        reproject(
            source=raster,
            destination=reprojected,
            src_transform=transform,
            src_crs=projected_crs,
            dst_transform=dst_transform,
            dst_crs=original_crs,
            resampling=Resampling.nearest
        )
        raster = reprojected
        transform = dst_transform
        meta.update({
            'height': dst_height,
            'width': dst_width,
            'transform': dst_transform,
            'crs': original_crs
        })

    # Save to file
    if output_raster:
        with rasterio.open(output_raster, 'w', **meta) as dst:
            dst.write(raster, 1)
        print(f"Raster saved to {output_raster}")

    # Return in-memory raster
    if return_memoryfile:
        memfile = MemoryFile()
        with memfile.open(**meta) as dataset:
            dataset.write(raster, 1)
        return memfile

    return (raster, meta) if return_result else None
