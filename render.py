import numpy as np
from tqdm import tqdm

from utils import batch_project


def compute_projected_bboxes(cov_2d, positions, transform_matrix, screen_height, screen_width):
    """Compute projected bounding boxes for 2D Gaussians in both camera and NDC space.

    Parameters
    ----------
    cov_2d : ndarray
        2D covariance matrices for each Gaussian, shape (N, 2, 2)
    positions : ndarray
        3D positions of Gaussians in world space, shape (N, 3)
    transform_matrix : ndarray
        Transformation matrix from world to NDC space, shape (4, 4)
    screen_height : int
        Height of the output image in pixels
    screen_width : int
        Width of the output image in pixels

    Returns
    -------
    tuple
        bbox_cam : ndarray
            Camera-space bounding boxes, shape (N, 4, 2)
        bbox_ndc : ndarray
            Normalized Device Coordinate (NDC) bounding boxes, shape (N, 4, 2)

    Notes
    -----
    The bounding boxes are computed using 3σ rule (99.7% of the Gaussian's mass).
    The corners are ordered as: top-left, top-right, bottom-right, bottom-left.
    """
    # Define unit square corners in order: tl, tr, br, bl
    corners = np.array([[-1, 1], [1, 1], [1, -1], [-1, -1]])
    # 1. Compute bounding box size in camera space (3σ standard deviations for 99.7% coverage)
    #     The covariance matrix's diagonal elements are the variances along x and y.
    #     The standard deviation is the square root of variance.
    std = 3.0 * np.sqrt(np.diagonal(cov_2d, axis1=1, axis2=2))
    # 2. Convert bbox size from camera space to NDC ([-1, 1] range)
    #     Divide out camera plane size to get bounding box size in NDC (Normalized Device Coordinates)
    #     NDC ranges from -1 to 1 in both axes, so dividing by the camera's dimensions scales the size to the NDC range.
    #     For example, if the camera's width is w pixels, then each pixel in x corresponds to 2/w in NDC (since from -1 to 1 is 2 units).
    ndc = std / np.array([screen_width, screen_height]) * 2
    """
    3. Scale corners by bbox size in camera space for each Gaussian
      (-3σx, 3σy) ┌───────┐ (3σx, 3σy)
                  │       │
                  │   ·   │
                  │       │
     (-3σx, -3σy) └───────┘ (3σx, -3σy)
    """
    bbox_cam = corners * std[:, None, :]
    """
    4. Compute bbox corners in NDC by offsetting from the projected center
      (Cx - 6σx/W, Cy + 6σy/H) ┌───────┐ (Cx + 6σx/W, Cy + 6σy/H)
                               │       │
                               │   ·   │
                               │       │
      (Cx - 6σx/W, Cy - 6σy/H) └───────┘ (Cx + 6σx/W, Cy - 6σy/H)
    """
    bbox_ndc = corners * ndc[:, None, :] + batch_project(positions, transform_matrix=transform_matrix)[:, None, :2]
    return bbox_cam, bbox_ndc


def ndc_to_screen(ndc, h, w):
    """
    NDC
    (-1,1)        (1,1)
       ┌───────────┐
       │           │
       │     ↑ y   │
       │     │     │
       │     └──→ x│
       │           │
       └───────────┘
    (-1,-1)       (1,-1)

    Screen
    (0,h)        (w,h)
       ┌───────────┐
       │           │
       │     ↑ y   │
       │     │     │
       │     └──→ x│
       │           │
       └───────────┘
    (0,0)        (w,0)
    """
    return (ndc + 1) * np.array([w, h]) * 0.5


def compute_gaussian_opacity(opacity, color, conic, bbox_cam, bounds, coords, bitmap):
    """Computes Gaussian opacity contribution and updates render buffers.

    Parameters
    ----------
    opacity : float
        Base opacity of the Gaussian
    color : ndarray
        RGB color of the Gaussian, shape (3,)
    conic : ndarray
        Inverse covariance matrix (conic), shape (2, 2)
    bbox_cam : ndarray
        Camera-space bounding box corners, shape (4, 2)
    bounds : tuple
        Screen-space bounds (x_min, x_max, y_min, y_max)
    coords : tuple
        Tuple of flattened pixel coordinates (xx_flat, yy_flat)
    bitmap : ndarray
        RGB color buffer being updated, shape (H, W, 3)
    """
    x_min, x_max, y_min, y_max = bounds
    xx_flat, yy_flat = coords

    # Map screen coordinates to camera space using linear interpolation
    # bbox_cam: [tl, tr, br, bl]
    x = np.column_stack(
        [
            np.interp(xx_flat, [x_min, x_max], bbox_cam[[0, 1], 0]),
            np.interp(yy_flat, [y_min, y_max], bbox_cam[[2, 1], 1]),
        ]
    )

    # Compute Gaussian opacity (alpha) at each pixel
    # α = opacity * exp(-0.5 * x^T * sigma * x)
    alpha = np.clip(opacity * np.exp(-0.5 * np.einsum("ni,ij,nj->n", x, conic, x)), 0.0, 0.99)

    # Create mask for pixels with significant contribution
    # Avoid computations for negligible contributions
    mask = alpha >= 1 / 255
    if not mask.any():
        return

    # Update color buffer using alpha blending
    valid_yx, valid_alpha = (yy_flat[mask], xx_flat[mask]), alpha[mask]
    bitmap[valid_yx] = color * valid_alpha[:, None] + bitmap[valid_yx] * (1 - valid_alpha[:, None])


def render_gaussian(depth, opacity, color, conic, bbox_cam, bbox_ndc, screen_height, screen_width):
    """Renders 2D Gaussians to screen buffers using splatting.

    Parameters
    ----------
    depth : ndarray
        Per-Gaussian depth values (for depth sorting), shape (N,)
    opacity : ndarray
        Per-Gaussian opacity values, shape (N,)
    color : ndarray
        Per-Gaussian RGB colors, shape (N, 3)
    conic : ndarray
        Per-Gaussian conic matrices (inverse covariance), shape (N, 2, 2)
    bbox_cam : ndarray
        Camera-space bounding boxes, shape (N, 4, 2)
    bbox_ndc : ndarray
        Normalized Device Coordinate (NDC) bounding boxes, shape (N, 4, 2)
    screen_height : int
        Output image height (pixels)
    screen_width : int
        Output image width (pixels)

    Returns
    -------
    ndarray
        Rendered RGB image, shape (screen_height, screen_width, 3)
    """

    # Convert NDC coordinates to screen pixel coordinates
    # This ensures Gaussians are positioned correctly on screen
    bbox_screen = ndc_to_screen(bbox_ndc, screen_height, screen_width)

    # Separate x and y coordinates from bounding boxes: [N, 4, 2] -> [2, N, 4]
    # Resulting shapes: x_coords [N, 4], y_coords [N, 4]
    x_coords, y_coords = bbox_screen.transpose(2, 0, 1)

    # Calculate screen-space bounding boxes for each Gaussian
    # Using floor/ceil to cover all potentially affected pixels
    x_min, x_max = np.floor(np.min(x_coords, axis=1)), np.ceil(np.max(x_coords, axis=1))
    y_min, y_max = np.floor(np.min(y_coords, axis=1)), np.ceil(np.max(y_coords, axis=1))

    # Clip bounding boxes to screen boundaries
    # Prevents out-of-bounds memory access
    x_start, x_end = np.maximum(0, x_min), np.minimum(screen_width, x_max)
    y_start, y_end = np.maximum(0, y_min), np.minimum(screen_height, y_max)

    # Create visibility mask: True for Gaussians with at least one visible pixel
    # Return blank image if no visible Gaussians
    mask = (x_start < x_end) & (y_start < y_end)
    if not mask.any():
        return np.zeros((screen_height, screen_width, 3))

    # Filter data for visible Gaussians only
    # Reduces computation for off-screen elements
    x_min, x_max = x_min[mask], x_max[mask]
    y_min, y_max = y_min[mask], y_max[mask]

    x_start, x_end = x_start[mask], x_end[mask]
    y_start, y_end = y_start[mask], y_end[mask]

    valid_color = color[mask]
    valid_conic = conic[mask]
    valid_opacity = opacity[mask]
    valid_bbox_cam = bbox_cam[mask]

    # Initialize RGB color buffer
    bitmap = np.zeros((screen_height, screen_width, 3), dtype=np.float32)

    # Process each Gaussian in depth-sorted order (painter's algorithm)
    # Critical for correct alpha blending of transparent surfaces
    for idx in tqdm(np.argsort(-depth[mask]), desc="🎨 Rendering"):
        # Obtain integer pixel boundaries for current Gaussian
        xs, xe = x_start[idx], x_end[idx]
        ys, ye = y_start[idx], y_end[idx]

        # Generate pixel coordinates within bounding box (only update local region for efficiency).
        xx, yy = np.meshgrid(np.arange(xs, xe, dtype=np.int32), np.arange(ys, ye, dtype=np.int32))

        # Compute Gaussian contribution to local region
        compute_gaussian_opacity(
            opacity=valid_opacity[idx],
            color=valid_color[idx],
            conic=valid_conic[idx],
            bbox_cam=valid_bbox_cam[idx],
            bounds=(x_min[idx], x_max[idx], y_min[idx], y_max[idx]),
            coords=(xx.ravel(), yy.ravel()),
            bitmap=bitmap,
        )
    return bitmap
