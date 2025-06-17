import numpy as np
from plyfile import PlyData

from spherical import eval_sh
from utils import sigmoid, normalize, batch_project
from render import compute_projected_bboxes, render_gaussian


def quaternion_to_rotation(quaternions):
    """Convert quaternions to rotation matrices.

    This function converts a batch of normalized quaternions to their corresponding 3x3 rotation matrices.
    The quaternions should be in the format q = r + xi + yj + zk, where r is the real part
    and x, y, z are the imaginary components.

    Parameters
    ----------
    quaternions : ndarray
        Input quaternions array of shape (N, 4), where N is the number of quaternions.
        Each quaternion should be in the format [r, x, y, z].

    Returns
    -------
    ndarray
        Rotation matrices of shape (N, 3, 3), where N is the number of input quaternions.
        Each 3x3 matrix represents the corresponding rotation.

    Notes
    -----
    The rotation matrix R is computed as:
    R = [
        [1 - 2y² - 2z², 2xy - 2rz,     2xz + 2ry    ],
        [2xy + 2rz,     1 - 2x² - 2z², 2yz - 2rx    ],
        [2xz - 2ry,     2yz + 2rx,     1 - 2x² - 2y²]
    ]

    Examples
    --------
    >>> quaternions = np.array([[1, 1, 1, 1]])
    >>> R = quaternion_to_rotation(quaternions)
    >>> print(R[0])
    [[-3. 0. 4.]
     [4. -3. 0.]
     [0. 4. -3.]]
    """
    r, x, y, z = quaternions.T

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    rx, ry, rz = r * x, r * y, r * z

    R = np.empty((len(quaternions), 3, 3), dtype=np.float32)
    R[:, 0] = np.stack([1 - 2 * (yy + zz), 2 * (xy - rz), 2 * (xz + ry)], axis=1)
    R[:, 1] = np.stack([2 * (xy + rz), 1 - 2 * (xx + zz), 2 * (yz - rx)], axis=1)
    R[:, 2] = np.stack([2 * (xz - ry), 2 * (yz + rx), 1 - 2 * (xx + yy)], axis=1)
    return R


def build_scaling_rotation(scales, quaternions):
    """Build transformation matrix from scales and quaternions.

    Constructs a transformation matrix A = R @ S by efficiently combining rotation
    and scaling operations. The computation is optimized to avoid explicit matrix
    multiplication by leveraging broadcasting.

    Parameters
    ----------
    scales : ndarray
        Array of shape (N, 3) containing the scale factors for each dimension.
    quaternions : ndarray
        Array of shape (N, 4) containing the quaternions representing rotations.
        Quaternions should be in (r, x, y, z) format.

    Returns
    -------
    ndarray
        Array of shape (N, 3, 3) containing the transformation matrices.
        Each matrix is the product of rotation and scaling matrices.

    Notes
    -----
    The computation is optimized to avoid explicit matrix multiplication:
    - Instead of computing R @ S where S = np.eye(3) * scales[:, None]
    - We directly compute R * scales[:, None] using broadcasting

    This is equivalent to:
    [[1, 2, 3],    [[1, 0, 0],    [[1,  4,  9],
     [4, 5, 6],  @  [0, 2, 0], =   [4, 10, 18],
     [7, 8, 9]]     [0, 0, 3]]     [7, 16, 27]]

    Which is the same as:
    [[1, 2, 3],    [[1, 2, 3],    [[1,  4,  9],
     [4, 5, 6],  *  [1, 2, 3], =   [4, 10, 18],
     [7, 8, 9]]     [1, 2, 3]]     [7, 16, 27]]

    Examples
    --------
    >>> scales = np.array([[1.0, 2.0, 3.0]])
    >>> quaternions = np.array([[1.0, 0.0, 0.0, 0.0]])
    >>> transform = build_scaling_rotation(scales, quaternions)
    >>> print(transform.shape)
    (1, 3, 3)
    """
    return quaternion_to_rotation(quaternions) * scales[:, None]


def compute_covariance_3d(scales, quaternions):
    """Compute 3D covariance matrix from scales and quaternions.

    The covariance matrix is computed as Σ = A @ A^T, where A = R @ S is the
    transformation matrix obtained from scaling and rotation operations.

    Parameters
    ----------
    scales : ndarray
        Array of shape (N, 3) containing the scale factors for each dimension.
    quaternions : ndarray
        Array of shape (N, 4) containing the quaternions representing rotations.
        Quaternions should be in (r, x, y, z) format.

    Returns
    -------
    ndarray
        Array of shape (N, 3, 3) containing the covariance matrices.
        Each matrix is symmetric and positive semi-definite.

    Notes
    -----
    The computation follows these steps:
    1. Build transformation matrix A = R @ S where:
       - R is the rotation matrix from quaternions
       - S is the scaling matrix from scales
    2. Compute covariance as Σ = A @ A^T

    Examples
    --------
    >>> scales = np.array([[1.0, 2.0, 3.0]])
    >>> quaternions = np.array([[1.0, 0.0, 0.0, 0.0]])  # identity rotation
    >>> cov = compute_covariance_3d(scales, quaternions)
    >>> print(cov.shape)
    (1, 3, 3)
    """
    transformation = build_scaling_rotation(scales, quaternions)
    return transformation @ transformation.transpose(0, 2, 1)


def batch_compute_projected_covariance(mean_3d, cov_3d, focal_x, focal_y, tan_fovx, tan_fovy, view_matrix):
    """Compute the projected 2D covariance matrices for 3D Gaussians.

    Parameters
    ----------
    mean_3d : ndarray
        3D positions of Gaussians in world space, shape (N, 3)
    cov_3d : ndarray
        3D covariance matrices for each Gaussian, shape (N, 3, 3)
    focal_x : float
        Focal length in x direction
    focal_y : float
        Focal length in y direction
    tan_fovx : float
        Tangent of half the field of view in x direction
    tan_fovy : float
        Tangent of half the field of view in y direction
    view_matrix : ndarray
        View transformation matrix, shape (4, 4)

    Returns
    -------
    ndarray
        Projected 2D covariance matrices, shape (N, 2, 2)

    Notes
    -----
    This function implements the projection of 3D Gaussian covariances to 2D screen space.
    The projection follows the perspective camera model and includes:
    1. Projection of 3D points to camera space
    2. Clipping to field of view
    3. Computation of Jacobian for the projection
    4. Transformation of covariance through the view matrix
    5. Application of a low-pass filter to ensure minimum size
    """

    # Project 3D points to camera space and unpack coordinates
    x_cam, y_cam, z_cam = batch_project(mean_3d, view_matrix).T

    # Define clipping bounds for field of view (with 30% margin)
    max_x, max_y = 1.3 * tan_fovx, 1.3 * tan_fovy
    x_proj = np.clip(x_cam / z_cam, -max_x, max_x) * z_cam
    y_proj = np.clip(y_cam / z_cam, -max_y, max_y) * z_cam

    inv_z = 1.0 / z_cam
    inv_z_sq = inv_z**2

    """
    The Jacobian captures the first-order derivative of the projection function. 
    It is a linear approximation of the projection at a specific point.
    J = [
        fx/z, 0,   -(fx)x/z²
        0,   fy/z, -(fy)y/z²
    ]
    """
    J = np.zeros((len(x_cam), 2, 3))  # [N, 2, 3]
    J[:, 0, 0] = focal_x * inv_z
    J[:, 0, 2] = -focal_x * x_proj * inv_z_sq
    J[:, 1, 1] = focal_y * inv_z
    J[:, 1, 2] = -focal_y * y_proj * inv_z_sq

    # Extract rotation part of view matrix
    W = view_matrix[:3, :3]  # [3, 3]
    T = J @ W  # [N, 2, 3]
    cov_2d = T @ cov_3d @ T.transpose(0, 2, 1)  # [N, 2, 2]

    # Apply low-pass filter to ensure minimum size of 0.3 pixels
    # This prevents Gaussians from becoming too small in screen space
    cov_2d[:, 0, 0] += 0.3
    cov_2d[:, 1, 1] += 0.3

    return cov_2d


class Gaussian:
    def __init__(self, sh_degree=3):
        self.max_sh_degree = sh_degree
        self._mean = np.empty((0, 3))
        self._features_dc = np.empty((0, 3, 1))
        self._features_rest = np.empty((0, 3, (sh_degree + 1) ** 2 - 1))
        self._scaling = np.empty((0, 3))
        self._rotation = np.empty((0, 4))
        self._opacity = np.empty((0, 1))

    @property
    def scalings(self):
        return np.exp(self._scaling)

    @property
    def rotations(self):
        return normalize(self._rotation)

    @property
    def positions(self):
        return self._mean

    @property
    def features(self):
        return np.concatenate([self._features_dc, self._features_rest], axis=-1)

    @property
    def opacities(self):
        return sigmoid(self._opacity)

    @property
    def covariances(self):
        return compute_covariance_3d(self.scalings, self.rotations)

    def compute_projected_covariance(self, camera):
        return batch_compute_projected_covariance(
            self.positions,
            self.covariances,
            camera.focal_x,
            camera.focal_y,
            camera.tan_x,
            camera.tan_y,
            camera.view_matrix,
        )

    def compute_depth(self, camera):
        return batch_project(points=self.positions, transform_matrix=camera.view_matrix)[:, 2]

    def load_ply(self, path):
        plydata = PlyData.read(path)
        xyz = np.column_stack(
            [
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ]
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[:, np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3

        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._mean = xyz.astype(np.float32)
        self._features_dc = features_dc.astype(np.float32)
        self._features_rest = features_extra.astype(np.float32)
        self._opacity = opacities.astype(np.float32)
        self._scaling = scales.astype(np.float32)
        self._rotation = rots.astype(np.float32)

        self.active_sh_degree = self.max_sh_degree

    def render(self, camera, indices=None):
        indices = indices or list(range(len(self.positions)))
        cov_2d = self.compute_projected_covariance(camera=camera)

        camera_direction = normalize(self.positions - camera.camera_center)
        colors = np.clip(eval_sh(self.max_sh_degree, self.features, camera_direction) + 0.5, 0.0, 1.0)

        bbox_cam, bbox_ndc = compute_projected_bboxes(
            cov_2d=cov_2d,
            positions=self.positions,
            transform_matrix=camera.transform_matrix,
            screen_height=camera.height,
            screen_width=camera.width,
        )

        return render_gaussian(
            depth=self.compute_depth(camera=camera)[indices],
            opacity=self.opacities[indices],
            color=colors[indices],
            conic=np.linalg.inv(cov_2d)[indices],
            bbox_cam=bbox_cam[indices],
            bbox_ndc=bbox_ndc[indices],
            screen_height=camera.height,
            screen_width=camera.width,
        )


def print_gaussian_stats(gs):
    print("\n" + "=" * 50)
    print(" " * 10 + "Gaussian Splatting Statistics")
    print("=" * 50)

    print(f"""
    🎯 Positions  : {gs.positions.shape}
    🔄 Rotations  : {gs.rotations.shape}
    📏 Scalings   : {gs.scalings.shape}
    💫 Opacities  : {gs.opacities.shape}
    🎨 Features   : {gs.features.shape}
    📊 Covariances: {gs.covariances.shape}
    """)

    print("-" * 50)
    print(f"Total Gaussians: {gs.positions.shape[0]:,}")
    print(f"SH Degree      : {gs.max_sh_degree}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--model_path", default="data/point_cloud.ply", type=str)
    parser.add_argument("--sh_degree", default=3, type=int)
    args = parser.parse_args()

    gs = Gaussian(sh_degree=args.sh_degree)
    gs.load_ply(args.model_path)
    print_gaussian_stats(gs)
