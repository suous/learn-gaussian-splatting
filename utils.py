import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def normalize(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def to_homogeneous(x):
    # Add a column of ones to the right of the matrix. [N, 3] -> [N, 4]
    return np.hstack([x, np.ones((x.shape[0], 1))])


def batch_project(points, transform_matrix):
    # Project points to clip space.
    transformed = to_homogeneous(points) @ transform_matrix.T
    # Perform perspective divide to get Normalized Device Coordinates (NDC).
    return transformed[..., :3] / transformed[..., 3:4]
