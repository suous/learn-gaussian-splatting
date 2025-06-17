import math
import numpy as np


def build_world_view_matrix(rotation, translation):
    Rt = np.eye(4, dtype=np.float32)
    Rt[:3, :3] = rotation
    Rt[:3, 3] = translation
    return Rt


def build_projection_matrix(znear, zfar, tan_x, tan_y):
    # fmt: off
    return np.array([
        [1/tan_x, 0,       0,          0          ],
        [0,       1/tan_y, 0,          0          ],
        [0,       0,       znear+zfar, -znear*zfar],
        [0,       0,       1,          0          ]
    ], dtype=np.float32)
    # fmt: on


class Camera:
    def __init__(self, image, rotation, translation, fov_x, fov_y):
        self.zfar, self.znear = 100.0, 0.01
        self.fov_x, self.fov_y = fov_x, fov_y
        self.rotation = rotation
        self.translation = translation
        self.original_image = np.clip(np.asarray(image, dtype=np.float32) / 255.0, 0.0, 1.0)
        self.height, self.width = self.original_image.shape[:2]

        self.tan_x, self.tan_y = math.tan(fov_x * 0.5), math.tan(fov_y * 0.5)
        self.focal_x, self.focal_y = self.width / (2 * self.tan_x), self.height / (2 * self.tan_y)

        self.view_matrix = build_world_view_matrix(rotation, translation)
        self.projection_matrix = build_projection_matrix(self.znear, self.zfar, self.tan_x, self.tan_y)
        self.transform_matrix = self.projection_matrix @ self.view_matrix
        self.camera_center = np.linalg.inv(self.view_matrix)[:3, 3]
