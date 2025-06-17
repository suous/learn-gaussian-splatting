import argparse

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from camera import Camera
from gaussian import Gaussian
from dataset_readers import readColmapSceneInfo


def main(args):
    scene_info = readColmapSceneInfo(args.colmap_path)
    scene = scene_info.train_cameras[args.image_number]
    image = Image.open(scene.image_path)
    camera = Camera(image=image, rotation=scene.R.T, translation=scene.T, fov_x=scene.FovX, fov_y=scene.FovY)
    gaussian = Gaussian()
    gaussian.load_ply(args.ply_path)
    num_gaussians = len(gaussian.positions)
    print(f"total number of gaussians: {num_gaussians}")

    render_size = min(args.render_size or num_gaussians, num_gaussians)
    indices = range(0, render_size, args.step_size)
    print(f"number of gaussians to render: {len(indices)}")

    bitmap = gaussian.render(camera, indices=indices)
    print(f"mean absolute error: {np.mean(np.abs(bitmap - camera.original_image)):.4f}")

    if args.generate_animation:
        save_image(bitmap, f"data/images/{args.render_size:05d}.png")
        return

    fig, axes = plt.subplots(nrows=2, sharex=True, frameon=False)
    for ax, img in zip(axes, [camera.original_image, bitmap]):
        ax.imshow(img)
        ax.set_axis_off()
    fig.tight_layout()
    plt.show()


def save_image(bitmap, output_path):
    if np.issubdtype(bitmap.dtype, np.floating):
        bitmap = (np.clip(bitmap, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(bitmap).save(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--colmap_path", type=str, help="Path to the colmap directory", default="data/colmap")
    parser.add_argument("--ply_path", type=str, help="Path to the ply file", default="data/point_cloud.ply")
    parser.add_argument("--image_number", type=int, help="Which image to render", default=0)
    parser.add_argument("--render_size", type=int, help="Render gaussian size", default=None)
    parser.add_argument("--step_size", type=int, help="Render step size", default=1)
    parser.add_argument("--generate_animation", action="store_true", help="Generate animation")
    args = parser.parse_args()
    if args.generate_animation:
        for render_size in np.logspace(1, np.log10(40000), num=20).astype(int):
            args.render_size = render_size
            main(args)
    else:
        main(args)
