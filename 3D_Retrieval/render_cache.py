# render_cache.py
import os
import yaml
from glob import glob

import numpy as np
from PIL import Image
from tqdm import tqdm
import trimesh
import pyrender


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def render_views(mesh_path: str,
                 num_views: int = 8,
                 image_size: int = 128):
    """
    Render `num_views` views of a single mesh using pyrender off-screen rendering.

    We place virtual cameras evenly around the object on a sphere and
    project the shaded mesh into 2D images.

    Returns a list of PIL Images (RGB).
    """
    mesh = trimesh.load(mesh_path)
    mesh.apply_translation(-mesh.centroid)  # center the object

    scene = pyrender.Scene()
    mesh_node = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    scene.add(mesh_node)

    # Simple distant directional light.
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(light)

    # Spherical camera placement.
    radius = 2.5 * max(mesh.extents)
    angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)

    cam = pyrender.PerspectiveCamera(yfov=np.pi / 4.0)
    r = pyrender.OffscreenRenderer(viewport_width=image_size,
                                   viewport_height=image_size)

    images = []
    for theta in angles:
        camera_pose = np.array([
            [np.cos(theta), 0, np.sin(theta), radius * np.cos(theta)],
            [0,             1, 0,             0],
            [-np.sin(theta), 0, np.cos(theta), radius * np.sin(theta)],
            [0,             0, 0,             1]
        ])
        scene.add(cam, pose=camera_pose)
        color, _ = r.render(scene)
        scene.remove_node(scene.get_nodes(name=None, type=pyrender.Camera)[0])

        img = Image.fromarray(color).convert("RGB")
        images.append(img)

    r.delete()
    return images


def cache_views_for_meshes(mesh_paths,
                           cache_root: str,
                           split_name: str,
                           num_views: int,
                           image_size: int):
    """
    For each .off mesh in `mesh_paths`, render multi–view PNGs and save them
    under:
        cache_root/split_name/class_name/obj_name/view_XX.png
    """
    for mesh_path in tqdm(mesh_paths, desc=f"Caching {split_name}"):

        class_name = os.path.basename(os.path.dirname(mesh_path))     # e.g. 'chair'
        obj_name = os.path.splitext(os.path.basename(mesh_path))[0]   # e.g. 'chair_0001'

        save_dir = os.path.join(cache_root, split_name, class_name, obj_name)
        os.makedirs(save_dir, exist_ok=True)

        # If we already cached enough views, we skip this object.
        png_files = [f for f in os.listdir(save_dir) if f.endswith(".png")]
        if len(png_files) >= num_views:
            continue

        try:
            images = render_views(mesh_path, num_views=num_views,
                                  image_size=image_size)
        except Exception as e:
            print("Failed to render:", mesh_path, "->", e)
            continue

        for i, img in enumerate(images):
            out_path = os.path.join(save_dir, f"view_{i:02d}.png")
            img.save(out_path)


def main():
    cfg = load_config("configs/config.yaml")

    data_root = cfg["data_root"]
    cache_root = cfg["cache_root"]
    num_views = int(cfg["num_views"])
    img_size = int(cfg["img_size"])

    # ModelNet10 layout: data_root/class/{train,test}/*.off
    train_paths = sorted(glob(os.path.join(data_root, "*", "train", "*.off")))
    test_paths = sorted(glob(os.path.join(data_root, "*", "test", "*.off")))

    os.makedirs(cache_root, exist_ok=True)

    cache_views_for_meshes(train_paths, cache_root, "train",
                           num_views=num_views, image_size=img_size)
    cache_views_for_meshes(test_paths, cache_root, "test",
                           num_views=num_views, image_size=img_size)


if __name__ == "__main__":
    main()
