"""
COLMAP binary format reader.

Reads cameras.bin, images.bin, points3D.bin from a standard COLMAP sparse model
and returns data in the same tensor format used by the rest of the codebase.
"""

import os
import struct
import collections
import numpy as np
import torch
import torch.nn.functional as F
import imageio

# ── COLMAP data structures ────────────────────────────────────────────────────

CameraModel = collections.namedtuple('CameraModel', ['id', 'model', 'width', 'height', 'params'])
ImageModel  = collections.namedtuple('ImageModel',  ['id', 'qvec', 'tvec', 'camera_id', 'name'])
Point3D     = collections.namedtuple('Point3D',     ['id', 'xyz', 'rgb', 'error'])

# model_id → (name, num_params)
CAMERA_MODEL_PARAMS = {
    0: ('SIMPLE_PINHOLE', 3),   # f, cx, cy
    1: ('PINHOLE',        4),   # fx, fy, cx, cy
    2: ('SIMPLE_RADIAL',  4),   # f, cx, cy, k1
    3: ('RADIAL',         5),   # f, cx, cy, k1, k2
    4: ('OPENCV',         8),   # fx, fy, cx, cy, k1, k2, p1, p2
}


# ── binary readers ────────────────────────────────────────────────────────────

def read_cameras_binary(path):
    cameras = {}
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        for _ in range(n):
            cam_id   = struct.unpack('<I', f.read(4))[0]   # uint32
            model_id = struct.unpack('<i', f.read(4))[0]   # int32
            w        = struct.unpack('<Q', f.read(8))[0]   # uint64
            h        = struct.unpack('<Q', f.read(8))[0]   # uint64
            _, n_params = CAMERA_MODEL_PARAMS.get(model_id, ('UNKNOWN', 0))
            params = np.array(struct.unpack('<%sd' % n_params, f.read(8 * n_params)))
            cameras[cam_id] = CameraModel(id=cam_id, model=model_id, width=w, height=h, params=params)
    return cameras


def read_images_binary(path):
    images = {}
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        for _ in range(n):
            img_id  = struct.unpack('<I', f.read(4))[0]             # uint32
            qvec    = np.array(struct.unpack('<4d', f.read(32)))    # qw qx qy qz
            tvec    = np.array(struct.unpack('<3d', f.read(24)))
            cam_id  = struct.unpack('<I', f.read(4))[0]             # uint32
            name    = b''
            while True:
                c = f.read(1)
                if c == b'\x00':
                    break
                name += c
            name = name.decode('utf-8')
            n_pts2d = struct.unpack('<Q', f.read(8))[0]
            f.read(n_pts2d * 24)                                    # skip 2D keypoints
            images[img_id] = ImageModel(id=img_id, qvec=qvec, tvec=tvec,
                                        camera_id=cam_id, name=name)
    return images


def read_points3d_binary(path):
    points = {}
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        for _ in range(n):
            pt_id   = struct.unpack('<Q', f.read(8))[0]   # uint64
            xyz     = np.array(struct.unpack('<3d', f.read(24)))
            rgb     = np.array(struct.unpack('<3B', f.read(3)))
            error   = struct.unpack('<d', f.read(8))[0]
            n_track = struct.unpack('<Q', f.read(8))[0]
            f.read(n_track * 8)                                    # skip track
            points[pt_id] = Point3D(id=pt_id, xyz=xyz, rgb=rgb, error=error)
    return points


# ── coordinate conversion ─────────────────────────────────────────────────────

def qvec2rotmat(qvec):
    """COLMAP quaternion (qw, qx, qy, qz) → 3×3 rotation matrix."""
    w, x, y, z = qvec / np.linalg.norm(qvec)
    return np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [    2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
        [    2*x*z - 2*y*w,     2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y],
    ], dtype=np.float32)


def colmap_image_to_c2w(image):
    """COLMAP extrinsics (R,t: p_cam = R @ p_world + t) → c2w 4×4."""
    R = qvec2rotmat(image.qvec)
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :3] = R
    w2c[:3,  3] = image.tvec
    return np.linalg.inv(w2c)


def camera_to_intrinsic(cam):
    """Build 3×3 intrinsic matrix from COLMAP CameraModel."""
    K = np.eye(3, dtype=np.float32)
    p = cam.params
    if cam.model in (0, 2, 3):      # SIMPLE_PINHOLE / SIMPLE_RADIAL / RADIAL
        K[0, 0] = K[1, 1] = p[0]
        K[0, 2] = p[1];  K[1, 2] = p[2]
    elif cam.model in (1, 4):       # PINHOLE / OPENCV
        K[0, 0] = p[0];  K[1, 1] = p[1]
        K[0, 2] = p[2];  K[1, 2] = p[3]
    return K


# ── high-level loader ─────────────────────────────────────────────────────────

def read_colmap(folder, resize_factor=1.0, max_images=None):
    """
    Load a COLMAP dataset.

    Expected layout:
        <folder>/images/         RGB images
        <folder>/sparse/0/       cameras.bin, images.bin, points3D.bin

    Returns:
        dict with keys:
            rgb    : (N, H, W, 3) float32 [0,1]
            camera : (N, 34)      float32  [H, W, K4x4_flat(16), c2w_flat(16)]
            points : PointCloud   COLMAP sparse 3D points for Gaussian init
    """
    from gaussian_splatting.utils.point_utils import PointCloud

    sparse_dir = os.path.join(folder, 'sparse', '0')
    images_dir = os.path.join(folder, 'images')

    cameras_map = read_cameras_binary(os.path.join(sparse_dir, 'cameras.bin'))
    images_map  = read_images_binary(os.path.join(sparse_dir,  'images.bin'))
    pts3d_map   = read_points3d_binary(os.path.join(sparse_dir, 'points3D.bin'))

    sorted_imgs = sorted(images_map.values(), key=lambda im: im.name)
    if max_images is not None:
        sorted_imgs = sorted_imgs[:max_images]

    src_rgbs    = []
    src_cameras = []

    for colmap_img in sorted_imgs:
        img_path = os.path.join(images_dir, colmap_img.name)
        if not os.path.exists(img_path):
            continue

        cam = cameras_map[colmap_img.camera_id]
        K   = camera_to_intrinsic(cam)           # 3×3
        c2w = colmap_image_to_c2w(colmap_img)    # 4×4

        rgb = imageio.imread(img_path).astype(np.float32) / 255.0
        if rgb.ndim == 2:
            rgb = np.stack([rgb] * 3, axis=-1)
        rgb = rgb[..., :3]
        H, W = rgb.shape[:2]

        if resize_factor != 1.0:
            nH, nW = int(H * resize_factor), int(W * resize_factor)
            t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
            t = F.interpolate(t, size=(nH, nW), mode='bilinear', align_corners=False)
            rgb = t.squeeze(0).permute(1, 2, 0).numpy()
            K[0, 0] *= resize_factor   # fx
            K[1, 1] *= resize_factor   # fy
            K[0, 2] *= resize_factor   # cx
            K[1, 2] *= resize_factor   # cy
            H, W = nH, nW

        K4 = np.eye(4, dtype=np.float32)
        K4[:3, :3] = K
        cam_vec = np.concatenate([[H, W], K4.flatten(), c2w.flatten()]).astype(np.float32)

        src_rgbs.append(torch.from_numpy(rgb))
        src_cameras.append(torch.from_numpy(cam_vec))

    print(f"Loaded {len(src_rgbs)} images  |  {len(pts3d_map)} sparse points")

    coords = np.array([p.xyz for p in pts3d_map.values()], dtype=np.float32)
    colors = np.array([p.rgb for p in pts3d_map.values()], dtype=np.float32)
    pcd = PointCloud(
        coords=coords,
        channels={'R': colors[:, 0], 'G': colors[:, 1], 'B': colors[:, 2]},
    )

    return {
        'rgb':    torch.stack(src_rgbs,    dim=0),   # (N, H, W, 3)
        'camera': torch.stack(src_cameras, dim=0),   # (N, 34)
        'points': pcd,
    }
