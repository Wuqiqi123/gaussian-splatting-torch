import torch
import torch.nn as nn
import math
from einops import reduce

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def homogeneous(points):
    """
    homogeneous points
    :param points: [..., 3]
    """
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device=r.device)

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R



def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device=s.device)
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device=L.device)
    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    return strip_lowerdiag(sym)



def build_covariance_3d(s, r):
    L = build_scaling_rotation(s, r)
    actual_covariance = L @ L.transpose(1, 2)
    return actual_covariance
    # symm = strip_symmetric(actual_covariance)
    # return symm



def build_covariance_2d(
    mean3d, cov3d, viewmatrix, fov_x, fov_y, focal_x, focal_y
):
    # The following models the steps outlined by equations 29
	# and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	# Additionally considers aspect / scaling of viewport.
	# Transposes used to account for row-/column-major conventions.
    tan_fovx = math.tan(fov_x * 0.5)
    tan_fovy = math.tan(fov_y * 0.5)
    t = (mean3d @ viewmatrix[:3,:3]) + viewmatrix[-1:,:3]

    # truncate the influences of gaussians far outside the frustum.
    tx = (t[..., 0] / t[..., 2]).clip(min=-tan_fovx*1.3, max=tan_fovx*1.3) * t[..., 2]
    ty = (t[..., 1] / t[..., 2]).clip(min=-tan_fovy*1.3, max=tan_fovy*1.3) * t[..., 2]
    tz = t[..., 2]

    # Eq.29 locally affine transform 
    # perspective transform is not affine so we approximate with first-order taylor expansion
    # notice that we multiply by the intrinsic so that the variance is at the sceen space
    J = torch.zeros(mean3d.shape[0], 3, 3).to(mean3d)
    J[..., 0, 0] = 1 / tz * focal_x
    J[..., 0, 2] = -tx / (tz * tz) * focal_x
    J[..., 1, 1] = 1 / tz * focal_y
    J[..., 1, 2] = -ty / (tz * tz) * focal_y
    # J[..., 2, 0] = tx / t.norm(dim=-1) # discard
    # J[..., 2, 1] = ty / t.norm(dim=-1) # discard
    # J[..., 2, 2] = tz / t.norm(dim=-1) # discard
    W = viewmatrix[:3,:3].T # transpose to correct viewmatrix
    cov2d = J @ W @ cov3d @ W.T @ J.permute(0,2,1)
    
    # add low pass filter here according to E.q. 32
    filter = torch.eye(2,2).to(cov2d) * 0.3
    return cov2d[:, :2, :2] + filter[None]


def projection_ndc(points, viewmatrix, projmatrix):
    points_o = homogeneous(points) # object space
    points_h = points_o @ viewmatrix @ projmatrix # screen space # RHS
    p_w = 1.0 / (points_h[..., -1:] + 0.000001)
    p_proj = points_h * p_w
    p_view = points_o @ viewmatrix
    in_mask = p_view[..., 2] >= 0.2
    return p_proj, p_view, in_mask


@torch.no_grad()
def get_radius(cov2d):
    det = cov2d[:, 0, 0] * cov2d[:,1,1] - cov2d[:, 0, 1] * cov2d[:,1,0]
    mid = 0.5 * (cov2d[:, 0,0] + cov2d[:,1,1])
    lambda1 = mid + torch.sqrt((mid**2-det).clip(min=0.1))
    lambda2 = mid - torch.sqrt((mid**2-det).clip(min=0.1))
    return 3.0 * torch.sqrt(torch.max(lambda1, lambda2)).ceil()

@torch.no_grad()
def get_rect(pix_coord, radii, width, height):
    rect_min = (pix_coord - radii[:,None])
    rect_max = (pix_coord + radii[:,None])
    rect_min[..., 0] = rect_min[..., 0].clip(0, width - 1.0)
    rect_min[..., 1] = rect_min[..., 1].clip(0, height - 1.0)
    rect_max[..., 0] = rect_max[..., 0].clip(0, width - 1.0)
    rect_max[..., 1] = rect_max[..., 1].clip(0, height - 1.0)
    return rect_min, rect_max


class GaussRenderer(nn.Module):
    """
    A gaussian splatting renderer

    >>> gaussModel = GaussModel.create_from_pcd(pts)
    >>> gaussRender = GaussRenderer()
    >>> out = gaussRender(pc=gaussModel, camera=camera)
    """

    def __init__(self, white_bkgd=True, **kwargs):
        super(GaussRenderer, self).__init__()
        self.white_bkgd = white_bkgd

    def build_color(self, means3D, camera, pc):
        rays_d = means3D - camera.camera_center
        return pc.color_net(means3D, rays_d)
    
    def render(self, camera, means2D, cov2d, color, opacity, depths):
        radii = get_radius(cov2d)
        rect = get_rect(means2D, radii, width=camera.image_width, height=camera.image_height)

        device = means2D.device
        H, W = camera.image_height, camera.image_width
        # pixel coordinate grid: pix_coord[i,j] = [x=j, y=i]
        pix_coord = torch.stack(
            torch.meshgrid(torch.arange(W, device=device), torch.arange(H, device=device), indexing='xy'),
            dim=-1,
        )  # H x W x 2

        render_color = torch.ones(H, W, 3, device=device)
        render_depth = torch.zeros(H, W, 1, device=device)
        render_alpha = torch.zeros(H, W, 1, device=device)

        TILE_SIZE = 64
        for h in range(0, H, TILE_SIZE):
            for w in range(0, W, TILE_SIZE):
                # check if any gaussian bounding rect overlaps this tile
                over_tl = rect[0][..., 0].clip(min=w), rect[0][..., 1].clip(min=h)
                over_br = rect[1][..., 0].clip(max=w+TILE_SIZE-1), rect[1][..., 1].clip(max=h+TILE_SIZE-1)
                in_mask = (over_br[0] > over_tl[0]) & (over_br[1] > over_tl[1])

                if not in_mask.sum() > 0:
                    continue

                tile_h = min(TILE_SIZE, H - h)
                tile_w = min(TILE_SIZE, W - w)
                tile_coord = pix_coord[h:h+tile_h, w:w+tile_w].flatten(0, -2)  # (tile_h*tile_w, 2)

                sorted_depths, index = torch.sort(depths[in_mask])
                sorted_means2D  = means2D[in_mask][index]
                sorted_cov2d    = cov2d[in_mask][index]       # P 2 2
                sorted_conic    = sorted_cov2d.inverse()       # inverse of covariance
                sorted_opacity  = opacity[in_mask][index]
                sorted_color    = color[in_mask][index]

                dx = tile_coord[:, None, :] - sorted_means2D[None, :]  # (tile_pixels, P, 2)

                gauss_weight = torch.exp(-0.5 * (
                    dx[:, :, 0]**2 * sorted_conic[:, 0, 0]
                    + dx[:, :, 1]**2 * sorted_conic[:, 1, 1]
                    + dx[:, :, 0] * dx[:, :, 1] * sorted_conic[:, 0, 1]
                    + dx[:, :, 0] * dx[:, :, 1] * sorted_conic[:, 1, 0]
                ))

                alpha     = (gauss_weight[..., None] * sorted_opacity[None]).clip(max=0.99)  # (pixels, P, 1)
                T         = torch.cat([torch.ones_like(alpha[:, :1]), 1 - alpha[:, :-1]], dim=1).cumprod(dim=1)
                acc_alpha = (alpha * T).sum(dim=1)
                tile_color = (T * alpha * sorted_color[None]).sum(dim=1) + (1 - acc_alpha) * (1 if self.white_bkgd else 0)
                tile_depth = ((T * alpha) * sorted_depths[None, :, None]).sum(dim=1)

                render_color[h:h+tile_h, w:w+tile_w] = tile_color.reshape(tile_h, tile_w, -1)
                render_depth[h:h+tile_h, w:w+tile_w] = tile_depth.reshape(tile_h, tile_w, -1)
                render_alpha[h:h+tile_h, w:w+tile_w] = acc_alpha.reshape(tile_h, tile_w, -1)

        return {
            "render": render_color,
            "depth": render_depth,
            "alpha": render_alpha,
            "visibility_filter": radii > 0,
            "radii": radii,
        }


    def forward(self, camera, pc, **kwargs):
        means3D   = pc.get_xyz
        opacity   = pc.get_opacity
        scales    = pc.get_scaling
        rotations = pc.get_rotation

        # --- projection & visibility filter ---
        mean_ndc, mean_view, in_mask = projection_ndc(
            means3D,
            viewmatrix=camera.world_view_transform,
            projmatrix=camera.projection_matrix,
        )
        mean_ndc  = mean_ndc[in_mask]
        mean_view = mean_view[in_mask]
        depths    = mean_view[:, 2]

        # keep only visible gaussians for all subsequent ops
        means3D_vis   = means3D[in_mask]
        scales_vis    = scales[in_mask]
        rotations_vis = rotations[in_mask]
        opacity_vis   = opacity[in_mask]

        color = self.build_color(means3D=means3D_vis, camera=camera, pc=pc)

        cov3d = build_covariance_3d(scales_vis, rotations_vis)
        cov2d = build_covariance_2d(
            mean3d=means3D_vis,
            cov3d=cov3d,
            viewmatrix=camera.world_view_transform,
            fov_x=camera.FoVx,
            fov_y=camera.FoVy,
            focal_x=camera.focal_x,
            focal_y=camera.focal_y,
        )

        mean_coord_x = ((mean_ndc[..., 0] + 1) * camera.image_width  - 1.0) * 0.5
        mean_coord_y = ((mean_ndc[..., 1] + 1) * camera.image_height - 1.0) * 0.5
        means2D = torch.stack([mean_coord_x, mean_coord_y], dim=-1)

        return self.render(
            camera=camera,
            means2D=means2D,
            cov2d=cov2d,
            color=color,
            opacity=opacity_vis,
            depths=depths,
        )
