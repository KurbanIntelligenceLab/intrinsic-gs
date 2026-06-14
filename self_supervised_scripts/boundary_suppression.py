"""
Boundary suppression term B(i,j) for the affinity graph (proposal Sec. 4.3, Eq. 6).

    B(i,j) = sigmoid( α · g_depth(i,j) + β · g_rgb(i,j) − γ )
    W(i,j) ← W(i,j) · (1 − B(i,j))

g_depth and g_rgb are per-edge image-space boundary responses: Sobel gradient
magnitudes of rendered depth and GT RGB respectively, sampled along the image-
space line segment connecting the projections of Gaussians i and j, then
aggregated (max) across a set of keyframe views and (max) along the segment.

GT RGB is `view.original_image` (loaded by Scene from the dataset `rgb/` folder).
Depth is the per-pixel depth returned by the differentiable rasterizer.
"""

import math
import torch
import torch.nn.functional as F


# Sobel kernels (fixed, no learned params).
_SOBEL_X = torch.tensor([[-1., 0., 1.],
                         [-2., 0., 2.],
                         [-1., 0., 1.]])
_SOBEL_Y = torch.tensor([[-1., -2., -1.],
                         [ 0.,  0.,  0.],
                         [ 1.,  2.,  1.]])


def _gaussian_blur_2d(img_1hw, sigma):
    """Separable 2D Gaussian blur. img: [1, H, W] → [1, H, W]. sigma in pixels."""
    if sigma is None or sigma <= 0:
        return img_1hw
    radius = int(math.ceil(3.0 * float(sigma)))
    k = 2 * radius + 1
    x = torch.arange(k, device=img_1hw.device, dtype=img_1hw.dtype) - radius
    g = torch.exp(-0.5 * (x / float(sigma)) ** 2)
    g = g / g.sum()
    kx = g.view(1, 1, 1, k)
    ky = g.view(1, 1, k, 1)
    y = img_1hw.unsqueeze(0)                                # [1,1,H,W]
    y = F.conv2d(y, kx, padding=(0, radius))
    y = F.conv2d(y, ky, padding=(radius, 0))
    return y.squeeze(0)


def _sobel_magnitude(img_1hw, presmooth_sigma=0.0):
    """img: [1, H, W] → [H, W] gradient magnitude.

    If `presmooth_sigma > 0`, apply a separable Gaussian blur first. This
    suppresses high-frequency texture (e.g. woven mats, fabric weaves) so
    that Sobel responds to coarse object silhouettes rather than texture
    edges.
    """
    device = img_1hw.device
    if presmooth_sigma and presmooth_sigma > 0:
        img_1hw = _gaussian_blur_2d(img_1hw, presmooth_sigma)
    kx = _SOBEL_X.to(device).view(1, 1, 3, 3)
    ky = _SOBEL_Y.to(device).view(1, 1, 3, 3)
    x  = img_1hw.unsqueeze(0)          # [1,1,H,W]
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy).squeeze(0).squeeze(0)


def _rgb_to_luminance(rgb_3hw):
    r, g, b = rgb_3hw[0], rgb_3hw[1], rgb_3hw[2]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _normalize_by_percentile(t, q=0.95, eps=1e-6):
    """Divide by the q-quantile and clamp to [0,1]. Subsample for speed."""
    flat = t.flatten()
    n = flat.numel()
    if n > 200_000:
        idx = torch.randint(0, n, (200_000,), device=t.device)
        sample = flat[idx]
    else:
        sample = flat
    scale = torch.quantile(sample, q).clamp(min=eps)
    return (t / scale).clamp(0.0, 1.0)


# torch.quantile rejects tensors above an element-count cap (e.g. kNN with k=50
# gives O(N·k) edges; debug p95 over all visible edges can exceed that cap).
_QUANTILE_SAFE_MAX = 8_000_000


def _quantile_for_log(t, q=0.95):
    """Scalar quantile for logging; subsample when t is huge."""
    flat = t.flatten()
    n = flat.numel()
    if n == 0:
        return 0.0
    if n > _QUANTILE_SAFE_MAX:
        idx = torch.randint(0, n, (_QUANTILE_SAFE_MAX,), device=t.device)
        flat = flat[idx]
    return torch.quantile(flat.float(), q).item()


class BoundarySuppression:
    """
    Build B(i,j) for a given edge list using a set of keyframe views.

    Usage:
        bs = BoundarySuppression(views, pipe, background,
                                 deform_model=deform,  # optional
                                 n_views=12, n_samples=8,
                                 alpha_depth=5.0, beta_rgb=2.0, gamma=2.0,
                                 presmooth_sigma=0.0,  # >0 = blur before Sobel
                                                       #      to kill texture
                                                       #      (e.g. woven mat)
                                 edge_method=None)     # callable rgb_3hw→[H,W]
                                                       # for the RGB branch;
                                                       # default = inline Sobel
                                                       # on luminance.
        B = bs.compute(gaussians, edge_index_on_valid, valid_mask)  # [E]
    """

    def __init__(
        self,
        views,
        pipe,
        background,
        deform_model=None,
        is_6dof: bool = False,
        n_views: int = 12,
        n_samples: int = 8,
        alpha_depth: float = 5.0,
        beta_rgb: float = 2.0,
        gamma: float = 2.0,
        edge_chunk: int = 500_000,
        clamp_max: float = 0.99,
        use_gt_rgb: bool = True,
        presmooth_sigma: float = 0.0,
        edge_method=None,
    ):
        if n_views < 1:
            raise ValueError("n_views must be ≥ 1")
        stride = max(1, len(views) // n_views)
        self.views        = list(views)[::stride][:n_views]
        self.pipe         = pipe
        self.background   = background
        self.deform_model = deform_model
        self.is_6dof      = is_6dof
        self.n_samples    = n_samples
        self.alpha_depth  = alpha_depth
        self.beta_rgb     = beta_rgb
        self.gamma        = gamma
        self.edge_chunk   = edge_chunk
        self.clamp_max    = clamp_max
        self.use_gt_rgb   = use_gt_rgb
        self.presmooth_sigma = float(presmooth_sigma)
        # rgb_edge.SobelEdge / PiDiNetEdge / any callable rgb_3hw→[H,W].
        # Left as None to keep the original inline-Sobel path for callers
        # that don't construct an edge_method.
        self.edge_method  = edge_method

    # ------------------------------------------------------------------
    #  Per-view projection & boundary maps
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _project(self, xyz_world, view):
        """
        Project world-space points into pixel coords for `view`.

        Returns:
            px  [N,2] pixel (x,y) in image coords ([0,W], [0,H])
            vis [N]   bool, in-frustum and in front of camera
        """
        N = xyz_world.shape[0]
        ones = torch.ones(N, 1, device=xyz_world.device, dtype=xyz_world.dtype)
        pts_h = torch.cat([xyz_world, ones], dim=-1)                    # [N,4]

        # Camera-space z (for in-front test).  world_view_transform is already
        # transposed in this codebase, so we post-multiply (row-vector convention).
        cam   = pts_h @ view.world_view_transform.to(xyz_world.device)  # [N,4]
        z_cam = cam[..., 2]

        clip  = pts_h @ view.full_proj_transform.to(xyz_world.device)   # [N,4]
        w     = clip[..., 3:4].clamp(min=1e-6)
        ndc   = clip[..., :2] / w                                       # [-1,1] for in-frame
        W_img = view.image_width
        H_img = view.image_height
        px = (ndc + 1.0) * 0.5 * torch.tensor([W_img, H_img],
                                              device=xyz_world.device,
                                              dtype=xyz_world.dtype)
        vis = (z_cam > 0) & \
              (px[:, 0] >= 0) & (px[:, 0] < W_img) & \
              (px[:, 1] >= 0) & (px[:, 1] < H_img)
        return px, vis

    @torch.no_grad()
    def _deform_for(self, gaussians, view):
        """Query the deformation MLP at this view's time, or return zeros."""
        xyz = gaussians.get_xyz
        N   = xyz.shape[0]
        if self.deform_model is None:
            device = xyz.device
            z3 = torch.zeros(N, 3, device=device, dtype=xyz.dtype)
            z4 = torch.zeros(N, 4, device=device, dtype=xyz.dtype)
            return z3, z4, z3
        fid = view.fid
        time_input = fid.unsqueeze(0).expand(N, -1).to(xyz.device)
        return self.deform_model.step(xyz.detach(), time_input)

    @torch.no_grad()
    def _boundary_maps(self, gaussians, view):
        """Return normalized depth-edge map and rgb-edge map, both [H, W]."""
        # Local import to avoid a heavy renderer import at module load.
        from gaussian_renderer import render

        d_xyz, d_rot, d_scale = self._deform_for(gaussians, view)
        out = render(view, gaussians, self.pipe, self.background,
                     d_xyz, d_rot, d_scale, is_6dof=self.is_6dof)
        depth = out["depth"]                       # [1,H,W] or [H,W]
        if depth.dim() == 2:
            depth = depth.unsqueeze(0)

        if self.use_gt_rgb and view.original_image is not None:
            rgb = view.original_image.to(depth.device).float()
        else:
            rgb = out["render"].clamp(0, 1)

        # Depth branch is always Sobel (depth is smooth across textured
        # surfaces — no learned detector needed).
        g_d = _sobel_magnitude(depth, presmooth_sigma=self.presmooth_sigma)

        # RGB branch is pluggable. Default (edge_method=None) preserves the
        # original inline-Sobel path bit-for-bit.
        if self.edge_method is not None:
            g_c = self.edge_method(rgb)
        else:
            lum = _rgb_to_luminance(rgb).unsqueeze(0)  # [1,H,W]
            g_c = _sobel_magnitude(lum, presmooth_sigma=self.presmooth_sigma)

        g_d = _normalize_by_percentile(g_d, q=0.95)
        g_c = _normalize_by_percentile(g_c, q=0.95)

        # Also return the world-space positions at this time so projection
        # is consistent with the rendered depth.
        xyz_world = gaussians.get_xyz + d_xyz
        return g_d, g_c, xyz_world

    # ------------------------------------------------------------------
    #  Edge-level line-sample
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _sample_segment_max(self, edge_map_hw, px_i, px_j, both_vis):
        """
        Max of `edge_map_hw` along the image-space segment p_i → p_j, for every
        edge.  Edges where `both_vis` is False get 0.

        edge_map_hw : [H, W]
        px_i, px_j  : [E, 2] pixel coordinates (x, y)
        both_vis    : [E] bool
        Returns     : [E] float in [0, 1]
        """
        E = px_i.shape[0]
        K = self.n_samples
        H, W = edge_map_hw.shape

        out = torch.zeros(E, device=edge_map_hw.device, dtype=edge_map_hw.dtype)
        if E == 0 or not both_vis.any():
            return out

        # grid_sample needs normalized coords in [-1,1] where (-1,-1) is the
        # top-left of the tensor. We use align_corners=True so that pixel
        # centers 0..(W-1) map linearly to -1..1.
        t = torch.linspace(0.0, 1.0, K, device=edge_map_hw.device).view(1, K, 1)
        # [E, K, 2] samples in pixel space
        seg = px_i.unsqueeze(1) * (1 - t) + px_j.unsqueeze(1) * t

        # Normalize to [-1, 1]
        seg_nx = 2.0 * seg[..., 0] / max(W - 1, 1) - 1.0
        seg_ny = 2.0 * seg[..., 1] / max(H - 1, 1) - 1.0
        grid = torch.stack([seg_nx, seg_ny], dim=-1)          # [E, K, 2]

        # grid_sample input: [N, C, H, W], grid: [N, Hg, Wg, 2] → [N, C, Hg, Wg]
        inp  = edge_map_hw.unsqueeze(0).unsqueeze(0)           # [1,1,H,W]

        # Chunk to bound memory
        CHUNK = self.edge_chunk
        for s in range(0, E, CHUNK):
            e = min(s + CHUNK, E)
            g = grid[s:e].unsqueeze(0)                         # [1, Ec, K, 2]
            sampled = F.grid_sample(inp, g,
                                    mode='bilinear',
                                    padding_mode='zeros',
                                    align_corners=True)        # [1,1,Ec,K]
            out[s:e] = sampled.squeeze(0).squeeze(0).max(dim=1).values

        out = out * both_vis.to(out.dtype)
        return out

    # ------------------------------------------------------------------
    #  Public entry point
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute(self, gaussians, edge_index, valid_mask):
        """
        Compute B for each edge.

        Args:
            gaussians    : GaussianModel (all N Gaussians, pre-filter)
            edge_index   : [2, E] indices into the VALID subset (0..N_valid-1)
            valid_mask   : [N] bool, which original Gaussians passed the
                           opacity filter (same one AffinityGraph used)
        Returns:
            B            : [E] in [0, clamp_max]
        """
        device = gaussians.get_xyz.device
        i_idx = edge_index[0]
        j_idx = edge_index[1]
        E = i_idx.shape[0]

        g_d_max = torch.zeros(E, device=device)
        g_c_max = torch.zeros(E, device=device)
        vis_any = torch.zeros(E, dtype=torch.bool, device=device)

        valid_mask = valid_mask.to(device)

        for v_idx, view in enumerate(self.views):
            g_d, g_c, xyz_world = self._boundary_maps(gaussians, view)

            # Project ALL Gaussians, then subset to valid and pick by edge idx.
            px_all, vis_all = self._project(xyz_world, view)
            px_valid  = px_all[valid_mask]      # [N_valid, 2]
            vis_valid = vis_all[valid_mask]     # [N_valid]

            px_i = px_valid[i_idx]
            px_j = px_valid[j_idx]
            both = vis_valid[i_idx] & vis_valid[j_idx]

            gd_edge = self._sample_segment_max(g_d, px_i, px_j, both)
            gc_edge = self._sample_segment_max(g_c, px_i, px_j, both)

            g_d_max = torch.maximum(g_d_max, gd_edge)
            g_c_max = torch.maximum(g_c_max, gc_edge)
            vis_any = vis_any | both

            n_vis = both.sum().item()
            if n_vis:
                gd_p95 = _quantile_for_log(gd_edge[both])
                gc_p95 = _quantile_for_log(gc_edge[both])
            else:
                gd_p95 = gc_p95 = 0.0
            print(f"  [boundary] view {v_idx+1}/{len(self.views)} "
                  f"({view.image_name}):  visible edges={n_vis:,} "
                  f"({100*n_vis/max(E,1):.1f}%)  "
                  f"gd p95={gd_p95:.3f}  "
                  f"gc p95={gc_p95:.3f}")

        logits = self.alpha_depth * g_d_max + self.beta_rgb * g_c_max - self.gamma
        B = torch.sigmoid(logits)
        # Edges never visible in any keyframe: defer to Ageo·Amotion → B = 0.
        B = B * vis_any.to(B.dtype)
        B = B.clamp(max=self.clamp_max)

        n_seen = vis_any.sum().item()
        print(f"  [boundary] summary:  edges seen in ≥1 view={n_seen:,}/{E:,} "
              f"({100*n_seen/max(E,1):.1f}%)  "
              f"B mean={B.mean().item():.4f}  "
              f"B>0.5={100*(B>0.5).float().mean().item():.2f}%")
        return B
