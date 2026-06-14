import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch_cluster import knn as torch_knn

from utils.general_utils import build_rotation


class AffinityGraph:
    """
    Builds a sparse Gaussian affinity graph from frozen 4DGS parameters.

    Ageo-only mode (Phase 1):
        W(i,j) = (Acolor · Aorient · Ascale) ^ (1/3)

    Fused mode (Phase 2):
        W(i,j) = Ageo · fuse(Amotion)   where fuse depends on pair type:
          both-static  → 1.0            (delegate to geometric signals)
          both-moving  → Amotion        (pure product, trust motion)
          one-static   → floor + (1-floor)·Amotion   (uncertain, soft gate)

    Amotion = Atraj · Arot  (Eq. 2 from proposal)
        Atraj: vectorial centered cosine similarity of displacement trajectories
               over T time steps (Pearson correlation of 3D trajectories, Eq. 3)
        Arot:  mean |Δq(t)_i · Δq(t)_j| over T steps, where Δq(t) is the
               additive rotation delta from the deformation MLP
    """

    def __init__(
        self,
        gaussians,
        k: int = 20,
        opacity_thresh: float = 0.1,
        use_geometry: bool = True,
        sigma_pos: float = 0.1,
        sigma_color: float = 0.3,
        sigma_scale: float = 1.0,
        power: float = 1.0,
        deform_model=None,
        n_time_steps: int = 20,
        static_motion_thresh: float = 1e-3,
        motion_floor: float = 0.2,
        boundary=None,
    ):
        """
        Args:
            gaussians:             GaussianModel (Stage 1 checkpoint, frozen)
            k:                     number of nearest neighbors per Gaussian
            opacity_thresh:        Gaussians below this opacity are excluded
            use_geometry:          if False, use unit edge weights before
                                   optional motion/boundary terms. The graph
                                   topology still comes from spatial kNN.
            sigma_pos:             bandwidth for spatial proximity kernel
            sigma_color:           bandwidth for color similarity kernel
            sigma_scale:           bandwidth for scale ratio kernel
            power:                 sharpening exponent applied to W after geometric mean
            deform_model:          DeformModel instance; if None, Amotion is skipped
            n_time_steps:          T — number of uniformly sampled time steps in [0,1]
            static_motion_thresh:  threshold on RMS trajectory norm r[n];
                                   r[n] < thresh ⇒ Gaussian n is "static"
            motion_floor:          minimum Amotion weight for one-static pairs
                                   (0=hard gate, 1=ignore motion)
            boundary:              optional BoundarySuppression instance; if
                                   provided, multiplies W by (1 - B) per Eq. 6
                                   of the proposal (Sec. 4.3)
        """
        self.gaussians = gaussians
        self.k = k
        self.opacity_thresh = opacity_thresh
        self.use_geometry = use_geometry
        self.sigma_pos = sigma_pos
        self.sigma_color = sigma_color
        self.sigma_scale = sigma_scale
        self.power = power
        self.deform_model = deform_model
        self.n_time_steps = n_time_steps
        self.static_motion_thresh = static_motion_thresh
        self.motion_floor = motion_floor
        self.boundary = boundary

    @torch.no_grad()
    def _compute_trajectories(self, pos):
        """
        Query the deformation MLP at T uniformly-spaced time steps in [0, 1].

        Returns:
            traj_xyz: [T, N, 3] displacement trajectories Δμ(t) = d_xyz at time t
            traj_rot: [T, N, 4] additive rotation deltas Δq(t) = d_rotation at time t
        """
        device = pos.device
        N = pos.shape[0]
        T = self.n_time_steps
        times = torch.linspace(0.0, 1.0, T, device=device)

        xyz_list, rot_list = [], []
        for t_val in times:
            time_input = torch.full((N, 1), t_val.item(), dtype=torch.float32, device=device)
            d_xyz, d_rot, _ = self.deform_model.step(pos.detach(), time_input)
            xyz_list.append(d_xyz)
            rot_list.append(d_rot)

        traj_xyz = torch.stack(xyz_list, dim=0)  # [T, N, 3]
        traj_rot = torch.stack(rot_list, dim=0)  # [T, N, 4]
        return traj_xyz, traj_rot

    @torch.no_grad()
    def build(self, return_components=False):
        """
        Build the affinity graph.

        Args:
            return_components: if True, also return dict of individual A_x tensors

        Returns:
            edge_index:  [2, E] LongTensor  — (source, target) index pairs
            weights:     [E]    FloatTensor — W(i,j) = Ageo per edge
            valid_mask:  [N]    BoolTensor  — which original Gaussians passed opacity filter
            components:  dict with keys Apos, Acolor, Aorient, Ascale  (only if return_components=True)
        """
        g = self.gaussians

        # ── 1. Opacity filter ─────────────────────────────────────────────
        valid = g.get_opacity.squeeze(1) > self.opacity_thresh      # [N]
        pos   = g.get_xyz[valid]                                    # [N', 3]
        scale = g.get_scaling[valid]                                # [N', 3]  exp already applied
        color = g._features_dc[valid].squeeze(1)                   # [N', 3]  DC SH component
        rot_q = g.get_rotation[valid]                               # [N', 4]  normalized quaternions

        N = pos.shape[0]
        device = pos.device

        # ── 2. Principal axis from rotation + scale ───────────────────────
        # Gaussian covariance: Σ = R · diag(s²) · R^T
        # Eigenvectors = columns of R, eigenvalues = s²
        # Principal axis = column of R with largest scale value
        R     = build_rotation(rot_q)                               # [N', 3, 3]
        max_s = scale.argmax(dim=1)                                 # [N']
        v1    = R[torch.arange(N, device=device), :, max_s]        # [N', 3]
        v1    = F.normalize(v1, dim=1)

        # ── 3. k-NN graph in canonical space (GPU) ────────────────────────
        # Graph topology is spatial (x,y,z) per the paper's theory.
        # Object separation is handled by edge weights (Acolor, Aorient,
        # Ascale), not by the graph topology.
        # torch_cluster.knn(x, y, k): for each point in y find k nearest in x
        # k+1 to account for self-loops, which are removed below
        edge_index = torch_knn(pos, pos, k=self.k + 1)             # [2, N'*(k+1)]
        self_loop  = edge_index[0] == edge_index[1]
        edge_index = edge_index[:, ~self_loop]                      # [2, E]
        i, j = edge_index[0], edge_index[1]                        # E each

        # ── 4. Affinity components ────────────────────────────────────────

        # Apos: spatial proximity
        Apos = torch.exp(
            -((pos[i] - pos[j]) ** 2).sum(dim=1)
            / (2 * self.sigma_pos ** 2)
        )

        # Acolor: DC SH appearance similarity
        Acolor = torch.exp(
            -((color[i] - color[j]) ** 2).sum(dim=1)
            / (2 * self.sigma_color ** 2)
        )

        # Aorient: principal axis alignment — key signal for static scenes
        # |v_i · v_j| = 1 if axes aligned, 0 if perpendicular
        Aorient = (v1[i] * v1[j]).sum(dim=1).abs()

        # Ascale: Gaussian size similarity via log scale ratio
        norm_i = scale[i].norm(dim=1).clamp(min=1e-6)
        norm_j = scale[j].norm(dim=1).clamp(min=1e-6)
        Ascale = torch.exp(
            -(norm_i / norm_j).log() ** 2
            / (2 * self.sigma_scale ** 2)
        )

        # ── 5. Final weight (geometric mean, Apos excluded) ──────────────
        # Apos is excluded: the k-NN graph is built on position alone, so
        # every edge already connects spatially close points → Apos ≈ 1.0
        # for all edges by construction → zero discriminative signal.
        # Spatial locality is already encoded in the graph topology.
        # W = geometric mean of the three discriminative terms.
        if self.use_geometry:
            W = (Acolor * Aorient * Ascale) ** (1.0 / 3.0)          # [E]
        else:
            # No-geometry ablation: retain spatial kNN topology, remove the
            # geometric edge affinity so motion/boundary terms can be isolated.
            W = torch.ones_like(Acolor)

        # ── 6. Power sharpening (optional) ───────────────────────────────
        if self.power != 1.0:
            W = W ** self.power

        # ── 7. Amotion = Atraj · Arot  (proposal Eq. 2–3) ───────────────────
        # Only computed when a deform_model is provided (Phase 2+).
        Amotion = None
        if self.deform_model is not None:
            print(f"  Computing trajectories ({self.n_time_steps} MLP queries)...")
            traj_xyz, traj_rot = self._compute_trajectories(pos)  # [T,N,3], [T,N,4]
            T = self.n_time_steps

            # ── Atraj: vectorial centered cosine similarity (Eq. 3) ───────
            # mean_traj: mean displacement over time  [N, 3]
            # centered:  Δμ(t) - mean_Δμ             [T, N, 3]
            # r_i:       sqrt(Σ_t ‖centered_t_i‖²)   [N]  (RMS trajectory norm)
            mean_traj = traj_xyz.mean(dim=0)           # [N, 3]
            centered  = traj_xyz - mean_traj.unsqueeze(0)  # [T, N, 3]
            r = centered.pow(2).sum(dim=-1).sum(dim=0).sqrt()  # [N]

            # Numerator: Σ_t centered(t,i) · centered(t,j)
            # Iterate over T to avoid materialising [T, E, 3] (~3 GB for large scenes).
            numerator = torch.zeros(len(i), device=device)
            for t in range(T):
                numerator += (centered[t][i] * centered[t][j]).sum(dim=-1)  # [E]

            Atraj = (numerator / (r[i] * r[j] + 1e-8)).clamp(min=0.0)  # [E] ∈ [0,1]

            # ── Arot: mean |Δq(t)_i · Δq(t)_j| over T steps ─────────────
            # d_rotation from the MLP is an additive delta quaternion, so
            # traj_rot[t] already IS Δq(t) — no Hamilton product needed.
            arot_sum = torch.zeros(len(i), device=device)
            for t in range(T):
                arot_sum += (traj_rot[t][i] * traj_rot[t][j]).sum(dim=-1).abs()
            Arot = arot_sum / T  # [E] ∈ [0, 1]

            Amotion = Atraj * Arot  # [E] ∈ [0, 1]

            # Static-aware fusion: the ε=1e-8 in Atraj's denominator makes
            # static-vs-static pairs evaluate to ~0 regardless of rigidity,
            # which is wrong — they should delegate to the geometric signals.
            static    = r < self.static_motion_thresh          # [N]
            s_i, s_j  = static[i], static[j]
            both_stat = s_i & s_j
            one_stat  = s_i ^ s_j
            # both-moving falls through with raw Amotion (pure product)
            Amotion_fused = torch.where(
                both_stat,
                torch.ones_like(Amotion),
                torch.where(
                    one_stat,
                    self.motion_floor + (1.0 - self.motion_floor) * Amotion,
                    Amotion,
                ),
            )

            n_bs = both_stat.sum().item()
            n_os = one_stat.sum().item()
            n_bm = Amotion.numel() - n_bs - n_os
            print(f"  Atraj:   min={Atraj.min():.4f}  max={Atraj.max():.4f}  mean={Atraj.mean():.4f}")
            print(f"  Arot:    min={Arot.min():.4f}  max={Arot.max():.4f}  mean={Arot.mean():.4f}")
            print(f"  Amotion: min={Amotion.min():.4f}  max={Amotion.max():.4f}  mean={Amotion.mean():.4f}  "
                  f"<0.5={( Amotion < 0.5).float().mean()*100:.1f}%")
            print(f"  Pairs:   both-static={n_bs:,}  one-static={n_os:,}  both-moving={n_bm:,}  "
                  f"(static thresh r<{self.static_motion_thresh:g})")
            print(f"  Fused:   min={Amotion_fused.min():.4f}  max={Amotion_fused.max():.4f}  "
                  f"mean={Amotion_fused.mean():.4f}")

            W = W * Amotion_fused

        # ── 8. Boundary suppression (proposal Sec. 4.3, Eq. 6) ──────────────
        # W ← W · (1 − B), with B ∈ [0, 1) computed from image-space depth
        # and RGB edge responses along each projected edge segment.
        B = None
        if self.boundary is not None:
            print("  Computing boundary suppression B...")
            B = self.boundary.compute(g, edge_index, valid)     # [E]
            W = W * (1.0 - B)

        if return_components:
            components = {
                'Apos':    Apos,
                'Acolor':  Acolor,
                'Aorient': Aorient,
                'Ascale':  Ascale,
            }
            if Amotion is not None:
                components['Amotion'] = Amotion
            if B is not None:
                components['B'] = B
            return edge_index, W, valid, components

        return edge_index, W, valid
