import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

_src_path = Path(__file__).parent

_include_path = _src_path / "include"
_c_src_path = _src_path / "src"

_C = load(
    name='_bvh_tracing',
    extra_cuda_cflags=["-O3", "--expt-extended-lambda"],
    extra_cflags=["-O3"],
    sources=[
        str(_c_src_path / f)
        for f in [
            'bvh.cu',
            'trace.cu',
            'construct.cu',
            'bindings.cpp',
        ]
    ],
    extra_include_paths=[
        str(_include_path),
    ],
    verbose=False
)

def named_normalize(r: torch.Tensor, dim: str):
    r = r.align_to(..., dim)
    names = r.names

    return F.normalize(r.rename(None), dim=-1).refine_names(*names)

def quats2mats(q):
    q = named_normalize(q, dim="wxyz").align_to(..., "wxyz")
    w, x, y, z = torch.split(q.rename(None), [1, 1, 1, 1], dim=-1)
    w, x, y, z = w.squeeze(-1), x.squeeze(-1), y.squeeze(-1), z.squeeze(-1)

    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - w * z)
    r02 = 2 * (x * z + w * y)
    r10 = 2 * (x * y + w * z)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z - w * x)
    r20 = 2 * (x * z - w * y)
    r21 = 2 * (y * z + w * x)
    r22 = 1 - 2 * (x * x + y * y)

    R = torch.stack([
        r00, r01, r02,
        r10, r11, r12,
        r20, r21, r22,
    ], dim=-1).view(*q.shape[:-1], 3, 3).refine_names(*q.names[:-1], None, None)

    return R

class RayTracer:
    def __init__(self, *, means: torch.Tensor, scales: torch.Tensor, quats: torch.Tensor):
        num_gaussian, _ = means.size()

        rot = quats2mats(quats).rename(None)
        means = means.rename(None)
        scales = scales.rename(None)

        # shape checks. provide information to code LLMs.
        assert means.size() == (num_gaussian, 3)
        assert scales.size() == (num_gaussian, 3)
        assert quats.size() == (num_gaussian, 4)


        nodes = torch.full((2 * num_gaussian - 1, 5), -1, device="cuda").int()
        nodes[:num_gaussian - 1, 4] = 0
        nodes[num_gaussian - 1:, 4] = 1
        aabbs = torch.zeros(2 * num_gaussian - 1, 6, device="cuda").float()
        aabbs[:, :3] = 100000
        aabbs[:, 3:] = -100000

        m = 3
        a, b, c = rot[:, :, 0], rot[:, :, 1], rot[:, :, 2]
        sa, sb, sc = m * scales[:, None, 0], m * scales[:, None, 1], m * scales[:, None, 2]

        x111 = means + a * sa + b * sb + c * sc
        x110 = means + a * sa + b * sb - c * sc
        x101 = means + a * sa - b * sb + c * sc
        x100 = means + a * sa - b * sb - c * sc
        x011 = means - a * sa + b * sb + c * sc
        x010 = means - a * sa + b * sb - c * sc
        x001 = means - a * sa - b * sb + c * sc
        x000 = means - a * sa - b * sb - c * sc

        x = torch.stack([x000, x001, x010, x011, x100, x101, x110, x111])
        aabb_min = x.min(dim=0).values
        aabb_max = x.max(dim=0).values

        aabbs[num_gaussian - 1:] = torch.cat([aabb_min, aabb_max], dim=-1)

        self.tree, self.aabb, self.morton = _C.create_bvh(means, scales, quats, nodes, aabbs)

    @torch.no_grad()
    def trace_visibility(self, rays_o, rays_d, means, covs3Dinv, opacity, normals):
        """
        covs3Dinv: shape=(num_gaussian, 6), eqivalent to replacing scales -> 1. / scales in the original covariance computation and then take the triu part
        """
        cotrib, opa = _C.trace_bvh_opacity(self.tree, self.aabb,
                                                 rays_o, rays_d,
                                                 means, covs3Dinv,
                                                 opacity, normals)
        return {
            "visibility": opa.unsqueeze(-1),
            "contribute": cotrib.unsqueeze(-1),
        }


# region: tests & demos

def random_rotor():
    q = torch.randn(4, names=["wxyz"])
    while q.rename(None).norm() < 1e-3:
        q = torch.randn(4, names=["wxyz"])
    return q

def test():
    from tqdm.auto import tqdm, trange

    I = torch.eye(3)
    O = torch.zeros(3, 3)
    pbar = trange(1024)
    for _ in pbar:
        R = quats2mats(random_rotor())
        diff = I - R @ R.transpose(-1, -2)
        pbar.set_postfix(diff=diff.rename(None).max().item())
        assert torch.allclose(diff, O, atol=1e-6, rtol=1e-6)

    means = torch.randn((100000, 3), device="cuda", names=["gaussian", "xyz"]) / 3
    quats = torch.randn((100000, 4), device="cuda", names=["gaussian", "wxyz"])
    scales = torch.rand((100000, 3), device="cuda", names=["gaussian", "xyz"]) * .01
    opacities = torch.rand((100000, 1), device="cuda", names=["gaussian", "opacity"]) * .01

    # normals = torch.nn.functional.normalize(means.rename(None)) # face towards outside
    normals = torch.tensor([1, 0, 0]).float().cuda()[None, ...].repeat(100000, 1)

    import gsplat
    # shape: (num_gaussian, 3, 3), symmetric, eqivalent to replacing scales -> 1. / scales in the original covariance computation
    cov3Dinv, _ = gsplat.quat_scale_to_covar_preci(quats=quats.rename(None), scales=1. / scales.rename(None), compute_covar=True, compute_preci=False, triu=True)

    ray_tracer = RayTracer(means=means, quats=quats, scales=scales)

    # visualize the tracing results
    resolution = 1024

    pos_2d_x = torch.arange(0, resolution).cuda() / resolution # [0, 1)
    pos_2d_y = torch.arange(0, resolution).cuda() / resolution # [0, 1)
    pos_2d_y, pos_2d_x = torch.meshgrid(pos_2d_y, pos_2d_x, indexing="ij") # h, w

    pos_3d_z = -(pos_2d_y * 2 - 1)
    pos_3d_y = pos_2d_x * 2 - 1
    pos_3d_x = torch.tensor([10], device="cuda")[None, :].repeat(resolution, resolution, 1).squeeze(-1) # h, w

    rays_o = torch.stack([pos_3d_x, pos_3d_y, pos_3d_z], dim=-1).refine_names("h", "w", "xyz")
    rays_d = torch.tensor([-1., 0., 0.]).cuda()[None, None, :].repeat(resolution, resolution, 1).refine_names("h", "w", "xyz")

    traced = ray_tracer.trace_visibility(rays_o, rays_d, means, cov3Dinv, opacities, normals)

    import matplotlib.pyplot as plt
    plt.imshow(traced["visibility"].rename(None).cpu().numpy())
    plt.colorbar()
    plt.show()

    plt.imshow(traced["contribute"].rename(None).cpu().numpy())
    plt.colorbar()
    plt.show()

# endregion
