import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from quarot import matmul, matmul_hadamard, matmul_handwritten
from quarot.functional.quantization import pack_i4, unpack_i4
from quarot.nn import Linear4bit, OnlineHadamard, Quantizer
from scipy.linalg import hadamard as scipy_hadamard
from torch import nn

# ncu -f -o profile --set detailed --target-processes all -k regex:"hadamard|^Kernel$|quant" python benchmarks/test_fused.py
# ncu -f -o profile_matmul --set detailed --target-processes all -k regex:"^Kernel$|matmul_handwritten" python benchmarks/test_fused.py
# ncu -f -o profile_fused --set detailed --target-processes all -k regex:"Kernel|hadamard|quant|" python benchmarks/test_fused.py


def random_i4_matrix(m, n):
    i8 = torch.randint(-8, 7, (m, n), dtype=torch.int8, requires_grad=False)
    return pack_i4(i8.to("cuda"))


def test_matmul_handwritten(m, n, k):
    A = random_i4_matrix(m, k)
    B = random_i4_matrix(n, k)
    C = matmul(A, B)
    C_ = matmul_handwritten(A, B)
    assert torch.all(C == C_)


def segment_wise_hadamard(A, L):
    H = torch.as_tensor(scipy_hadamard(L)).to(torch.float16).to("cuda")
    H = torch.block_diag(*[H] * (A.shape[1] // L))
    return A @ H


def test_fused_kernel():
    torch.manual_seed(2)
    torch.set_printoptions(edgeitems=10, linewidth=1000, sci_mode=False)
    size = (256, 256)

    A = torch.randn(size, dtype=torch.float16).to("cuda")
    A_h = segment_wise_hadamard(A, 128)
    # scale = torch.max(torch.abs(A_h), dim=1, keepdim=True).values / 7
    # A_hq = (A_h / scale).round().to(torch.int8)
    B = torch.randint(
        1, 7, (size[0], size[1] // 2), dtype=torch.uint8, requires_grad=False
    ).to("cuda")
    B_fp16 = unpack_i4(B).to(torch.float16)
    C_gt = A_h @ B_fp16.T
    # C_gtq = A_hq.to(torch.float16) @ B_fp16.T
    # A_int4 = pack_i4(A_hq)

    C = matmul_hadamard(A, B)

    # Baseline: unfused implementation
    baseline_mod = (
        torch.nn.Linear(size[0], size[1], bias=False).cuda().to(torch.float16)
    )
    baseline_mod.weight.data = torch.randint_like(
        baseline_mod.weight.data, low=-8, high=7
    ).to(torch.float16)
    s_w = torch.ones((size[1], 1), dtype=torch.float16, device="cuda")
    int4_mod_fp16had = torch.nn.Sequential(
        OnlineHadamard(baseline_mod.in_features, force_fp32=False),
        Quantizer(input_clip_ratio=1.0),
        Linear4bit.from_float(baseline_mod, weight_scales=s_w),
    ).cuda()
    res = int4_mod_fp16had(A[None, :, :])

    fig, (ax1, ax2, ax3, cax) = plt.subplots(
        1, 4, figsize=(15, 5), width_ratios=[1, 1, 1, 0.05]
    )
    vmin = min(C.min(), C_gt.min())
    vmax = max(C.max(), C_gt.max())
    args = {
        "vmin": vmin,
        "vmax": vmax,
        # "cmap": "hot",
        "interpolation": "nearest",
    }
    ax1.set_title("C (fused quant w4a4)")
    ax1.imshow(C.cpu().numpy(), **args)
    ax2.set_title("C ground truth")
    ax2.imshow(C_gt.cpu().numpy(), **args)
    ax3.set_title("difference")
    im = ax3.imshow((C - C_gt).cpu().numpy(), **args)

    # divider = make_axes_locatable(ax3)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)
    fig.tight_layout()

    plt.show(block=True)
    fig.savefig("fused_quant_w4a4.png", dpi=300)


BENCHMARK_SIZES = [
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (3072, 3072, 3072),
    (4096, 4096, 4096),
    (128, 4096, 4096),
    (256, 4096, 4096),
    (512, 4096, 4096),
    (1024, 4096, 4096),
]


def benchmark_matmul_handwritten():
    for size in BENCHMARK_SIZES:
        test_matmul_handwritten(*size)


# ncu -f -o profile_fused --set detailed --target-processes all -k regex:"^Kernel$|hadamard|quant|vectorized_elementwise_kernel|reduce_kernel" python benchmarks/test_fused.py
def benchmark_fused_kernel():
    torch.manual_seed(42)
    for m, n, k in BENCHMARK_SIZES[1:]:
        A = torch.randn((m, k), dtype=torch.float16).to("cuda")
        B = random_i4_matrix(n, k)
        C = matmul_hadamard(A, B)

        baseline_mod = torch.nn.Linear(n, k, bias=False).cuda().to(torch.float16)
        baseline_mod.weight.data = torch.randint_like(
            baseline_mod.weight.data, low=-8, high=7
        ).to(torch.float16)
        s_w = torch.ones((n, 1), dtype=torch.float16, device="cuda")
        int4_mod_fp16had = torch.nn.Sequential(
            OnlineHadamard(baseline_mod.in_features, force_fp32=False),
            Quantizer(input_clip_ratio=1.0),
            Linear4bit.from_float(baseline_mod, weight_scales=s_w),
        ).cuda()
        C_ref = int4_mod_fp16had(A[None, :, :])
        print(f"Done for {m}x{n}x{k}")
        # if plot:
        #     fig, (ax1, ax2, ax3, cax) = plt.subplots(
        #         1, 4, figsize=(15, 5), width_ratios=[1, 1, 1, 0.05]
        #     )
        #     vmin = min(C.min(), C_gt.min())
        #     vmax = max(C.max(), C_gt.max())
        #     args = {"vmin": vmin, "vmax": vmax, "interpolation": "nearest"}
        #     ax1.set_aspect("equal")
        #     ax1.set_title("C (fused quant w4a4)")
        #     ax1.imshow(C.cpu().numpy(), **args)
        #     ax2.set_aspect("equal")
        #     ax2.set_title("C ground truth")
        #     ax2.imshow(C_gt.cpu().numpy(), **args)
        #     ax3.set_aspect("equal")
        #     ax3.set_title("difference")
        #     im = ax3.imshow((C - C_gt).cpu().numpy(), **args)
        #     fig.colorbar(im, cax=cax)
        #     fig.tight_layout()
        #     fig.savefig(f"plots/fused_quant_w4a4_{m}_{n}_{k}.png", dpi=300)

        #     fig, (hist_c, hist_diff) = plt.subplots(1, 2)
        #     hist_c.hist(C.cpu().numpy().flatten(), bins=100)
        #     rel_error = (C - C_gt) / (C_gt + 1e-6)
        #     rel_error = rel_error.cpu().numpy().flatten()
        #     rel_error = rel_error[(-2 < rel_error) & (rel_error < 2)]
        #     hist_diff.hist(rel_error, bins=100)
        #     mean = np.mean(rel_error)
        #     var = np.var(rel_error)
        #     print(f"Mean: {mean}, Variance: {var}")
        #     fig.savefig(f"plots/fused_quant_w4a4_{m}_{n}_{k}_hist.png", dpi=300)
        #     print(f"Saved plot for {m}x{n}x{k}")


if __name__ == "__main__":
    benchmark_fused_kernel()
