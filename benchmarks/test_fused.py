import matplotlib.pyplot as plt
import numpy as np
import torch
from quarot import matmul, matmul_hadamard, matmul_handwritten
from quarot.functional.quantization import pack_i4, unpack_i4
from quarot.nn import Linear4bit, OnlineHadamard, Quantizer
from scipy.linalg import hadamard as scipy_hadamard

# ncu -f -o profile --set detailed --target-processes all -k regex:"matmul_hadamard_kernel|Kernel" python benchmarks/test_fused.py


def test_matmul_handwritten():
    size = [4096, 4096]
    A = torch.randint(
        1, 7, (size[0], size[1] // 2), dtype=torch.uint8, requires_grad=False
    ).to("cuda")
    B = torch.randint(
        1, 7, (size[0], size[1] // 2), dtype=torch.uint8, requires_grad=False
    ).to("cuda")
    C = matmul(A, B)
    C_ = matmul_handwritten(A, B)
    assert torch.all(C == C_)


def segment_wise_hadamard(A):
    L = 128
    H = torch.as_tensor(scipy_hadamard(L)).to(torch.float16).to("cuda")
    H = torch.block_diag(*[H] * (A.shape[1] // L))
    return A @ H


def main():
    torch.manual_seed(2)
    torch.set_printoptions(edgeitems=10, linewidth=1000, sci_mode=False)
    size = (4096, 4096)

    A = torch.randn(size, dtype=torch.float16).to("cuda")
    A_h = segment_wise_hadamard(A)
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

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
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
    ax3.imshow((C - C_gt).cpu().numpy(), **args)
    # plt.hist((C - C_gt).cpu().numpy().flatten(), bins=100)
    plt.show(block=True)


if __name__ == "__main__":
    main()
    # test_matmul_handwritten()
