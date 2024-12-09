import matplotlib.pyplot as plt
import numpy as np
import torch
from quarot import matmul, matmul_hadamard, matmul_handwritten
from quarot.functional.quantization import pack_i4, unpack_i4
from quarot.nn import Linear4bit, OnlineHadamard, Quantizer
from scipy.linalg import hadamard as scipy_hadamard
import marlin_reproduction
import marlin
import torch.nn as nn
# ncu -f -o profile --set detailed --target-processes all -k regex:"matmul_hadamard_kernel|Kernel" python benchmarks/test_fused.py

DEV = "cuda"
def test_matmul_handwritten():
    size = [4096, 4096]
    A = torch.randint(
        1, 7, (size[0], size[1] // 2), dtype=torch.uint8, requires_grad=False
    ).to(DEV)
    B = torch.randint(
        1, 7, (size[0], size[1] // 2), dtype=torch.uint8, requires_grad=False
    ).to(DEV)
    C = matmul(A, B)
    C_ = matmul_handwritten(A, B)
    assert torch.all(C == C_)


def segment_wise_hadamard(A):
    L = 128
    H = torch.as_tensor(scipy_hadamard(L)).to(torch.float16).to(DEV)
    H = torch.block_diag(*[H] * (A.shape[1] // L))
    return A @ H

def gen_quant4(m, n, groupsize=-1, is_reproduction=False):
    tile = 16
    maxq = 2 ** 4 - 1
    w = torch.randn((m, n), dtype=torch.half, device=DEV)
    if groupsize != -1:
        w = w.reshape((-1, groupsize, n))
        w = w.permute(1, 0, 2)
        w = w.reshape((groupsize, -1))
    s = torch.max(torch.abs(w), 0, keepdim=True)[0]
    s *= 2 / maxq
    w = torch.round(w / s).int()
    w += (maxq + 1) // 2
    w = torch.clamp(w, 0, maxq)
    ref = (w - (maxq + 1) // 2).half() * s
    if groupsize != -1:
        def reshape(w):
            w = w.reshape((groupsize, -1, n))
            w = w.permute(1, 0, 2)
            w = w.reshape((m, n)).contiguous()
            return w
        ref = reshape(ref)
        w = reshape(w)
    s = s.reshape((-1, n)).contiguous()
    linear = nn.Linear(m, n)
    linear.weight.data = ref.t()
    # Workaround to test some special cases that are forbidden by the API
    if is_reproduction:
        layer = marlin_reproduction.Layer(256, 256, groupsize=groupsize)
    else:
        layer = marlin.Layer(256, 256, groupsize=groupsize)
    if groupsize == -1:
        groupsize = m
    layer.k = m
    layer.n = n
    layer.groupsize = groupsize
    layer.B = torch.empty((m // 16, n * 16 // 8), dtype=torch.int, device=DEV)
    layer.s = torch.empty((m // groupsize, n), dtype=torch.half, device=DEV)
    layer.pack(linear, s.t())
    q = layer.B
    s = layer.s
    return ref, q, s


def main():
    torch.manual_seed(2)
    torch.set_printoptions(edgeitems=10, linewidth=1000, sci_mode=False)
    size = (4096, 4096)

    A = torch.randn(size, dtype=torch.float16).to(DEV)
    A_h = segment_wise_hadamard(A)
    # scale = torch.max(torch.abs(A_h), dim=1, keepdim=True).values / 7
    # A_hq = (A_h / scale).round().to(torch.int8)
    # B = torch.randint(
    #     1, 7, (size[0], size[1] // 2), dtype=torch.uint8, requires_grad=False
    # ).to(DEV)
    groupsize = 128 
    B_fp16, B, s = gen_quant4(size[0], size[1], groupsize=groupsize)
    # B_fp16 = unpack_i4(B).to(torch.float16)
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
    s_w = torch.ones((size[1], 1), dtype=torch.float16, device=DEV)
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
