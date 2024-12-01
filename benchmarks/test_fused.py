import matplotlib.pyplot as plt
import numpy as np
import torch

from quarot import matmul, matmul_fused


def main():
    torch.manual_seed(2)
    torch.set_printoptions(edgeitems=128, linewidth=1000)
    size = [4096, 4096]
    A = torch.randint(
        1, 7, (size[0], size[1] // 2), dtype=torch.uint8, requires_grad=False
    ).to("cuda")
    B = torch.randint(
        1, 7, (size[0], size[1] // 2), dtype=torch.uint8, requires_grad=False
    ).to("cuda")
    C = matmul(A, B)
    C_ = matmul_fused(A, B)
    assert torch.all(C == C_)


if __name__ == "__main__":
    main()
