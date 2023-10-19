import time

import torch
import torch.nn as nn
from torchvision.models import resnet50

from thop import profile

KILO = 1e3
MEGA = 1e6
GIGA = 1e9
TERA = 1e12
EXA = 1e18


def measure_flops(model, shape, repeat=10, unit=TERA):
    # MAC: Multiply–accumulate operation
    batch = torch.randn(*shape)

    flops, params = profile(model, inputs=(batch,))

    with torch.no_grad():
        # Prepare
        torch.cuda.empty_cache()

        batch = batch.cuda()
        model = model.cuda()

        torch.cuda.synchronize()

        # Start
        start = time.time()

        for i in range(repeat):
            _ = model(batch)

        torch.cuda.synchronize()
        end = time.time()
        # --

    return (flops * repeat) / (end - start) / unit


x = 10
n, m = 1024 * x, 1024 * x

# print(measure_flops(nn.Linear(n, m), shape=(m, n)))
# print(measure_flops(resnet50(), shape=(256, 3, 224, 224)))


import torch, time

torch.backends.cuda.matmul.allow_tf32 = False


def f(N, m=5000000, n=256, unit=TERA, dtype=torch.float32):
    torch.cuda.empty_cache()
    a = torch.eye(n, dtype=dtype, device="cuda:0")
    x = torch.randn((m, n), dtype=dtype, device="cuda:0")
    y = torch.zeros_like(x)

    torch.cuda.synchronize()
    ts = -time.time()
    for _ in range(N):
        # No allocation in main loop using dual-out strategy
        y = torch.mm(x, a, out=y)
        x = torch.mm(y, a, out=x)
    torch.cuda.synchronize()
    ts += time.time()
    torch.cuda.empty_cache()
    F = N * (2 * m * n * n + 2 * m * n * n)
    return F / ts / unit


m = 1024 * 200
# Skinny matrix
print(f"{f(100,m=m, n=512, dtype=torch.half):6.2f}TF @ n=512")
print(f"{f(100,m=m, n=384, dtype=torch.half):6.2f}TF @ n=384")
print(f"{f(100,m=m, n=256, dtype=torch.half):6.2f}TF @ n=256")
print(f"{f(100,m=m, n=192, dtype=torch.half):6.2f}TF @ n=192")
print(f"{f(100,m=m, n=128, dtype=torch.half):6.2f}TF @ n=128")
print(f"{f(100,m=m, n=96 , dtype=torch.half):6.2f}TF @ n=96")
print(f"{f(100,m=m, n=64 , dtype=torch.half):6.2f}TF @ n=64")
print(f"{f(100,m=m, n=48 , dtype=torch.half):6.2f}TF @ n=48")
print(f"{f(100,m=m, n=32 , dtype=torch.half):6.2f}TF @ n=32")

# Square
n = 5
print(f"{f(100,m=1024 * n, n=1024 * n):6.2f}TF")
print(f"{f(100,m=1024 * n, n=1024 * n, dtype=torch.half):6.2f}TF")
