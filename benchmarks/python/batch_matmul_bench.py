# Copyright © 2023 Apple Inc.

# FL: only for testing functions, not for benchmarking (too short)

import argparse

import mlx.core as mx
from time_utils import time_fn

# B = 8
# T = 1024
# D = 512

# xzl
B = 4
T = 2048
D = 2048

def time_batch_matmul():
    mx.random.seed(3)
    a = mx.random.uniform(shape=(B, T, D))
    b = mx.random.uniform(shape=(D, D))
    c = mx.random.uniform(shape=(B, T, D))
    mx.eval(a, b, c)
    time_fn(mx.matmul, a, b)

    a16 = mx.random.uniform(shape=(B, T, D), dtype=mx.float16)
    b16 = mx.random.uniform(shape=(D, D), dtype=mx.float16)

    a16bf = mx.random.uniform(shape=(B, T, D), dtype=mx.bfloat16)
    b16bf = mx.random.uniform(shape=(D, D), dtype=mx.bfloat16)

    mx.eval(a16, b16, a16bf, b16bf)
    
    time_fn(mx.matmul, a16, b16)
    time_fn(mx.matmul, a16bf, b16bf)

    def batch_vjp_first():
        return mx.vjp(mx.matmul, [a, b], [c])[1][0]

    time_fn(batch_vjp_first)

    def batch_vjp_second():
        return mx.vjp(mx.matmul, [a, b], [c])[1][1]

    time_fn(batch_vjp_second)


def time_unbatch_matmul():
    mx.random.seed(3)
    a = mx.random.uniform(shape=(B * T, D))
    b = mx.random.uniform(shape=(D, D))
    c = mx.random.uniform(shape=(B * T, D))
    mx.eval(a, b, c)
    time_fn(mx.matmul, a, b)

    def unbatch_vjp_first():
        return mx.matmul(c, mx.transpose(b))

    time_fn(unbatch_vjp_first)

    def unbatch_vjp_second():
        return mx.matmul(mx.transpose(a), c)

    time_fn(unbatch_vjp_second)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("MLX benchmarks.")
    parser.add_argument("--gpu", action="store_true", help="Use the Metal back-end.")
    args = parser.parse_args()
    if args.gpu:
        mx.set_default_device(mx.gpu)
    else:
        mx.set_default_device(mx.cpu)

    time_batch_matmul()
    time_unbatch_matmul()
