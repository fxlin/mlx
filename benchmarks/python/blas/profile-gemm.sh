# NB: must modify bench_gemm.py to run light tests, otherwise the trace fill will be too large to render (on m2 air

PROG=mlx-gemm

TRACE="$(basename "$PROG" .py)_$(date +%Y%m%d%H%M%S).trace"

xctrace record --template "FL-gpu-counters1" \
--output /tmp/${TRACE} \
--launch -- \
/Users/felixlin/workspace-mps/myenv/bin/python3 \
bench_gemm.py --gpu