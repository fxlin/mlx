xctrace record --template "FL-gpu-counters" \
--output gemv.trace \
--launch -- \
/Users/felixlin/workspace-mps/myenv/bin/python3 \
bench_gemv.py --gpu