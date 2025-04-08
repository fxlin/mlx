xctrace record --template "FL-gpu-counters" \
--output batch-matmul.trace \
--launch -- \
/Users/felixlin/workspace-mps/myenv/bin/python3 \
batch_matmul_bench.py --gpu