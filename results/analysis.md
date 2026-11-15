# Performance notes: RTX 5060 Laptop GPU

## What the benchmark shows

At 2048³, going from the naive kernel to a shared-memory tile improved performance from 1.14 to 1.52 TFLOP/s. Changing to a 64 × 64 output tile and keeping 16 sums per thread in registers reached 4.41 TFLOP/s. The FP32 cuBLAS reference reached 10.65 TFLOP/s.

For a full 16-step slice of K, the 16 × 16 shared kernel requests 512 input values to update 256 output elements. The register kernel requests 2,048 values to update 4,096 output elements: one-quarter as many explicit input loads per output. This is a source-level count for full tiles. It does not measure DRAM traffic; caching, broadcast, coalescing, and the GPU's execution behavior affect actual transactions.

`cuobjdump --dump-resource-usage build\gemm.exe` showed 40 registers per thread for the naive and shared kernels and 72 for the register kernel. The wider tile improves reuse but costs registers, which can reduce the number of resident warps. The change also reduces the grid size: at 256³ it launches 16 blocks instead of 256. That limits available parallel work and is one plausible contributor to the small-shape slowdown; these measurements do not isolate its cause.

## Trace and reproducibility

Nsight Systems 2026.3.2 captured the 1024³ CUDA kernel activity. The [GPU kernel summary](nsys-1024-kernels.csv) reported the following medians across traced launches. These trace results include warm-ups and correctness launches, so use the separate CUDA-event CSV for benchmark comparisons.

| Kernel | Instances | Median duration (ms) |
| --- | ---: | ---: |
| Naive | 13 | 2.099 |
| Shared | 13 | 1.574 |
| Register | 13 | 0.485 |
| cuBLAS (SIMT SGEMM in this trace) | 14 | 0.221 |

Run from the project root, using the installed Nsight Systems CLI path if `nsys` is not on PATH:

```powershell
nsys profile --trace=cuda --sample=none --force-overwrite=true --output=build/gemm-timeline .\build\gemm.exe --shape 1024x1024x1024 --warmup 2 --iterations 5 --samples 2
nsys stats --force-export=true --report cuda_gpu_kern_sum --format csv build/gemm-timeline.nsys-rep
cuobjdump --dump-resource-usage .\build\gemm.exe
```

Reproduce the memory and shared-memory hazard checks with:

```powershell
compute-sanitizer --tool memcheck --error-exitcode 2 .\build\gemm.exe --verify-only --shape 63x65x17 --shape 97x81x33
compute-sanitizer --tool racecheck --error-exitcode 2 .\build\gemm.exe --verify-only --shape 63x65x17 --kernel shared
compute-sanitizer --tool racecheck --error-exitcode 2 .\build\gemm.exe --verify-only --shape 97x81x33 --kernel register
```

Nsight Compute returned `ERR_NVGPUCTRPERM` on this machine. No hardware counter results are claimed. With [counter access enabled](https://developer.nvidia.com/ERR_NVGPUCTRPERM), collect occupancy, memory traffic, and warp stall data to test the reuse and parallelism hypotheses:

```powershell
ncu --section SpeedOfLight --section Occupancy --section WarpStateStats --section MemoryWorkloadAnalysis --kernel-name 'regex:^gemm_' --export build/gemm-compute .\build\gemm.exe --verify-only --shape 256x256x256 --shape 2048x2048x2048
```

One useful experiment is to change the register tile shape while holding input precision and timing methodology fixed. Compare DRAM and L2 traffic, registers per thread, achieved occupancy, and time for both shapes. NVIDIA's [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) describes the reuse and occupancy tradeoffs.
