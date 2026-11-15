# CUDA GEMM Performance Lab

How much faster can an FP32 matrix multiplication get by reusing data on the GPU? This project implements three CUDA C++ kernels for row-major `C = A × B` and benchmarks them against NVIDIA cuBLAS at the same compute precision.

On an **RTX 5060 Laptop GPU**, the register-tiled kernel took **3.898 ms** for a 2048 × 2048 × 2048 GEMM: **3.86× faster** than the naive kernel. cuBLAS took **1.613 ms**. At 256 × 256 × 256, the smaller shared-memory tile was the fastest custom kernel.

| M × N × K | Naive (ms) | Shared (ms) | Register (ms) | cuBLAS (ms) | Best custom vs. naive |
| --- | ---: | ---: | ---: | ---: | ---: |
| 256 × 256 × 256 | 0.0390 | **0.0308** | 0.0328 | 0.0207 | 1.27× |
| 511 × 769 × 513 | 0.4103 | 0.3180 | **0.1400** | 0.0724 | 2.93× |
| 1024 × 1024 × 1024 | 1.8966 | 1.4232 | **0.4877** | 0.2250 | 3.89× |
| 2048 × 2048 × 2048 | 15.0297 | 11.2919 | **3.8981** | 1.6125 | 3.86× |

See the [benchmark CSV](../results/rtx5060-laptop.csv) for throughput, timing ranges, and numerical error, and the [performance notes](../results/analysis.md) for the profiling evidence.

## Build and run

On Windows, use a CUDA-capable NVIDIA GPU, the CUDA Toolkit with cuBLAS, and a CUDA-supported MSVC C++ toolchain. This was tested with CUDA 13.4, driver 621.24, and Visual Studio 2026 Build Tools. Put the toolkit's `bin` on PATH. If Windows cannot load cuBLAS, also add the toolkit's `bin\x64` to PATH for toolkits that place DLLs there.

```powershell
.\build.cmd
.\build\gemm.exe --verify-only
.\build\gemm.exe --warmup 10 --iterations 50 --samples 7 |
    Set-Content -Encoding ascii build\gemm-results.csv
.\build\gemm.exe --shape 511x769x513 --kernel register
```

`--shape` can be repeated and uses M (A's rows), N (B's columns), and K (the reduction dimension). The program emits CSV on stdout and sends validation and device messages to stderr. Run `--help` to see the other options.

## What the kernels change

| Variant | Work per thread | What to inspect |
| --- | --- | --- |
| [Naive](../src/kernels.cuh) | One output element | Each thread reads input values from global memory as it accumulates its dot product. |
| [Shared](../src/kernels.cuh) | One output element | A block loads input tiles into shared memory and reuses those values across threads. |
| [Register](../src/kernels.cuh) | 16 output elements | Each thread keeps several sums in registers and reuses values from shared memory across its sums. |
| [cuBLAS](../src/gemm.cu) | Library-selected | `cublasGemmEx` with `CUBLAS_COMPUTE_32F_PEDANTIC` is the library reference. |

The two tiled kernels zero-fill partial input tiles and guard output stores, so rectangular dimensions do not need to be multiples of a tile. `--verify-only` checks every element against a double-precision CPU reference for the small test shapes. Larger benchmarks compare every custom output against cuBLAS and check selected entries against the CPU. Compute Sanitizer found no memory errors on two irregular test shapes and no shared-memory hazards in the tested tiled kernels.

The timings are medians of seven samples, each a batch of 50 launches measured with CUDA events after ten warm-up launches. Allocation, transfers, and validation run outside the timed interval. Throughput counts `2 × M × N × K` operations. This is a single run on a laptop with repeated inputs and unlocked clocks; report the GPU, precision, shapes, and timing method when comparing results.

## Portfolio roadmap

Start with this GEMM project for CUDA kernel and GPU performance roles. The [project roadmap](../PORTFOLIO.md) suggests three follow-up projects and what to measure in each one.
