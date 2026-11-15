# CUDA projects for NVIDIA software roles

Lead with [GPU-Accelerated Visual Pattern Matching](vision/README.md) for inference and GPU performance roles. It connects the thesis to trained models, custom CUDA kernels, correctness evidence, and a [browser demo](vision/demo.html). Use the GEMM lab as a second project to discuss lower-level tiling.

Build one project deeply enough to defend every benchmark choice. NVIDIA's [AI Developer Technology role in India](https://nvidia.wd5.myworkdayjobs.com/en-US/NVIDIAExternalCareerSite/job/AI-Developer-Technology-Engineer_JR2008156) is one example of work that connects CUDA C++ with workload analysis and GPU optimization. These are portfolio ideas, not job prerequisites.

| Project | What to build | Evidence to publish |
| --- | --- | --- |
| **1. FP32 GEMM Performance Lab** (implemented here) | Progress from one output per thread to shared-memory tiling and register blocking. Keep cuBLAS as a reference. | Validated output, raw timing data, and an explanation of why the winner changes with matrix size. |
| **2. Warp reduction and prefix scan** | Implement a reduction using warp shuffles and a two-pass scan for arrays larger than one block. Compare against CUB. | Effective GB/s and latency across sizes, correctness for partial blocks and non-power-of-two inputs, and evidence for the launch-overhead crossover. |
| **3. Fused softmax for transformer workloads** | Implement numerically stable row-wise softmax in CUDA C++ with warp and block reductions. Compare against PyTorch softmax and a separate max/exp/sum/normalize baseline. | Latency by row width and batch size, error against a high-precision reference, memory traffic, and cases where fusion loses. |
| **4. GPU pipeline with asynchronous transfers** | Process batches through a CUDA image filter. Compare serial copies with pinned memory and multiple streams. | End-to-end throughput, per-batch latency, and a Nsight Systems trace showing whether transfers overlap with compute. |

For GPU libraries or kernel performance, extend GEMM with a tile-size sweep and one measured explanation before building the reduction project. For AI inference, take the fused-softmax route. For profiling or runtime work, take the streams route and explain the timeline.

If your target is **designing GPU cores**, choose a hardware project alongside CUDA: build a small pipelined FP32 datapath or a warp scheduler simulator and verify its behavior on several workloads. CUDA C++ kernels show how you use a GPU; RTL, verification, and performance modeling give a different kind of evidence for hardware roles. NVIDIA's [GPU Architect role](https://nvidia.wd5.myworkdayjobs.com/en-US/NVIDIAExternalCareerSite/job/GPU-Architect---New-College-Grad-2026_JR2019445) illustrates that distinction.

Once you can reproduce and explain your results, a résumé bullet could read: “Implemented and benchmarked three FP32 CUDA GEMM kernels; achieved a 3.86× speedup over a naive kernel on a 2048³ workload by combining shared-memory tiling with register blocking.”
