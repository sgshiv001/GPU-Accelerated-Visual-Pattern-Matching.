# Measurement notes

Recorded on an Intel Core i7-14650HX and NVIDIA GeForce RTX 5060 Laptop GPU (SM 12.0) on Windows, with driver 621.24, CUDA 13.4, and Visual Studio 2026 Build Tools. The PyTorch version and dataset split are captured in [evaluation.json](results/evaluation.json), and sample settings in [benchmark.json](results/benchmark.json).

- **Accuracy:** 20 training epochs on CPU with Adam, learning rate 0.001, and training batches of 64. Test images are evaluated after training. This single split does not establish accuracy on new writers or on larger image benchmarks.
- **Latency:** [latency.csv](results/latency.csv) reports medians of nine samples, each averaged over 30 calls after eight warm-ups per implementation. CPU timing uses one PyTorch thread. CUDA resident timing uses events around repeated launches in one stream with weights and inputs already on the device. CUDA copy-inclusive timing uses host wall time for input upload, the forward pass, output download, and stream synchronization. Host input/output arrays are pageable. The measurements exclude one-time allocation and weight upload. Launch gaps can affect resident timing, especially for small batches. Batch sizes above 360 repeat test images for timing only.
- **Parity:** Each CUDA implementation must match the PyTorch argmax and logits within `atol=5e-5, rtol=1e-4`. This compares execution of the same model; parity is independent of accuracy against labels. The CPU and GPU implementations both return logits rather than probabilities.
- **Interpretation:** The CPU baseline is single-thread PyTorch, not a tuned CPU implementation. The direct CUDA baseline is deliberately simple. The fused CUDA kernel changes both data staging and launch count, so the measured improvement cannot be assigned to either change in isolation. No claim is made about speed versus cuDNN, TensorRT, or Tensor Core implementations.

Run memory and race checks from the project root:

```powershell
compute-sanitizer --tool memcheck --error-exitcode 2 python.exe vision/benchmark.py --verify-only
compute-sanitizer --tool racecheck --error-exitcode 2 python.exe vision/benchmark.py --verify-only
```

Reproduce the Nsight Systems capture with a scratch results directory. Use the Nsight Systems CLI path if `nsys` is not on PATH:

```powershell
New-Item -ItemType Directory -Force build/vision-profile | Out-Null
Copy-Item vision/results/cnn_weights.npz build/vision-profile/
nsys profile --trace=cuda --sample=none --force-overwrite=true --output=build/vision-trace python.exe vision/benchmark.py --results build/vision-profile --batches 128 --warmup 1 --samples 1 --repeats 2
nsys stats --force-export=true --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum --format csv build/vision-trace.nsys-rep
```

The trace includes a 360-image verification batch before the timed 128-image workload. The [kernel summary](results/nsys-kernels.csv) and [copy summary](results/nsys-copies.csv) combine those launches; use them to identify kernels and transfers, and use the non-profiled benchmark CSV for performance comparisons. A useful next experiment is to isolate batch 128 in the trace, benchmark a tuned library reference, and test whether CUDA Graphs improve single-image latency.
