# GPU-Accelerated Visual Pattern Matching

**Shivanshu Gupta · Personal thesis project · CUDA C++ and computer vision**

[Personal thesis report (PDF)](../docs/Thesis-Report.pdf)

Double-click **Visual Pattern Matching - CUDA Dashboard** on the Windows desktop or open [Launch Dashboard.cmd](../Launch%20Dashboard.cmd) from the project folder. Use **Predict on CUDA** to run the selected digit through the trained CNN, or select a batch size and choose **Run benchmark** to replace its saved timing on screen with a live measurement. Each live run saves its log and any benchmark files under `build/dashboard-runs/`. If the native engine has yet to be built, use **Build CUDA** in the window.

[Open the interactive demo](demo.html). It lets you browse held-out handwritten digits, inspect mistakes, and switch batch sizes to see when GPU acceleration pays off. Download or share the single HTML file; the browser replays measured results without a server or GPU.

This project compares HOG–PCA–SVM with a two-block CNN on the same [scikit-learn Digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html) split. It then runs the trained CNN on an RTX 5060 Laptop GPU with two native FP32 CUDA implementations. The fused version combines convolution, ReLU, and max pooling to avoid writing an intermediate activation tensor and reduce launches.

| Held-out result | Correct / 360 | Accuracy |
| --- | ---: | ---: |
| HOG–PCA–SVM | 336 | 93.33% |
| Two-block CNN | 347 | 96.39% |

Both CUDA paths matched all 360 PyTorch class predictions, including the CNN's 13 incorrect predictions. The maximum absolute logit difference was **8.58 × 10⁻⁶**. Compute Sanitizer reported zero memory errors and zero shared-memory hazards at batches 1, 37, and 360.

## What CUDA changed

The direct path writes all convolution outputs to global memory, then launches a second kernel for ReLU and max pooling. In the fused path, each block stages an input image or feature map in shared memory. A group of four lanes computes the four convolution values in a pooling window and exchanges maxima with warp shuffles. One lane writes the pooled result. Both variants use the same warp-reduction dense kernels, trained weights, and FP32 precision.

| Batch | PyTorch CPU (ms) | Direct CUDA, resident (ms) | Fused CUDA, resident (ms) | Fused CUDA, with copies (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0570 | 0.0864 | 0.0543 | 0.0649 |
| 8 | 0.3155 | 0.0472 | 0.0311 | 0.0686 |
| 128 | 3.2704 | 0.1223 | 0.1055 | 0.1413 |
| 512 | 12.7377 | 0.3985 | 0.3706 | 0.4534 |

The CPU had lower copy-inclusive latency for one image; the first tested batch where fused CUDA won was 8. At batch 128, fusion cut device-resident latency by **13.7%** relative to direct CUDA. Including both copies, fused CUDA took **0.141 ms**, about **23× as fast** as the single-thread PyTorch CPU baseline on this workload. The copy-inclusive result compares implementations and hardware, so the direct versus fused CUDA measurement is the better estimate of what the kernel changes accomplished. At batches 32 and 1024, fusion improved resident time but did not improve copy-inclusive time in this run.

For batch 128, the direct implementation explicitly writes 1.5 MiB of intermediate convolution output across the two stages, then reads it back during pooling. The fused implementation avoids those intermediate loads and stores. This is a count from tensor dimensions; it does not establish how many bytes went to DRAM. An Nsight Systems trace of a verification pass followed by a batch-128 workload confirms kernel and copy activity, with the second convolution stage taking the most kernel time in either path. See the [kernel trace summary](results/nsys-kernels.csv) and [measurement notes](BENCHMARK.md).

## How this relates to the thesis

The thesis compares a gradient-based classical pipeline with a two-stage CNN and asks when the GPU overcomes transfer and launch overhead. This implementation uses a fixed stratified split (seed 42; 1,437 train and 360 test), 8×8 inputs scaled by 16, and two Conv–ReLU–Pool blocks followed by Dense(128) and a ten-class output.

- **PCA:** Four 9-bin HOG cells produce 36 features, so this implementation selects enough components to retain at least 95% of training-set variance instead of requesting 40 components. This run retained 27. PCA and SVM are fitted only on the training split.
- **CNN shape:** Both 3×3 convolutions use padding of one. Two valid convolutions separated by 2×2 pooling would leave a 1×1 map before the second pool, which cannot accept it. Padding gives spatial sizes 8 → 4 → 2 after pooling.
- **Measurements:** The supplied thesis versions describe a simulated CNN result and analytical GPU estimates. The numbers here come from the trained model and the implemented kernels; the historical thesis values are not used as this project's benchmark results. The native engine uses direct cross-correlation and fusion rather than lowering convolution to GEMM.

## Reproduce on Windows

Use Python 3.14, the [Python dependencies](requirements.txt), an NVIDIA GPU, a CUDA Toolkit on PATH, and a compatible x64 Visual Studio C++ toolchain. The benchmark uses a CPU PyTorch build; CUDA inference comes from the native DLL. From the repository root:

```powershell
python -m pip install -r vision/requirements.txt
.\build.cmd vision
python vision/train.py
python vision/benchmark.py --verify-only
python vision/benchmark.py --predict-id 618
python vision/benchmark.py
python vision/make_demo.py
Start-Process .\vision\demo.html
```

The trained FP32 weights and captured evidence are included in [results](results/). The verification, prediction, and benchmark commands can run with those weights before retraining. They validate logits against PyTorch before reporting results. For a live demo, pass any held-out dataset sample ID from the browser. The native engine supports normalized NCHW 1×8×8 inputs with batches of 1–2048 and returns ten logits per image.

| File | Role |
| --- | --- |
| [model.py](model.py), [train.py](train.py) | Dataset split, HOG, PCA/SVM fit, CNN training and evaluation |
| [inference.cu](inference.cu) | Direct and fused CUDA inference, device buffers, and timing |
| [benchmark.py](benchmark.py) | Numerical parity, CPU/GPU measurements, and CSV output |
| [demo.html](demo.html) | Standalone interactive recruiter demo |

## Interview walkthrough

1. Start with the first test image: its label is **5**, the SVM predicts **9**, and the CNN predicts **5**. Use **Show mistakes** to explore failures.
2. Switch the benchmark from batch **1** to **128**. Explain why GPU event timing and host wall-clock timing answer different questions.
3. Open the fused kernel. Explain how four lanes implement a pooling window, why the layer needs no global intermediate tensor, and how PyTorch logit parity checks for subtle bugs beyond class accuracy.

**Résumé bullet:** Implemented and validated native FP32 CUDA inference for a two-block digit CNN; fused convolution, ReLU, and pooling to cut batch-128 device latency by 13.7%, matching PyTorch predictions on all 360 held-out images.
