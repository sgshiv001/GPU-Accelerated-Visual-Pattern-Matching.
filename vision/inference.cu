#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C"
#endif

namespace {
constexpr int conv1_weights = 32 * 1 * 3 * 3, conv2_weights = 64 * 32 * 3 * 3;
constexpr int fc1_weights = 128 * 256, fc2_weights = 10 * 128;
constexpr int w1 = 0, b1 = w1 + conv1_weights, w2 = b1 + 32, b2 = w2 + conv2_weights;
constexpr int w3 = b2 + 64, b3 = w3 + fc1_weights, w4 = b3 + 128, b4 = w4 + fc2_weights;
constexpr int weights_count = b4 + 10;
constexpr int threads = 128;
constexpr int max_batch = 2048;

void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

template<int Side, int In>
__device__ float conv_pixel(const float* x, const float* w, float bias, int y, int col) {
    float sum = bias;
    for (int ch = 0; ch < In; ++ch) {
        #pragma unroll
        for (int ky = -1; ky <= 1; ++ky) {
            #pragma unroll
            for (int kx = -1; kx <= 1; ++kx) {
                const int iy = y + ky, ix = col + kx;
                if (iy >= 0 && iy < Side && ix >= 0 && ix < Side) {
                    sum = fmaf(x[(ch * Side + iy) * Side + ix], w[ch * 9 + (ky + 1) * 3 + kx + 1], sum);
                }
            }
        }
    }
    return sum;
}

// Baseline: materialize the full convolution tensor in global memory.
template<int Side, int In, int Out>
__global__ void conv_direct(const float* x, const float* w, const float* bias, float* y, int batch) {
    constexpr int area = Side * Side;
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= batch * Out * area) return;
    const int item = id / (Out * area), channel = id / area % Out;
    y[id] = conv_pixel<Side, In>(x + item * In * area, w + channel * In * 9,
                                 bias[channel], id / Side % Side, id % Side);
}

template<int Side, int Out>
__global__ void relu_pool(const float* x, float* y, int batch) {
    constexpr int pooled = Side / 2;
    constexpr int area = pooled * pooled;
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= batch * Out * area) return;
    const int base = (id / area) * Side * Side + 2 * (id / pooled % pooled) * Side + 2 * (id % pooled);
    y[id] = fmaxf(0.0f, fmaxf(fmaxf(x[base], x[base + 1]), fmaxf(x[base + Side], x[base + Side + 1])));
}

// Stage one image in shared memory per block. Groups of four contiguous lanes
// compute the four distinct pixels of a pooling window, then exchange maxima.
template<int Side, int In, int Out>
__global__ void conv_pool_fused(const float* x, const float* w, const float* bias, float* y) {
    constexpr int area = Side * Side, pooled = Side / 2;
    constexpr int out_per_block = threads / area;
    static_assert(Out % out_per_block == 0);
    __shared__ float input[In * area];
    const int item = blockIdx.y;
    const int lane = threadIdx.x;
    for (int i = lane; i < In * area; i += threads) input[i] = x[item * In * area + i];
    __syncthreads();
    const int ch = blockIdx.x * out_per_block + lane / area;
    const int p = (lane % area) / 4, corner = lane % 4;
    const int oy = 2 * (p / pooled) + corner / 2, ox = 2 * (p % pooled) + corner % 2;
    float best = fmaxf(0.0f, conv_pixel<Side, In>(input, w + ch * In * 9, bias[ch], oy, ox));
    best = fmaxf(best, __shfl_xor_sync(0xffffffffu, best, 1, 4));
    best = fmaxf(best, __shfl_xor_sync(0xffffffffu, best, 2, 4));
    if (corner == 0) y[(item * Out + ch) * (area / 4) + p] = best;
}

// A warp computes one dot product; the two convolution paths share these layers.
template<int In, int Out, bool Relu>
__global__ void dense(const float* x, const float* w, const float* bias, float* y, int batch) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (warp >= batch * Out) return;  // Entire warps agree.
    const int lane = threadIdx.x % 32, item = warp / Out, unit = warp % Out;
    float sum = 0.0f;
    for (int i = lane; i < In; i += 32) sum = fmaf(x[item * In + i], w[unit * In + i], sum);
    for (int offset = 16; offset > 0; offset /= 2) sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0) {
        const float value = sum + bias[unit];
        y[warp] = Relu ? fmaxf(0.0f, value) : value;
    }
}

struct Buffer {
    float* data = nullptr;
    explicit Buffer(size_t count) { check(cudaMalloc(&data, count * sizeof(float))); }
    ~Buffer() { cudaFree(data); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};
struct Stream {
    cudaStream_t value = nullptr;
    Stream() { check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~Stream() { cudaStreamDestroy(value); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
};
struct Event {
    cudaEvent_t value = nullptr;
    Event() { check(cudaEventCreate(&value)); }
    ~Event() { cudaEventDestroy(value); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
};

int blocks(int n) { return (n + threads - 1) / threads; }
struct Context {
    const int batch;
    Buffer weight, input, conv, pool1, pool2, hidden, output;
    Stream stream;
    Event start, end;
    bool uploaded = false;

    Context(int n, const float* w) :
        batch(n), weight(weights_count), input(n * 64), conv(n * 32 * 64),
        pool1(n * 32 * 16), pool2(n * 64 * 4), hidden(n * 128), output(n * 10) {
        check(cudaMemcpy(weight.data, w, weights_count * sizeof(float), cudaMemcpyHostToDevice));
    }

    void launch(int fused) {
        const float* w = weight.data;
        const auto s = stream.value;
        if (fused) {
            conv_pool_fused<8, 1, 32><<<dim3(16, batch), threads, 0, s>>>(
                input.data, w + w1, w + b1, pool1.data);
            conv_pool_fused<4, 32, 64><<<dim3(8, batch), threads, 0, s>>>(
                pool1.data, w + w2, w + b2, pool2.data);
        } else {
            conv_direct<8, 1, 32><<<blocks(batch * 32 * 64), threads, 0, s>>>(
                input.data, w + w1, w + b1, conv.data, batch);
            relu_pool<8, 32><<<blocks(batch * 32 * 16), threads, 0, s>>>(
                conv.data, pool1.data, batch);
            conv_direct<4, 32, 64><<<blocks(batch * 64 * 16), threads, 0, s>>>(
                pool1.data, w + w2, w + b2, conv.data, batch);
            relu_pool<4, 64><<<blocks(batch * 64 * 4), threads, 0, s>>>(
                conv.data, pool2.data, batch);
        }
        dense<256, 128, true><<<blocks(batch * 128 * 32), threads, 0, s>>>(
            pool2.data, w + w3, w + b3, hidden.data, batch);
        dense<128, 10, false><<<blocks(batch * 10 * 32), threads, 0, s>>>(
            hidden.data, w + w4, w + b4, output.data, batch);
        check(cudaGetLastError());
    }
};

thread_local char error[512] = {};
template<typename F>
int status(F&& operation) noexcept {
    try {
        operation();
        error[0] = '\0';
        return 0;
    } catch (const std::exception& e) {
        std::snprintf(error, sizeof(error), "%s", e.what());
        return -1;
    }
}

void require(bool ok, const char* message) {
    if (!ok) throw std::invalid_argument(message);
}
}  // namespace

API const char* vision_error() { return error; }
API int vision_device(char* name, int capacity) {
    return status([&] {
        require(name && capacity > 0, "Invalid device name buffer");
        int device;
        cudaDeviceProp props{};
        check(cudaGetDevice(&device));
        check(cudaGetDeviceProperties(&props, device));
        std::snprintf(name, capacity, "%s (SM %d.%d)", props.name, props.major, props.minor);
    });
}
API int vision_create(int batch, const float* weights, int count, Context** out) {
    return status([&] {
        require(batch >= 1 && batch <= max_batch && weights && count == weights_count && out,
                "Invalid batch or model weights");
        *out = new Context(batch, weights);
    });
}
API void vision_destroy(Context* context) { delete context; }
API int vision_forward(Context* ctx, const float* x, float* y, int fused) {
    return status([&] {
        require(ctx && x && y && (fused == 0 || fused == 1), "Invalid forward arguments");
        const auto s = ctx->stream.value;
        check(cudaMemcpyAsync(ctx->input.data, x, ctx->batch * 64 * sizeof(float), cudaMemcpyHostToDevice, s));
        ctx->launch(fused);
        check(cudaMemcpyAsync(y, ctx->output.data, ctx->batch * 10 * sizeof(float), cudaMemcpyDeviceToHost, s));
        check(cudaStreamSynchronize(s));
        ctx->uploaded = true;
    });
}
API int vision_resident_ms(Context* ctx, int fused, int repeats, float* result) {
    return status([&] {
        require(ctx && ctx->uploaded && result && repeats > 0 && (fused == 0 || fused == 1),
                "Resident timing requires a successful forward call and valid timing arguments");
        const auto s = ctx->stream.value;
        check(cudaEventRecord(ctx->start.value, s));
        for (int i = 0; i < repeats; ++i) ctx->launch(fused);
        check(cudaEventRecord(ctx->end.value, s));
        check(cudaEventSynchronize(ctx->end.value));
        check(cudaEventElapsedTime(result, ctx->start.value, ctx->end.value));
        *result /= repeats;
    });
}
