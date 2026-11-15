#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "kernels.cuh"

namespace {

void check(cudaError_t result, const char* call) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(call) + ": " + cudaGetErrorString(result));
    }
}

void check(cublasStatus_t result, const char* call) {
    if (result != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(call) + ": " + cublasGetStatusString(result));
    }
}

#define CHECK(call) check((call), #call)

class DeviceFloats {
public:
    explicit DeviceFloats(std::size_t count) { CHECK(cudaMalloc(&data, count * sizeof(float))); }
    ~DeviceFloats() { cudaFree(data); }
    DeviceFloats(const DeviceFloats&) = delete;
    DeviceFloats& operator=(const DeviceFloats&) = delete;
    float* data = nullptr;
};

class Event {
public:
    Event() { CHECK(cudaEventCreate(&value)); }
    ~Event() { cudaEventDestroy(value); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    cudaEvent_t value = nullptr;
};

class Blas {
public:
    Blas() { CHECK(cublasCreate(&value)); }
    ~Blas() { cublasDestroy(value); }
    Blas(const Blas&) = delete;
    Blas& operator=(const Blas&) = delete;
    cublasHandle_t value = nullptr;
};

struct Shape { int m, n, k; };
enum class Kernel { naive, shared, reg, cublas };
struct Variant { Kernel kernel; std::string_view name; };
constexpr std::array variants{
    Variant{Kernel::naive, "naive"}, Variant{Kernel::shared, "shared"},
    Variant{Kernel::reg, "register"}, Variant{Kernel::cublas, "cublas"}
};

struct Options {
    std::vector<Shape> shapes;
    std::string_view kernel = "all";
    int warmup = 5;
    int iterations = 10;
    int samples = 5;
    bool verify_only = false;
};

int parse_int(std::string_view s, int min, int max) {
    int value = 0;
    const auto result = std::from_chars(s.data(), s.data() + s.size(), value);
    if (result.ec != std::errc{} || result.ptr != s.data() + s.size() || value < min || value > max) {
        throw std::invalid_argument("Invalid integer: " + std::string(s));
    }
    return value;
}

Shape parse_shape(std::string_view s) {
    const auto a = s.find('x');
    const auto b = a == s.npos ? s.npos : s.find('x', a + 1);
    if (b == s.npos) throw std::invalid_argument("Expected a shape of the form MxNxK");
    return {parse_int(s.substr(0, a), 1, 8192), parse_int(s.substr(a + 1, b - a - 1), 1, 8192),
            parse_int(s.substr(b + 1), 1, 8192)};
}

void help() {
    std::cout << "gemm [--shape MxNxK ...] [--kernel all|naive|shared|register|cublas]\n"
                 "     [--warmup 5] [--iterations 10] [--samples 5] [--verify-only]\n"
                 "Rows are emitted as CSV. Device and verification status go to stderr.\n";
}

Options parse_options(int argc, char** argv) {
    Options opts;
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag(argv[i]);
        if (flag == "--help" || flag == "-h") {
            help();
            std::exit(0);
        }
        if (flag == "--verify-only") {
            opts.verify_only = true;
            continue;
        }
        if (++i >= argc) throw std::invalid_argument("Missing value for " + std::string(flag));
        const std::string_view arg(argv[i]);
        if (flag == "--shape") opts.shapes.push_back(parse_shape(arg));
        else if (flag == "--kernel") {
            if (arg != "all" && std::none_of(variants.begin(), variants.end(),
                                            [arg](Variant v) { return v.name == arg; })) {
                throw std::invalid_argument("Unknown kernel: " + std::string(arg));
            }
            opts.kernel = arg;
        } else if (flag == "--warmup") opts.warmup = parse_int(arg, 0, 100);
        else if (flag == "--iterations") opts.iterations = parse_int(arg, 1, 200);
        else if (flag == "--samples") opts.samples = parse_int(arg, 1, 100);
        else throw std::invalid_argument("Unknown option: " + std::string(flag));
    }
    if (opts.shapes.empty()) {
        opts.shapes = opts.verify_only
            ? std::vector<Shape>{{1, 1, 1}, {1, 33, 17}, {65, 1, 3}, {63, 65, 17}, {97, 81, 33}}
            : std::vector<Shape>{{256, 256, 256}, {511, 769, 513}, {1024, 1024, 1024}, {2048, 2048, 2048}};
    }
    return opts;
}

unsigned ceil_div(int n, int d) { return (n + d - 1) / d; }

void launch(Kernel kernel, Shape s, const float* a, const float* b, float* c,
            cublasHandle_t blas) {
    const dim3 small_grid(ceil_div(s.n, tile), ceil_div(s.m, tile));
    switch (kernel) {
    case Kernel::naive:
        gemm_naive<<<small_grid, dim3(tile, tile)>>>(a, b, c, s.m, s.n, s.k);
        break;
    case Kernel::shared:
        gemm_shared<<<small_grid, dim3(tile, tile)>>>(a, b, c, s.m, s.n, s.k);
        break;
    case Kernel::reg:
        gemm_register<<<dim3(ceil_div(s.n, block_n), ceil_div(s.m, block_m)), dim3(block_x, block_y)>>>
            (a, b, c, s.m, s.n, s.k);
        break;
    case Kernel::cublas: {
        // cuBLAS reads column-major data. The row-major C buffer has the same
        // layout as column-major C^T = B^T * A^T, so swap A and B.
        const float alpha = 1.0f, beta = 0.0f;
        CHECK(cublasGemmEx(blas, CUBLAS_OP_N, CUBLAS_OP_N, s.n, s.m, s.k, &alpha,
                           b, CUDA_R_32F, s.n, a, CUDA_R_32F, s.k, &beta,
                           c, CUDA_R_32F, s.n, CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT));
        break;
    }
    }
    CHECK(cudaPeekAtLastError());
}

// Fixed integer PRNG and binary fractions keep inputs consistent across standard libraries.
void fill_input(std::vector<float>& values, std::uint32_t& seed) {
    for (float& value : values) {
        seed = 1664525u * seed + 1013904223u;
        value = static_cast<float>(seed >> 8) * (1.0f / 16777216.0f) - 0.5f;
    }
}

struct Point { std::size_t index; double expected; };

std::vector<Point> cpu_reference(Shape s, const std::vector<float>& a,
                                 const std::vector<float>& b, bool full) {
    const std::size_t count = std::size_t(s.m) * s.n;
    std::vector<Point> ref;
    const auto add = [&](std::size_t i) {
        const auto row = i / s.n, col = i % s.n;
        double sum = 0;
        for (int p = 0; p < s.k; ++p) sum += double(a[row * s.k + p]) * b[std::size_t(p) * s.n + col];
        ref.push_back({i, sum});
    };
    if (full) {
        ref.reserve(count);
        for (std::size_t i = 0; i < count; ++i) add(i);
    } else {
        for (const int row : {0, s.m / 2, s.m - 1}) {
            for (const int col : {0, s.n / 2, s.n - 1}) add(std::size_t(row) * s.n + col);
        }
        std::uint32_t seed = 7421;
        for (int j = 0; j < 128; ++j) {
            seed = 1664525u * seed + 1013904223u;
            // Use the high bits so power-of-two matrix sizes do not reuse only
            // the low bits of the generator.
            add((std::uint64_t(seed) * count) >> 32);
        }
    }
    return ref;
}

double verify(Shape s, const std::vector<float>& got, const std::vector<Point>& cpu,
              const std::vector<float>* blas, Variant variant) {
    double max_cpu_error = 0;
    const auto check_element = [&](std::size_t i, double expected, const char* reference) {
        const double error = std::abs(got[i] - expected);
        if (!std::isfinite(got[i]) || !std::isfinite(expected) ||
            error > 2e-4 + 1e-4 * std::abs(expected)) {
            throw std::runtime_error(std::string(variant.name) + " disagrees with " + reference +
                " at (" + std::to_string(i / s.n) + "," + std::to_string(i % s.n) + ")" +
                ": got " + std::to_string(got[i]) + ", expected " + std::to_string(expected));
        }
        return error;
    };
    for (const auto& p : cpu) {
        max_cpu_error = std::max(max_cpu_error, check_element(p.index, p.expected, "CPU FP64"));
    }
    if (blas) {
        for (std::size_t i = 0; i < got.size(); ++i) check_element(i, (*blas)[i], "cuBLAS");
    }
    return max_cpu_error;
}

struct Timing { double median, minimum, maximum; };

template <class Launch>
Timing measure(const Options& opts, Launch run) {
    Event start, stop;
    for (int i = 0; i < opts.warmup; ++i) run();
    CHECK(cudaDeviceSynchronize());

    std::vector<double> ms;
    for (int i = 0; i < opts.samples; ++i) {
        CHECK(cudaEventRecord(start.value));
        for (int j = 0; j < opts.iterations; ++j) run();
        CHECK(cudaEventRecord(stop.value));
        CHECK(cudaEventSynchronize(stop.value));
        float elapsed = 0;
        CHECK(cudaEventElapsedTime(&elapsed, start.value, stop.value));
        if (elapsed <= 0) throw std::runtime_error("GPU timer did not advance; increase --iterations");
        ms.push_back(double(elapsed) / opts.iterations);
    }
    std::sort(ms.begin(), ms.end());
    return {(ms[(ms.size() - 1) / 2] + ms[ms.size() / 2]) / 2, ms.front(), ms.back()};
}

void run_shape(Shape s, const Options& opts, const Blas& blas) {
    std::vector<float> a(std::size_t(s.m) * s.k), b(std::size_t(s.k) * s.n),
                       out(std::size_t(s.m) * s.n);
    std::uint32_t seed = 1337;
    fill_input(a, seed);
    fill_input(b, seed);
    const bool full_cpu = std::uint64_t(s.m) * s.n * s.k <= 5'000'000;
    const auto cpu = cpu_reference(s, a, b, full_cpu);
    DeviceFloats da(a.size()), db(b.size()), dc(out.size());
    CHECK(cudaMemcpy(da.data, a.data(), a.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(db.data, b.data(), b.size() * sizeof(float), cudaMemcpyHostToDevice));

    const auto read = [&](Kernel kernel, std::vector<float>& dest) {
        // Poison the output to catch unwritten cells during full comparisons.
        CHECK(cudaMemset(dc.data, 0xff, dest.size() * sizeof(float)));
        launch(kernel, s, da.data, db.data, dc.data, blas.value);
        CHECK(cudaMemcpy(dest.data(), dc.data, dest.size() * sizeof(float), cudaMemcpyDeviceToHost));
    };

    std::vector<float> blas_reference;
    if (!full_cpu) {
        blas_reference.resize(out.size());
        read(Kernel::cublas, blas_reference);
        verify(s, blas_reference, cpu, nullptr, variants.back());
    }
    for (const auto variant : variants) {
        if (opts.kernel != "all" && opts.kernel != variant.name) continue;
        read(variant.kernel, out);
        const auto* ref = full_cpu || variant.kernel == Kernel::cublas ? nullptr : &blas_reference;
        const double error = verify(s, out, cpu, ref, variant);
        if (opts.verify_only) {
            std::cerr << "PASS " << s.m << 'x' << s.n << 'x' << s.k << ' ' << variant.name
                      << " (max CPU absolute error " << error << ")\n";
            continue;
        }
        const auto timing = measure(opts, [&] { launch(variant.kernel, s, da.data, db.data, dc.data, blas.value); });
        const double tflops = (2.0 * s.m * s.n * s.k) / (timing.median * 1e9);
        std::cout << s.m << ',' << s.n << ',' << s.k << ',' << variant.name << ',' << std::fixed
                  << std::setprecision(6) << timing.median << ',' << timing.minimum << ','
                  << timing.maximum << ',' << tflops << ',' << std::scientific << error << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto opts = parse_options(argc, argv);
        int device = 0;
        CHECK(cudaGetDevice(&device));
        cudaDeviceProp props{};
        CHECK(cudaGetDeviceProperties(&props, device));
        std::cerr << props.name << " (SM " << props.major << '.' << props.minor << ")"
                  << ", CUDA " << CUDART_VERSION / 1000 << '.' << CUDART_VERSION % 1000 / 10 << '\n';
        const Blas blas;
        if (!opts.verify_only) {
            std::cout << "m,n,k,kernel,median_ms,min_ms,max_ms,tflops,max_cpu_abs_error\n";
        }
        for (const auto shape : opts.shapes) run_shape(shape, opts, blas);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
