// Headless benchmark: times one simulation step (all sub-steps) for each solver
// configuration on the same initial state. No window or OpenGL needed.
//
//   circle_bench [count=5000] [steps=300]

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "circle.h"

namespace {

const float kWidth = 1600.0f;
const float kHeight = 900.0f;
const float kDt = 1.0f / 120.0f;

struct Result {
    std::string name;
    double msPerStep;
};

void copy_state(CircleArrays *dst, const CircleArrays *src, int n)
{
    std::memcpy(dst->pos, src->pos, sizeof(float) * 2 * n);
    std::memcpy(dst->vel, src->vel, sizeof(float) * 2 * n);
    std::memcpy(dst->radius, src->radius, sizeof(float) * n);
    std::memcpy(dst->invMass, src->invMass, sizeof(float) * n);
    std::memcpy(dst->color, src->color, sizeof(uint32_t) * n);
}

double now_ms()
{
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

double bench_cpu(const CircleArrays &initial, int n, float maxRadius, const SimParams &params,
                 int steps, bool useGrid, int threads)
{
    CircleArrays c;
    circles_alloc(&c, n);
    copy_state(&c, &initial, n);
    CpuSolver *solver = cpu_solver_create(n);
    cpu_solver_configure_grid(solver, kWidth, kHeight, maxRadius);

    for (int i = 0; i < 5; i++) cpu_solver_step(solver, &c, n, &params, kDt, useGrid, threads, nullptr);
    const double start = now_ms();
    for (int i = 0; i < steps; i++) cpu_solver_step(solver, &c, n, &params, kDt, useGrid, threads, nullptr);
    const double elapsed = now_ms() - start;

    cpu_solver_destroy(solver);
    circles_free(&c);
    return elapsed / steps;
}

double bench_gpu(const CircleArrays &initial, int n, float maxRadius, const SimParams &params, int steps)
{
    physics_gpu_upload(&initial, n, maxRadius, kWidth, kHeight);
    for (int i = 0; i < 10; i++) physics_gpu_step(&params, kDt, GPU_OUTPUT_NONE);
    physics_gpu_synchronize();

    const double start = now_ms();
    for (int i = 0; i < steps; i++) physics_gpu_step(&params, kDt, GPU_OUTPUT_NONE);
    physics_gpu_synchronize();
    return (now_ms() - start) / steps;
}

} // namespace

int main(int argc, char **argv)
{
    const int n = argc > 1 ? std::max(1, std::atoi(argv[1])) : 5000;
    const int steps = argc > 2 ? std::max(1, std::atoi(argv[2])) : 300;

    SimParams params;
    params.gravity = 100.0f;
    params.damping = 0.995f;
    params.elasticity = 0.8f;
    params.width = kWidth;
    params.height = kHeight;
    params.substeps = 8;
    params.iterations = 2;

    CircleArrays initial;
    circles_alloc(&initial, n);
    const float maxRadius = circles_init_random(&initial, n, kWidth, kHeight, 42u);

    // Let the pile form first so the timings reflect a dense, settled scene.
    {
        CpuSolver *solver = cpu_solver_create(n);
        cpu_solver_configure_grid(solver, kWidth, kHeight, maxRadius);
        for (int i = 0; i < 240; i++) {
            cpu_solver_step(solver, &initial, n, &params, kDt, 1, cpu_max_threads(), nullptr);
        }
        cpu_solver_destroy(solver);
    }

    const int threads = cpu_max_threads();
    std::printf("circles: %d   steps: %d   sub-steps x iterations: %d x %d   CPU threads: %d\n",
                n, steps, params.substeps, params.iterations, threads);

    std::vector<Result> results;
    if (n <= 10000) {
        const int bruteSteps = std::max(1, steps / 10);
        results.push_back({"CPU, all pairs O(n^2), 1 thread",
                           bench_cpu(initial, n, maxRadius, params, bruteSteps, false, 1)});
    }
    const double cpuGrid1 = bench_cpu(initial, n, maxRadius, params, steps, true, 1);
    results.push_back({"CPU, uniform grid, 1 thread", cpuGrid1});
    double cpuGridN = cpuGrid1;
    if (threads > 1) {
        cpuGridN = bench_cpu(initial, n, maxRadius, params, steps, true, threads);
        results.push_back({"CPU, uniform grid, " + std::to_string(threads) + " threads (OpenMP)", cpuGridN});
    }

    double gpu = 0.0;
    if (physics_gpu_init(n, false)) {
        gpu = bench_gpu(initial, n, maxRadius, params, steps);
        results.push_back({std::string("GPU, ") + physics_gpu_device_name(), gpu});
        physics_gpu_shutdown();
    } else {
        std::printf("(no CUDA device: GPU benchmark skipped)\n");
    }

    std::printf("\n%-48s %12s %12s\n", "solver", "ms / step", "steps / s");
    for (const Result &r : results) {
        std::printf("%-48s %12.3f %12.1f\n", r.name.c_str(), r.msPerStep, 1000.0 / r.msPerStep);
    }
    std::printf("\n");
    if (n <= 10000) {
        std::printf("grid vs all pairs (1 thread):        %.1fx\n", results[0].msPerStep / cpuGrid1);
    }
    if (threads > 1) {
        std::printf("OpenMP %2d threads vs 1 thread:       %.1fx\n", threads, cpuGrid1 / cpuGridN);
    }
    if (gpu > 0.0) {
        std::printf("GPU vs CPU grid, 1 thread:           %.1fx\n", cpuGrid1 / gpu);
        if (threads > 1) std::printf("GPU vs CPU grid, %2d threads:          %.1fx\n", threads, cpuGridN / gpu);
    }

    circles_free(&initial);
    return 0;
}
