#ifndef CIRCLE_H
#define CIRCLE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Structure-of-arrays circle state. Each attribute is its own contiguous,
 * cache-line aligned array, so consecutive threads (GPU) or loop iterations
 * (CPU) touch consecutive addresses. Positions and velocities are stored as
 * interleaved (x, y) pairs, which is exactly a float2 array on the device.
 */
typedef struct {
    float    *pos;      /* 2 * capacity floats: x0, y0, x1, y1, ... */
    float    *vel;      /* 2 * capacity floats */
    float    *radius;
    float    *invMass;
    uint32_t *color;    /* packed RGBA8, R in the lowest byte */
    int       capacity;
} CircleArrays;

/* One instance in the render VBO: 16 bytes, so the GPU writes it with a single float4 store. */
typedef struct {
    float    x, y;
    float    radius;
    uint32_t color;
} CircleInstance;

typedef struct {
    float gravity;
    float damping;      /* per-1/60 s velocity multiplier */
    float elasticity;
    float width;
    float height;
    int   substeps;     /* sub-steps per step; more sub-steps converge stacks faster than more iterations */
    int   iterations;   /* Jacobi iterations per sub-step */
} SimParams;

typedef struct {
    float    maxPenetration;  /* deepest overlap seen in the last solver iteration (px) */
    float    kineticEnergy;
    uint32_t contacts;        /* body-contact pairs in the last iteration */
} SimStats;

/* Host-side circle storage */
int   circles_alloc(CircleArrays *c, int capacity);
void  circles_free(CircleArrays *c);
/* Fills `count` circles with a reproducible random layout. Radii shrink as
   the count grows so the circles fit in the box. Returns the largest radius. */
float circles_init_random(CircleArrays *c, int count, float width, float height, uint32_t seed);
void  circles_pack_instances(const CircleArrays *c, int count, CircleInstance *out);

/* CPU solver (uniform-grid broad phase, optionally multi-threaded with OpenMP) */
typedef struct CpuSolver CpuSolver;

CpuSolver *cpu_solver_create(int capacity);
void       cpu_solver_destroy(CpuSolver *s);
/* Must be called whenever the radius distribution or bounds change. */
int        cpu_solver_configure_grid(CpuSolver *s, float width, float height, float maxRadius);
/* useGrid = 0 tests every pair (O(n^2)), kept as a reference and benchmark baseline. */
void       cpu_solver_step(CpuSolver *s, CircleArrays *c, int count, const SimParams *params,
                           float dt, int useGrid, int numThreads, SimStats *stats);
int        cpu_max_threads(void);

#ifdef __cplusplus
}

/* GPU solver (C++ only; implemented in physics_gpu.cu, or physics_gpu_stub.cpp without CUDA) */
enum GpuOutput {
    GPU_OUTPUT_NONE,   /* keep results on the device only (benchmarking) */
    GPU_OUTPUT_HOST,   /* copy instance data into pinned host memory */
    GPU_OUTPUT_VBO     /* write instance data straight into the registered GL buffer */
};

struct GpuFrameStats {
    SimStats sim;
    float    stepMs;   /* measured with CUDA events */
};

/* Returns false when no CUDA device is usable. With preferGLDevice, picks the
   device driving the current OpenGL context (requires a current context). */
bool  physics_gpu_init(int capacity, bool preferGLDevice);
void  physics_gpu_shutdown();
const char *physics_gpu_device_name();
/* Registers the GL instance buffer for interop. Returns false if interop is unavailable. */
bool  physics_gpu_register_vbo(unsigned int vbo);
void  physics_gpu_upload(const CircleArrays *c, int count, float maxRadius, float width, float height);
void  physics_gpu_download(CircleArrays *c, int count);
void  physics_gpu_step(const SimParams *params, float dt, GpuOutput output);
/* Valid after a step with GPU_OUTPUT_HOST. */
const CircleInstance *physics_gpu_host_instances();
/* Non-blocking: returns true and fills `out` when a newer finished step is available. */
bool  physics_gpu_poll_stats(GpuFrameStats *out);
void  physics_gpu_synchronize();
#endif

#endif /* CIRCLE_H */
