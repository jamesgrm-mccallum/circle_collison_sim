// physics_gpu.cu
// CUDA port of the solver in physics_cpu.c, using the same per-body math (contact.h).
//
// Each sub-step:
//   1. k_integrate_hash   integrate, then compute each circle's grid cell (one thread per circle)
//   2. cub radix sort     sort (cell, index) pairs, only over the bits the cell ids use
//   3. k_reorder          find each cell's [start, end) range (shared memory) and gather
//                         every attribute into cell order, so neighbours sit next to each
//                         other in memory and the solver's loads are coalesced/cache friendly
//   4. k_solve x N        Jacobi iterations, one thread per circle, reading the 3x3 cells
//                         around it and writing only its own state: no atomics on bodies
// The whole frame is captured once into a CUDA Graph and replayed, which removes the
// per-launch CPU overhead of ~10 kernels per sub-step.

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cub/device/device_radix_sort.cuh>

#include <cstdio>
#include <cstring>

#include "circle.h"
#include "contact.h"

namespace {

constexpr int kBlockSize = 256;
constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = kBlockSize / kWarpSize;
constexpr uint32_t kEmptyCell = 0xFFFFFFFFu;

bool cuda_ok(cudaError_t err, const char *what)
{
    if (err == cudaSuccess) return true;
    std::fprintf(stderr, "CUDA error in %s: %s\n", what, cudaGetErrorString(err));
    return false;
}

#define CUDA_CHECK(call) cuda_ok((call), #call)

struct Grid {
    float invCellSize;
    int   width;
    int   height;
};

// Everything a captured graph bakes in. Compared with memcmp, so it is
// zero-initialised before being filled.
struct FrameConfig {
    SimParams params;
    float     dt;
    int       count;
    int       posCur;
    int       attrCur;
};

struct GraphEntry {
    cudaGraphExec_t exec = nullptr;
    FrameConfig     config;
    int             endPosCur = 0;
    int             endAttrCur = 0;
};

struct GpuState {
    bool initialized = false;
    char deviceName[320] = "none";
    int  capacity = 0;
    int  count = 0;

    cudaStream_t stream = nullptr;

    // Structure-of-arrays state, double buffered. Positions/velocities ping-pong
    // every solver iteration; the other attributes only move during the reorder.
    float2   *pos[2] = {};
    float2   *vel[2] = {};
    float    *radius[2] = {};
    float    *invMass[2] = {};
    uint32_t *color[2] = {};
    int posCur = 0;
    int attrCur = 0;

    // Spatial hash
    uint32_t *keys[2] = {};
    uint32_t *idx[2] = {};
    void     *sortTemp = nullptr;
    size_t    sortTempBytes = 0;
    uint32_t *cellStart = nullptr;
    uint32_t *cellEnd = nullptr;
    int       cellCapacity = 0;
    int       numCells = 0;
    int       keyBits = 1;
    Grid      grid = {1.0f, 1, 1};

    // Output
    float4               *dInstances = nullptr;
    CircleInstance       *hInstances = nullptr;   // pinned, for fast async D2H copies
    cudaGraphicsResource *vboResource = nullptr;

    // Statistics, double buffered so the host never waits on the GPU to read them
    SimStats    *dStats = nullptr;
    SimStats    *hStats[2] = {};                   // pinned
    cudaEvent_t  evStart[2] = {};
    cudaEvent_t  evStop[2] = {};
    cudaEvent_t  evDone[2] = {};
    bool         pending[2] = {};
    unsigned     frame = 0;

    // One cached graph per starting buffer parity
    GraphEntry graphs[4];
    bool       graphsDisabled = false;
};

GpuState g;

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

__device__ __forceinline__ int cell_coord(float v, float invCellSize, int n)
{
    int c = static_cast<int>(v * invCellSize);
    return min(max(c, 0), n - 1);
}

__device__ __forceinline__ float warp_reduce_sum(float v)
{
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xFFFFFFFFu, v, offset);
    return v;
}

__device__ __forceinline__ float warp_reduce_max(float v)
{
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1)
        v = fmaxf(v, __shfl_down_sync(0xFFFFFFFFu, v, offset));
    return v;
}

__device__ __forceinline__ unsigned warp_reduce_sum(unsigned v)
{
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xFFFFFFFFu, v, offset);
    return v;
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(kBlockSize)
k_integrate_hash(float2 *__restrict__ pos, float2 *__restrict__ vel,
                 const float *__restrict__ radius,
                 uint32_t *__restrict__ keys, uint32_t *__restrict__ idx,
                 int n, SimParams params, float h, float dampingStep, Grid grid)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        float2 p = pos[i];
        float2 v = vel[i];
        integrate_body(&p.x, &p.y, &v.x, &v.y, radius[i], params.gravity, dampingStep, h,
                       params.width, params.height, params.elasticity);
        pos[i] = p;
        vel[i] = v;

        int cx = cell_coord(p.x, grid.invCellSize, grid.width);
        int cy = cell_coord(p.y, grid.invCellSize, grid.height);
        keys[i] = static_cast<uint32_t>(cy * grid.width + cx);
        idx[i] = static_cast<uint32_t>(i);
    }
}

// Finds each cell's [start, end) range in the sorted keys and gathers all
// attributes into sorted order. Each thread compares its key with the previous
// thread's key; the block stages keys in shared memory so every key is read from
// global memory once instead of twice.
__global__ void __launch_bounds__(kBlockSize)
k_reorder(const uint32_t *__restrict__ sortedKeys, const uint32_t *__restrict__ sortedIdx, int n,
          uint32_t *__restrict__ cellStart, uint32_t *__restrict__ cellEnd,
          const float2 *__restrict__ posIn, const float2 *__restrict__ velIn,
          const float *__restrict__ radiusIn, const float *__restrict__ invMassIn,
          const uint32_t *__restrict__ colorIn,
          float2 *__restrict__ posOut, float2 *__restrict__ velOut,
          float *__restrict__ radiusOut, float *__restrict__ invMassOut,
          uint32_t *__restrict__ colorOut)
{
    __shared__ uint32_t sKeys[kBlockSize + 1];

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t key = 0;
    if (i < n) {
        key = sortedKeys[i];
        sKeys[threadIdx.x + 1] = key;
        if (threadIdx.x == 0 && i > 0) sKeys[0] = sortedKeys[i - 1];
    }
    __syncthreads();

    if (i >= n) return;

    if (i == 0 || key != sKeys[threadIdx.x]) {
        cellStart[key] = static_cast<uint32_t>(i);
        if (i > 0) cellEnd[sKeys[threadIdx.x]] = static_cast<uint32_t>(i);
    }
    if (i == n - 1) cellEnd[key] = static_cast<uint32_t>(n);

    // Gather: scattered reads once per sub-step, coalesced writes, and from then
    // on every solver iteration reads neighbours from nearby addresses.
    const uint32_t src = sortedIdx[i];
    posOut[i] = posIn[src];
    velOut[i] = velIn[src];
    radiusOut[i] = radiusIn[src];
    invMassOut[i] = invMassIn[src];
    colorOut[i] = colorIn[src];
}

// One Jacobi iteration, one thread per circle. kSleep is set on the last
// iteration of a sub-step and kStats on the last iteration of the frame.
template <bool kSleep, bool kStats>
__global__ void __launch_bounds__(kBlockSize)
k_solve(const float2 *__restrict__ posIn, const float2 *__restrict__ velIn,
        const float *__restrict__ radius, const float *__restrict__ invMass,
        const uint32_t *__restrict__ cellStart, const uint32_t *__restrict__ cellEnd,
        float2 *__restrict__ posOut, float2 *__restrict__ velOut,
        int n, SimParams params, float invH, Grid grid, SimStats *__restrict__ stats)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    float maxPen = 0.0f;
    float kinetic = 0.0f;
    unsigned contacts = 0;

    if (i < n) {
        float2 p = posIn[i];
        float2 v = velIn[i];
        const float r = radius[i];
        const float w = invMass[i];

        ContactAccum acc;
        contact_accum_reset(&acc);

        const int cx = cell_coord(p.x, grid.invCellSize, grid.width);
        const int cy = cell_coord(p.y, grid.invCellSize, grid.height);
        for (int ny = max(cy - 1, 0); ny <= min(cy + 1, grid.height - 1); ny++) {
            for (int nx = max(cx - 1, 0); nx <= min(cx + 1, grid.width - 1); nx++) {
                const int cell = ny * grid.width + nx;
                const uint32_t start = cellStart[cell];
                if (start == kEmptyCell) continue;
                const uint32_t end = cellEnd[cell];
                for (uint32_t j = start; j < end; j++) {
                    if (j == static_cast<uint32_t>(i)) continue;
                    const float2 q = posIn[j];
                    const float2 u = velIn[j];
                    contact_accumulate(&acc, i, p.x, p.y, v.x, v.y, r, w,
                                       static_cast<int>(j), q.x, q.y, u.x, u.y, radius[j], invMass[j],
                                       params.elasticity, invH);
                }
            }
        }

        contact_apply(&acc, &p.x, &p.y, &v.x, &v.y);
        apply_walls(&p.x, &p.y, &v.x, &v.y, r, params.width, params.height, params.elasticity);
        if (kSleep) sleep_body(&v.x, &v.y);

        posOut[i] = p;
        velOut[i] = v;

        if (kStats) {
            maxPen = acc.maxPenetration;
            contacts = static_cast<unsigned>(acc.contacts);
            if (w > 0.0f) kinetic = 0.5f * (v.x * v.x + v.y * v.y) / w;
        }
    }

    if (kStats) {
        // Warp shuffles, then one shared-memory slot per warp, then one atomic
        // per block instead of one per thread.
        __shared__ float    sMax[kWarpsPerBlock];
        __shared__ float    sKinetic[kWarpsPerBlock];
        __shared__ unsigned sContacts[kWarpsPerBlock];

        const int lane = threadIdx.x % kWarpSize;
        const int warp = threadIdx.x / kWarpSize;

        maxPen = warp_reduce_max(maxPen);
        kinetic = warp_reduce_sum(kinetic);
        contacts = warp_reduce_sum(contacts);
        if (lane == 0) {
            sMax[warp] = maxPen;
            sKinetic[warp] = kinetic;
            sContacts[warp] = contacts;
        }
        __syncthreads();

        if (warp == 0) {
            maxPen = lane < kWarpsPerBlock ? sMax[lane] : 0.0f;
            kinetic = lane < kWarpsPerBlock ? sKinetic[lane] : 0.0f;
            contacts = lane < kWarpsPerBlock ? sContacts[lane] : 0u;
            maxPen = warp_reduce_max(maxPen);
            kinetic = warp_reduce_sum(kinetic);
            contacts = warp_reduce_sum(contacts);
            if (lane == 0) {
                // Non-negative floats order the same as their bit patterns as ints
                atomicMax(reinterpret_cast<int *>(&stats->maxPenetration), __float_as_int(maxPen));
                atomicAdd(&stats->kineticEnergy, kinetic);
                atomicAdd(&stats->contacts, contacts);
            }
        }
    }
}

// Packs render data as one 16-byte float4 per circle: a single vectorised,
// coalesced store per thread, matching the VBO layout (x, y, radius, rgba8).
__global__ void __launch_bounds__(kBlockSize)
k_write_instances(const float2 *__restrict__ pos, const float *__restrict__ radius,
                  const uint32_t *__restrict__ color, float4 *__restrict__ out, int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        const float2 p = pos[i];
        out[i] = make_float4(p.x, p.y, radius[i], __uint_as_float(color[i]));
    }
}

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------

int blocks_for(int n)
{
    return (n + kBlockSize - 1) / kBlockSize;
}

void destroy_graphs()
{
    for (GraphEntry &e : g.graphs) {
        if (e.exec) cudaGraphExecDestroy(e.exec);
        e = GraphEntry();
    }
}

// Enqueues one full frame (all sub-steps) on the stream, starting from the given
// buffer indices. Returns the indices the data ends up in.
void enqueue_frame(const SimParams &params, float dt, int posCur, int attrCur,
                   int *endPosCur, int *endAttrCur)
{
    const int n = g.count;
    const int blocks = blocks_for(n);
    const int substeps = params.substeps > 0 ? params.substeps : 1;
    const int iterations = params.iterations > 0 ? params.iterations : 1;
    const float h = dt / static_cast<float>(substeps);
    const float invH = 1.0f / h;
    const float dampingStep = powf(params.damping, h * 60.0f);

    cudaMemsetAsync(g.dStats, 0, sizeof(SimStats), g.stream);

    for (int sub = 0; sub < substeps; sub++) {
        k_integrate_hash<<<blocks, kBlockSize, 0, g.stream>>>(
            g.pos[posCur], g.vel[posCur], g.radius[attrCur], g.keys[0], g.idx[0],
            n, params, h, dampingStep, g.grid);

        cub::DoubleBuffer<uint32_t> keys(g.keys[0], g.keys[1]);
        cub::DoubleBuffer<uint32_t> values(g.idx[0], g.idx[1]);
        cub::DeviceRadixSort::SortPairs(g.sortTemp, g.sortTempBytes, keys, values, n,
                                        0, g.keyBits, g.stream);

        cudaMemsetAsync(g.cellStart, 0xFF, sizeof(uint32_t) * static_cast<size_t>(g.numCells), g.stream);

        const int posDst = 1 - posCur;
        const int attrDst = 1 - attrCur;
        k_reorder<<<blocks, kBlockSize, 0, g.stream>>>(
            keys.Current(), values.Current(), n, g.cellStart, g.cellEnd,
            g.pos[posCur], g.vel[posCur], g.radius[attrCur], g.invMass[attrCur], g.color[attrCur],
            g.pos[posDst], g.vel[posDst], g.radius[attrDst], g.invMass[attrDst], g.color[attrDst]);
        posCur = posDst;
        attrCur = attrDst;

        for (int iter = 0; iter < iterations; iter++) {
            const bool lastIter = iter == iterations - 1;
            const bool lastOfFrame = lastIter && sub == substeps - 1;
            const int out = 1 - posCur;
            if (lastOfFrame) {
                k_solve<true, true><<<blocks, kBlockSize, 0, g.stream>>>(
                    g.pos[posCur], g.vel[posCur], g.radius[attrCur], g.invMass[attrCur],
                    g.cellStart, g.cellEnd, g.pos[out], g.vel[out], n, params, invH, g.grid, g.dStats);
            } else if (lastIter) {
                k_solve<true, false><<<blocks, kBlockSize, 0, g.stream>>>(
                    g.pos[posCur], g.vel[posCur], g.radius[attrCur], g.invMass[attrCur],
                    g.cellStart, g.cellEnd, g.pos[out], g.vel[out], n, params, invH, g.grid, nullptr);
            } else {
                k_solve<false, false><<<blocks, kBlockSize, 0, g.stream>>>(
                    g.pos[posCur], g.vel[posCur], g.radius[attrCur], g.invMass[attrCur],
                    g.cellStart, g.cellEnd, g.pos[out], g.vel[out], n, params, invH, g.grid, nullptr);
            }
            posCur = out;
        }
    }

    *endPosCur = posCur;
    *endAttrCur = attrCur;
}

// Captures the frame into a CUDA graph the first time a configuration is seen
// (or after a parameter change), then replays it. Falls back to direct launches
// if capture is not supported.
void run_frame(const SimParams &params, float dt)
{
    FrameConfig config;
    std::memset(&config, 0, sizeof(config));
    config.params = params;
    config.dt = dt;
    config.count = g.count;
    config.posCur = g.posCur;
    config.attrCur = g.attrCur;

    GraphEntry &entry = g.graphs[g.posCur * 2 + g.attrCur];

    if (!g.graphsDisabled && (!entry.exec || std::memcmp(&entry.config, &config, sizeof(config)) != 0)) {
        cudaGraph_t graph = nullptr;
        int endPos = 0, endAttr = 0;
        bool captured = CUDA_CHECK(cudaStreamBeginCapture(g.stream, cudaStreamCaptureModeThreadLocal));
        if (captured) {
            enqueue_frame(params, dt, g.posCur, g.attrCur, &endPos, &endAttr);
            captured = CUDA_CHECK(cudaStreamEndCapture(g.stream, &graph)) && graph != nullptr;
        }

        bool ready = false;
        if (captured && entry.exec) {
            cudaGraphExecUpdateResultInfo info;
            ready = cudaGraphExecUpdate(entry.exec, graph, &info) == cudaSuccess;
            if (!ready) {
                cudaGetLastError();
                cudaGraphExecDestroy(entry.exec);
                entry.exec = nullptr;
            }
        }
        if (captured && !ready) {
            ready = CUDA_CHECK(cudaGraphInstantiate(&entry.exec, graph, 0));
        }
        if (graph) cudaGraphDestroy(graph);

        if (ready) {
            entry.config = config;
            entry.endPosCur = endPos;
            entry.endAttrCur = endAttr;
        } else {
            std::fprintf(stderr, "CUDA graph capture unavailable, using direct launches\n");
            cudaGetLastError();
            destroy_graphs();
            g.graphsDisabled = true;
        }
    }

    if (!g.graphsDisabled && entry.exec) {
        CUDA_CHECK(cudaGraphLaunch(entry.exec, g.stream));
        g.posCur = entry.endPosCur;
        g.attrCur = entry.endAttrCur;
    } else {
        enqueue_frame(params, dt, g.posCur, g.attrCur, &g.posCur, &g.attrCur);
        CUDA_CHECK(cudaGetLastError());
    }
}

void write_instances(float4 *out)
{
    k_write_instances<<<blocks_for(g.count), kBlockSize, 0, g.stream>>>(
        g.pos[g.posCur], g.radius[g.attrCur], g.color[g.attrCur], out, g.count);
}

bool ensure_cell_capacity(int numCells)
{
    if (numCells <= g.cellCapacity) return true;
    cudaFree(g.cellStart);
    cudaFree(g.cellEnd);
    g.cellStart = g.cellEnd = nullptr;
    g.cellCapacity = 0;
    if (!CUDA_CHECK(cudaMalloc(&g.cellStart, sizeof(uint32_t) * static_cast<size_t>(numCells))) ||
        !CUDA_CHECK(cudaMalloc(&g.cellEnd, sizeof(uint32_t) * static_cast<size_t>(numCells)))) {
        return false;
    }
    g.cellCapacity = numCells;
    return true;
}

} // namespace

bool physics_gpu_init(int capacity, bool preferGLDevice)
{
    if (g.initialized) return true;

    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
        cudaGetLastError();
        std::fprintf(stderr, "No CUDA device available; GPU modes disabled\n");
        return false;
    }

    // On multi-GPU systems interop only works with the device driving the GL context.
    int device = 0;
    if (preferGLDevice) {
        unsigned int glDeviceCount = 0;
        int glDevices[4];
        if (cudaGLGetDevices(&glDeviceCount, glDevices, 4, cudaGLDeviceListAll) == cudaSuccess &&
            glDeviceCount > 0) {
            device = glDevices[0];
        } else {
            cudaGetLastError();
        }
    }
    if (!CUDA_CHECK(cudaSetDevice(device))) return false;

    cudaDeviceProp props;
    if (CUDA_CHECK(cudaGetDeviceProperties(&props, device))) {
        std::snprintf(g.deviceName, sizeof(g.deviceName), "%s (sm_%d%d)", props.name, props.major, props.minor);
    }

    g.capacity = capacity;
    const size_t n = static_cast<size_t>(capacity);
    bool ok = CUDA_CHECK(cudaStreamCreateWithFlags(&g.stream, cudaStreamNonBlocking));
    for (int b = 0; b < 2 && ok; b++) {
        ok = CUDA_CHECK(cudaMalloc(&g.pos[b], sizeof(float2) * n)) &&
             CUDA_CHECK(cudaMalloc(&g.vel[b], sizeof(float2) * n)) &&
             CUDA_CHECK(cudaMalloc(&g.radius[b], sizeof(float) * n)) &&
             CUDA_CHECK(cudaMalloc(&g.invMass[b], sizeof(float) * n)) &&
             CUDA_CHECK(cudaMalloc(&g.color[b], sizeof(uint32_t) * n)) &&
             CUDA_CHECK(cudaMalloc(&g.keys[b], sizeof(uint32_t) * n)) &&
             CUDA_CHECK(cudaMalloc(&g.idx[b], sizeof(uint32_t) * n)) &&
             CUDA_CHECK(cudaEventCreate(&g.evStart[b])) &&
             CUDA_CHECK(cudaEventCreate(&g.evStop[b])) &&
             CUDA_CHECK(cudaEventCreateWithFlags(&g.evDone[b], cudaEventDisableTiming)) &&
             CUDA_CHECK(cudaMallocHost(&g.hStats[b], sizeof(SimStats)));
    }
    if (ok) {
        // Size the sort's scratch space once for the worst case, so nothing is
        // allocated per frame (which would also break graph capture).
        cub::DoubleBuffer<uint32_t> keys(g.keys[0], g.keys[1]);
        cub::DoubleBuffer<uint32_t> values(g.idx[0], g.idx[1]);
        ok = CUDA_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, g.sortTempBytes, keys, values,
                                                        capacity, 0, 32)) &&
             CUDA_CHECK(cudaMalloc(&g.sortTemp, g.sortTempBytes)) &&
             CUDA_CHECK(cudaMalloc(&g.dInstances, sizeof(float4) * n)) &&
             CUDA_CHECK(cudaMallocHost(&g.hInstances, sizeof(CircleInstance) * n)) &&
             CUDA_CHECK(cudaMalloc(&g.dStats, sizeof(SimStats)));
    }

    g.initialized = true;   // lets shutdown free a partial allocation
    if (!ok) {
        physics_gpu_shutdown();
        return false;
    }
    return true;
}

void physics_gpu_shutdown()
{
    if (!g.initialized) return;
    if (g.stream) cudaStreamSynchronize(g.stream);
    destroy_graphs();
    if (g.vboResource) cudaGraphicsUnregisterResource(g.vboResource);
    for (int b = 0; b < 2; b++) {
        cudaFree(g.pos[b]);
        cudaFree(g.vel[b]);
        cudaFree(g.radius[b]);
        cudaFree(g.invMass[b]);
        cudaFree(g.color[b]);
        cudaFree(g.keys[b]);
        cudaFree(g.idx[b]);
        if (g.evStart[b]) cudaEventDestroy(g.evStart[b]);
        if (g.evStop[b]) cudaEventDestroy(g.evStop[b]);
        if (g.evDone[b]) cudaEventDestroy(g.evDone[b]);
        cudaFreeHost(g.hStats[b]);
    }
    cudaFree(g.sortTemp);
    cudaFree(g.cellStart);
    cudaFree(g.cellEnd);
    cudaFree(g.dInstances);
    cudaFreeHost(g.hInstances);
    cudaFree(g.dStats);
    if (g.stream) cudaStreamDestroy(g.stream);
    g = GpuState();
}

const char *physics_gpu_device_name()
{
    return g.deviceName;
}

bool physics_gpu_register_vbo(unsigned int vbo)
{
    if (!g.initialized || vbo == 0) return false;
    if (g.vboResource) {
        cudaGraphicsUnregisterResource(g.vboResource);
        g.vboResource = nullptr;
    }
    if (cudaGraphicsGLRegisterBuffer(&g.vboResource, vbo, cudaGraphicsRegisterFlagsWriteDiscard) != cudaSuccess) {
        cudaGetLastError();
        g.vboResource = nullptr;
        std::fprintf(stderr, "CUDA-OpenGL interop unavailable; GPU (Interop) mode disabled\n");
        return false;
    }

    // Registration can succeed on setups where mapping still fails, so try it once.
    bool mapped = cudaGraphicsMapResources(1, &g.vboResource, g.stream) == cudaSuccess;
    if (mapped) mapped = cudaGraphicsUnmapResources(1, &g.vboResource, g.stream) == cudaSuccess;
    if (!mapped) {
        cudaGetLastError();
        cudaGraphicsUnregisterResource(g.vboResource);
        g.vboResource = nullptr;
        std::fprintf(stderr, "CUDA-OpenGL interop mapping failed; GPU (Interop) mode disabled\n");
        return false;
    }
    return true;
}

void physics_gpu_upload(const CircleArrays *c, int count, float maxRadius, float width, float height)
{
    if (!g.initialized) return;
    cudaStreamSynchronize(g.stream);
    destroy_graphs();

    g.count = count < g.capacity ? count : g.capacity;
    g.posCur = 0;
    g.attrCur = 0;

    const float cellSize = fmaxf(2.0f * maxRadius, 1.0f);
    g.grid.invCellSize = 1.0f / cellSize;
    g.grid.width = static_cast<int>(ceilf(width / cellSize));
    g.grid.height = static_cast<int>(ceilf(height / cellSize));
    if (g.grid.width < 1) g.grid.width = 1;
    if (g.grid.height < 1) g.grid.height = 1;
    g.numCells = g.grid.width * g.grid.height;

    // Sort only as many key bits as the largest cell id needs: fewer radix passes.
    g.keyBits = 1;
    while ((1u << g.keyBits) < static_cast<unsigned>(g.numCells)) g.keyBits++;

    if (!ensure_cell_capacity(g.numCells)) {
        g.count = 0;
        return;
    }

    const size_t n = static_cast<size_t>(g.count);
    CUDA_CHECK(cudaMemcpy(g.pos[0], c->pos, sizeof(float2) * n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g.vel[0], c->vel, sizeof(float2) * n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g.radius[0], c->radius, sizeof(float) * n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g.invMass[0], c->invMass, sizeof(float) * n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g.color[0], c->color, sizeof(uint32_t) * n, cudaMemcpyHostToDevice));
}

void physics_gpu_download(CircleArrays *c, int count)
{
    if (!g.initialized) return;
    cudaStreamSynchronize(g.stream);
    const size_t n = static_cast<size_t>(count < g.count ? count : g.count);
    CUDA_CHECK(cudaMemcpy(c->pos, g.pos[g.posCur], sizeof(float2) * n, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(c->vel, g.vel[g.posCur], sizeof(float2) * n, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(c->radius, g.radius[g.attrCur], sizeof(float) * n, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(c->invMass, g.invMass[g.attrCur], sizeof(float) * n, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(c->color, g.color[g.attrCur], sizeof(uint32_t) * n, cudaMemcpyDeviceToHost));
}

void physics_gpu_step(const SimParams *params, float dt, GpuOutput output)
{
    if (!g.initialized || g.count <= 0 || dt <= 0.0f) return;
    if (output == GPU_OUTPUT_VBO && !g.vboResource) output = GPU_OUTPUT_HOST;

    const int slot = static_cast<int>(g.frame & 1u);
    cudaEventRecord(g.evStart[slot], g.stream);

    run_frame(*params, dt);

    if (output == GPU_OUTPUT_VBO) {
        void *mapped = nullptr;
        size_t bytes = 0;
        if (CUDA_CHECK(cudaGraphicsMapResources(1, &g.vboResource, g.stream))) {
            if (CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(&mapped, &bytes, g.vboResource)) &&
                bytes >= sizeof(float4) * static_cast<size_t>(g.count)) {
                write_instances(static_cast<float4 *>(mapped));
            }
            CUDA_CHECK(cudaGraphicsUnmapResources(1, &g.vboResource, g.stream));
        }
    } else if (output == GPU_OUTPUT_HOST) {
        write_instances(g.dInstances);
        CUDA_CHECK(cudaMemcpyAsync(g.hInstances, g.dInstances, sizeof(float4) * static_cast<size_t>(g.count),
                                   cudaMemcpyDeviceToHost, g.stream));
    }

    cudaEventRecord(g.evStop[slot], g.stream);
    cudaMemcpyAsync(g.hStats[slot], g.dStats, sizeof(SimStats), cudaMemcpyDeviceToHost, g.stream);
    cudaEventRecord(g.evDone[slot], g.stream);
    g.pending[slot] = true;
    g.frame++;

    if (output == GPU_OUTPUT_HOST) {
        CUDA_CHECK(cudaStreamSynchronize(g.stream));
    }
}

const CircleInstance *physics_gpu_host_instances()
{
    return g.hInstances;
}

bool physics_gpu_poll_stats(GpuFrameStats *out)
{
    if (!g.initialized) return false;
    bool found = false;
    // Oldest first, so the newest finished frame wins.
    for (unsigned back = 2; back >= 1; back--) {
        if (g.frame < back) continue;
        const int slot = static_cast<int>((g.frame - back) & 1u);
        if (!g.pending[slot]) continue;
        cudaError_t status = cudaEventQuery(g.evDone[slot]);
        if (status == cudaErrorNotReady) {
            cudaGetLastError();
            continue;
        }
        if (status != cudaSuccess) {
            cuda_ok(status, "cudaEventQuery");
            g.pending[slot] = false;
            continue;
        }
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, g.evStart[slot], g.evStop[slot]);
        out->sim = *g.hStats[slot];
        out->sim.contacts /= 2;   // every contact is counted by both bodies
        out->stepMs = ms;
        g.pending[slot] = false;
        found = true;
    }
    return found;
}

void physics_gpu_synchronize()
{
    if (g.initialized) cudaStreamSynchronize(g.stream);
}
