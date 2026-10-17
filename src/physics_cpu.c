#include "circle.h"
#include "contact.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define CACHE_LINE 64

/* ---------------------------------------------------------------------------
 * Allocation
 * ------------------------------------------------------------------------ */

/* Every array starts on its own cache line, so two arrays never share a line
   and the static OpenMP partition below only shares lines at chunk borders. */
static void *alloc_aligned(size_t bytes)
{
    size_t rounded = (bytes + CACHE_LINE - 1) / CACHE_LINE * CACHE_LINE;
    if (rounded == 0) rounded = CACHE_LINE;
#if defined(_WIN32)
    return _aligned_malloc(rounded, CACHE_LINE);
#else
    return aligned_alloc(CACHE_LINE, rounded);
#endif
}

static void free_aligned(void *p)
{
#if defined(_WIN32)
    _aligned_free(p);
#else
    free(p);
#endif
}

int circles_alloc(CircleArrays *c, int capacity)
{
    memset(c, 0, sizeof(*c));
    c->pos     = (float *)alloc_aligned(sizeof(float) * 2 * (size_t)capacity);
    c->vel     = (float *)alloc_aligned(sizeof(float) * 2 * (size_t)capacity);
    c->radius  = (float *)alloc_aligned(sizeof(float) * (size_t)capacity);
    c->invMass = (float *)alloc_aligned(sizeof(float) * (size_t)capacity);
    c->color   = (uint32_t *)alloc_aligned(sizeof(uint32_t) * (size_t)capacity);
    c->capacity = capacity;
    if (!c->pos || !c->vel || !c->radius || !c->invMass || !c->color) {
        circles_free(c);
        return 0;
    }
    return 1;
}

void circles_free(CircleArrays *c)
{
    free_aligned(c->pos);
    free_aligned(c->vel);
    free_aligned(c->radius);
    free_aligned(c->invMass);
    free_aligned(c->color);
    memset(c, 0, sizeof(*c));
}

/* ---------------------------------------------------------------------------
 * Initialisation
 * ------------------------------------------------------------------------ */

static uint32_t xorshift32(uint32_t *state)
{
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static float rand_unit(uint32_t *state)
{
    return (float)(xorshift32(state) >> 8) * (1.0f / 16777216.0f);
}

float circles_init_random(CircleArrays *c, int count, float width, float height, uint32_t seed)
{
    const float kPi = 3.14159265f;
    const float kFill = 0.45f;   /* target fraction of the box covered by circles */

    /* Radii are uniform in [R/2, R], so E[r^2] = 7/12 R^2. Pick R so the
       expected total circle area is kFill of the box. */
    float maxRadius = 19.0f;
    if (count > 0) {
        float fit = sqrtf(kFill * width * height / ((float)count * kPi * (7.0f / 12.0f)));
        if (fit < maxRadius) maxRadius = fit;
    }
    if (maxRadius < 2.0f) maxRadius = 2.0f;
    const float minRadius = 0.5f * maxRadius;

    uint32_t state = seed ? seed : 0x9E3779B9u;
    for (int i = 0; i < count; i++) {
        float r = minRadius + (maxRadius - minRadius) * rand_unit(&state);
        c->radius[i] = r;
        c->pos[2 * i + 0] = r + (width - 2.0f * r) * rand_unit(&state);
        c->pos[2 * i + 1] = r + (height - 2.0f * r) * rand_unit(&state);
        c->vel[2 * i + 0] = -100.0f + 200.0f * rand_unit(&state);
        c->vel[2 * i + 1] = -100.0f + 200.0f * rand_unit(&state);
        c->invMass[i] = 1.0f / (kPi * r * r);

        uint32_t rgb = xorshift32(&state);
        c->color[i] = (rgb & 0x00FFFFFFu) | 0xFF000000u;
    }
    return maxRadius;
}

void circles_pack_instances(const CircleArrays *c, int count, CircleInstance *out)
{
    for (int i = 0; i < count; i++) {
        out[i].x = c->pos[2 * i + 0];
        out[i].y = c->pos[2 * i + 1];
        out[i].radius = c->radius[i];
        out[i].color = c->color[i];
    }
}

/* ---------------------------------------------------------------------------
 * Solver
 * ------------------------------------------------------------------------ */

struct CpuSolver {
    int    capacity;
    float *pos;          /* Jacobi output buffers, swapped with the CircleArrays ones */
    float *vel;

    /* Uniform grid, built with a counting sort each step */
    int   *cellOf;       /* cell of each body */
    int   *sortedIdx;    /* body indices grouped by cell */
    int   *cellStart;    /* numCells + 1 prefix offsets into sortedIdx */
    int   *cellCursor;
    int    cellCapacity;
    int    gridW, gridH;
    float  invCellSize;
};

CpuSolver *cpu_solver_create(int capacity)
{
    CpuSolver *s = (CpuSolver *)calloc(1, sizeof(CpuSolver));
    if (!s) return NULL;
    s->capacity  = capacity;
    s->pos       = (float *)alloc_aligned(sizeof(float) * 2 * (size_t)capacity);
    s->vel       = (float *)alloc_aligned(sizeof(float) * 2 * (size_t)capacity);
    s->cellOf    = (int *)alloc_aligned(sizeof(int) * (size_t)capacity);
    s->sortedIdx = (int *)alloc_aligned(sizeof(int) * (size_t)capacity);
    if (!s->pos || !s->vel || !s->cellOf || !s->sortedIdx) {
        cpu_solver_destroy(s);
        return NULL;
    }
    return s;
}

void cpu_solver_destroy(CpuSolver *s)
{
    if (!s) return;
    free_aligned(s->pos);
    free_aligned(s->vel);
    free_aligned(s->cellOf);
    free_aligned(s->sortedIdx);
    free_aligned(s->cellStart);
    free_aligned(s->cellCursor);
    free(s);
}

int cpu_solver_configure_grid(CpuSolver *s, float width, float height, float maxRadius)
{
    /* A cell as wide as the largest diameter guarantees every overlapping
       pair lies in the same or an adjacent cell. */
    float cellSize = fmaxf(2.0f * maxRadius, 1.0f);
    s->invCellSize = 1.0f / cellSize;
    s->gridW = (int)ceilf(width / cellSize);
    s->gridH = (int)ceilf(height / cellSize);
    if (s->gridW < 1) s->gridW = 1;
    if (s->gridH < 1) s->gridH = 1;

    int numCells = s->gridW * s->gridH;
    if (numCells > s->cellCapacity) {
        free_aligned(s->cellStart);
        free_aligned(s->cellCursor);
        s->cellStart  = (int *)alloc_aligned(sizeof(int) * (size_t)(numCells + 1));
        s->cellCursor = (int *)alloc_aligned(sizeof(int) * (size_t)numCells);
        s->cellCapacity = (s->cellStart && s->cellCursor) ? numCells : 0;
    }
    return s->cellCapacity >= numCells;
}

int cpu_max_threads(void)
{
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

static inline int cell_coord(float v, float invCellSize, int n)
{
    int c = (int)(v * invCellSize);
    return c < 0 ? 0 : (c >= n ? n - 1 : c);
}

static void build_grid(CpuSolver *s, const float *pos, int count)
{
    const int numCells = s->gridW * s->gridH;
    memset(s->cellStart, 0, sizeof(int) * (size_t)(numCells + 1));

    for (int i = 0; i < count; i++) {
        int cx = cell_coord(pos[2 * i + 0], s->invCellSize, s->gridW);
        int cy = cell_coord(pos[2 * i + 1], s->invCellSize, s->gridH);
        int cell = cy * s->gridW + cx;
        s->cellOf[i] = cell;
        s->cellStart[cell + 1]++;
    }
    for (int c = 0; c < numCells; c++) {
        s->cellStart[c + 1] += s->cellStart[c];
    }
    memcpy(s->cellCursor, s->cellStart, sizeof(int) * (size_t)numCells);
    for (int i = 0; i < count; i++) {
        s->sortedIdx[s->cellCursor[s->cellOf[i]]++] = i;
    }
}

/* One sub-step: integrate, rebuild the grid, then Jacobi iterations. */
static void cpu_substep(CpuSolver *s, CircleArrays *c, int count, const SimParams *params,
                        float h, int useGrid, int numThreads, SimStats *stats)
{
    (void)numThreads;  /* unused without OpenMP */
    const float dampingStep = powf(params->damping, h * 60.0f);
    const float invH = 1.0f / h;
    const float width = params->width;
    const float height = params->height;
    const float elasticity = params->elasticity;
    const int iterations = params->iterations > 0 ? params->iterations : 1;

    /* 1. Integrate */
    {
        float *pos = c->pos, *vel = c->vel;
        const float *radius = c->radius;
        #pragma omp parallel for schedule(static) num_threads(numThreads)
        for (int i = 0; i < count; i++) {
            integrate_body(&pos[2 * i], &pos[2 * i + 1], &vel[2 * i], &vel[2 * i + 1], radius[i],
                           params->gravity, dampingStep, h, width, height, elasticity);
        }
    }

    /* 2. Broad phase */
    if (useGrid) build_grid(s, c->pos, count);

    /* 3. Jacobi iterations: read c->pos/vel, write s->pos/vel, then swap. */
    for (int iter = 0; iter < iterations; iter++) {
        const int last = (iter == iterations - 1) && stats;
        const float *inPos = c->pos, *inVel = c->vel;
        const float *radius = c->radius, *invMass = c->invMass;
        float *outPos = s->pos, *outVel = s->vel;

        /* Statistics go through OpenMP reductions (private per-thread copies),
           not a shared per-thread array that would false-share a cache line. */
        float maxPen = 0.0f;
        double kinetic = 0.0;
        long contacts = 0;

        #pragma omp parallel for schedule(static) num_threads(numThreads) \
            reduction(max:maxPen) reduction(+:kinetic) reduction(+:contacts)
        for (int i = 0; i < count; i++) {
            float x = inPos[2 * i], y = inPos[2 * i + 1];
            float vx = inVel[2 * i], vy = inVel[2 * i + 1];
            const float r = radius[i], w = invMass[i];

            ContactAccum acc;
            contact_accum_reset(&acc);

            if (useGrid) {
                int cx = cell_coord(x, s->invCellSize, s->gridW);
                int cy = cell_coord(y, s->invCellSize, s->gridH);
                for (int ny = cy - 1; ny <= cy + 1; ny++) {
                    if (ny < 0 || ny >= s->gridH) continue;
                    for (int nx = cx - 1; nx <= cx + 1; nx++) {
                        if (nx < 0 || nx >= s->gridW) continue;
                        int cell = ny * s->gridW + nx;
                        for (int k = s->cellStart[cell]; k < s->cellStart[cell + 1]; k++) {
                            int j = s->sortedIdx[k];
                            if (j == i) continue;
                            contact_accumulate(&acc, i, x, y, vx, vy, r, w,
                                               j, inPos[2 * j], inPos[2 * j + 1],
                                               inVel[2 * j], inVel[2 * j + 1],
                                               radius[j], invMass[j], elasticity, invH);
                        }
                    }
                }
            } else {
                for (int j = 0; j < count; j++) {
                    if (j == i) continue;
                    contact_accumulate(&acc, i, x, y, vx, vy, r, w,
                                       j, inPos[2 * j], inPos[2 * j + 1],
                                       inVel[2 * j], inVel[2 * j + 1],
                                       radius[j], invMass[j], elasticity, invH);
                }
            }

            contact_apply(&acc, &x, &y, &vx, &vy);
            apply_walls(&x, &y, &vx, &vy, r, width, height, elasticity);

            outPos[2 * i] = x;
            outPos[2 * i + 1] = y;
            outVel[2 * i] = vx;
            outVel[2 * i + 1] = vy;

            if (last) {
                maxPen = fmaxf(maxPen, acc.maxPenetration);
                contacts += acc.contacts;
                if (w > 0.0f) kinetic += 0.5 * (double)(vx * vx + vy * vy) / (double)w;
            }
        }

        float *tmp = c->pos; c->pos = s->pos; s->pos = tmp;
        tmp = c->vel; c->vel = s->vel; s->vel = tmp;

        if (last) {
            stats->maxPenetration = maxPen;
            stats->kineticEnergy = (float)kinetic;
            stats->contacts = (uint32_t)(contacts / 2);
        }
    }

    /* 4. Sleep near-stationary bodies to stop resting piles from jittering */
    {
        float *vel = c->vel;
        #pragma omp parallel for schedule(static) num_threads(numThreads)
        for (int i = 0; i < count; i++) {
            sleep_body(&vel[2 * i], &vel[2 * i + 1]);
        }
    }
}

void cpu_solver_step(CpuSolver *s, CircleArrays *c, int count, const SimParams *params,
                     float dt, int useGrid, int numThreads, SimStats *stats)
{
    if (stats) memset(stats, 0, sizeof(*stats));
    if (count <= 0 || dt <= 0.0f || c->capacity != s->capacity) return;
    if (numThreads < 1) numThreads = 1;

    const int substeps = params->substeps > 0 ? params->substeps : 1;
    const float h = dt / (float)substeps;
    for (int sub = 0; sub < substeps; sub++) {
        /* Statistics describe the final sub-step */
        cpu_substep(s, c, count, params, h, useGrid, numThreads,
                    sub == substeps - 1 ? stats : NULL);
    }
}
