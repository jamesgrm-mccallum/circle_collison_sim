/*
 * Headless tests for the CPU solver. The GPU kernels share the same per-body
 * math (src/contact.h), so these also cover the algorithm the GPU runs.
 */
#include "circle.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_failures = 0;

#define CHECK(cond, ...)                                              \
    do {                                                              \
        if (!(cond)) {                                                \
            fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);      \
            fprintf(stderr, __VA_ARGS__);                             \
            fprintf(stderr, "\n");                                    \
            g_failures++;                                             \
        }                                                             \
    } while (0)

static const float kWidth = 1600.0f;
static const float kHeight = 900.0f;

static SimParams default_params(void)
{
    SimParams p;
    p.gravity = 100.0f;
    p.damping = 0.995f;
    p.elasticity = 0.8f;
    p.width = kWidth;
    p.height = kHeight;
    p.substeps = 8;
    p.iterations = 2;
    return p;
}

static void copy_circles(CircleArrays *dst, const CircleArrays *src, int n)
{
    memcpy(dst->pos, src->pos, sizeof(float) * 2 * (size_t)n);
    memcpy(dst->vel, src->vel, sizeof(float) * 2 * (size_t)n);
    memcpy(dst->radius, src->radius, sizeof(float) * (size_t)n);
    memcpy(dst->invMass, src->invMass, sizeof(float) * (size_t)n);
    memcpy(dst->color, src->color, sizeof(uint32_t) * (size_t)n);
}

/* The grid must find exactly the contacts brute force finds. */
static void test_grid_matches_brute_force(void)
{
    const int n = 3000;
    CircleArrays a, b;
    circles_alloc(&a, n);
    circles_alloc(&b, n);
    float maxR = circles_init_random(&a, n, kWidth, kHeight, 1234u);
    copy_circles(&b, &a, n);

    CpuSolver *s = cpu_solver_create(n);
    cpu_solver_configure_grid(s, kWidth, kHeight, maxR);

    SimParams p = default_params();
    p.substeps = 1;
    p.iterations = 1;   /* with one iteration the neighbour sets are identical */
    SimStats sa, sb;
    cpu_solver_step(s, &a, n, &p, 1.0f / 120.0f, 1, 1, &sa);
    cpu_solver_step(s, &b, n, &p, 1.0f / 120.0f, 0, 1, &sb);

    float maxDiff = 0.0f;
    for (int i = 0; i < 2 * n; i++) {
        maxDiff = fmaxf(maxDiff, fabsf(a.pos[i] - b.pos[i]));
        maxDiff = fmaxf(maxDiff, fabsf(a.vel[i] - b.vel[i]));
    }
    CHECK(sa.contacts == sb.contacts, "contacts grid=%u brute=%u", sa.contacts, sb.contacts);
    CHECK(maxDiff < 1e-3f, "grid vs brute force max diff %g", maxDiff);
    CHECK(sa.contacts > 0, "expected some initial overlaps");

    cpu_solver_destroy(s);
    circles_free(&a);
    circles_free(&b);
}

/* The Jacobi formulation has no shared writes, so results must be bitwise
   identical for any thread count. */
static void test_thread_count_determinism(void)
{
    const int n = 4000;
    int threads = cpu_max_threads();
    if (threads < 2) {
        printf("skip determinism test (single thread build)\n");
        return;
    }
    CircleArrays a, b;
    circles_alloc(&a, n);
    circles_alloc(&b, n);
    float maxR = circles_init_random(&a, n, kWidth, kHeight, 99u);
    copy_circles(&b, &a, n);

    CpuSolver *s = cpu_solver_create(n);
    cpu_solver_configure_grid(s, kWidth, kHeight, maxR);
    SimParams p = default_params();
    for (int step = 0; step < 60; step++) {
        cpu_solver_step(s, &a, n, &p, 1.0f / 120.0f, 1, 1, NULL);
        cpu_solver_step(s, &b, n, &p, 1.0f / 120.0f, 1, threads, NULL);
    }
    CHECK(memcmp(a.pos, b.pos, sizeof(float) * 2 * (size_t)n) == 0, "positions differ between 1 and %d threads", threads);
    CHECK(memcmp(a.vel, b.vel, sizeof(float) * 2 * (size_t)n) == 0, "velocities differ between 1 and %d threads", threads);

    cpu_solver_destroy(s);
    circles_free(&a);
    circles_free(&b);
}

/* A dense pile must settle: finite, in bounds, small overlap, low energy. */
static void test_pile_settles(int n, int substeps, int iterations)
{
    CircleArrays c;
    circles_alloc(&c, n);
    float maxR = circles_init_random(&c, n, kWidth, kHeight, 7u);
    CpuSolver *s = cpu_solver_create(n);
    cpu_solver_configure_grid(s, kWidth, kHeight, maxR);

    SimParams p = default_params();
    p.substeps = substeps;
    p.iterations = iterations;
    SimStats st;
    const int steps = 1200;   /* 10 s at 120 Hz */
    float initialEnergy = -1.0f;
    for (int step = 0; step < steps; step++) {
        cpu_solver_step(s, &c, n, &p, 1.0f / 120.0f, 1, cpu_max_threads(), &st);
        if (step == 0) initialEnergy = st.kineticEnergy;
    }

    int finite = 1, inBounds = 1;
    float minR = 1e9f;
    for (int i = 0; i < n; i++) {
        float x = c.pos[2 * i], y = c.pos[2 * i + 1], r = c.radius[i];
        if (!isfinite(x) || !isfinite(y) || !isfinite(c.vel[2 * i]) || !isfinite(c.vel[2 * i + 1])) finite = 0;
        if (x < r - 1e-3f || x > kWidth - r + 1e-3f || y < r - 1e-3f || y > kHeight - r + 1e-3f) inBounds = 0;
        if (r < minR) minR = r;
    }
    printf("pile n=%d %dx%d: maxPen=%.3f px (minR %.2f)  KE %.3g -> %.3g  contacts=%u\n",
           n, substeps, iterations, st.maxPenetration, minR, initialEnergy, st.kineticEnergy, st.contacts);

    CHECK(finite, "non-finite state (n=%d)", n);
    CHECK(inBounds, "body escaped the box (n=%d)", n);
    CHECK(st.maxPenetration < 0.5f * minR, "max penetration %.3f too deep (n=%d)", st.maxPenetration, n);
    CHECK(st.kineticEnergy < 0.05f * initialEnergy, "pile did not settle: KE %.3g of initial %.3g (n=%d)",
          st.kineticEnergy, initialEnergy, n);

    cpu_solver_destroy(s);
    circles_free(&c);
}

/* Two circles at the same centre must be pushed apart, not in the same direction. */
static void test_coincident_circles_separate(void)
{
    CircleArrays c;
    circles_alloc(&c, 2);
    for (int i = 0; i < 2; i++) {
        c.pos[2 * i] = 800.0f;
        c.pos[2 * i + 1] = 450.0f;
        c.vel[2 * i] = c.vel[2 * i + 1] = 0.0f;
        c.radius[i] = 10.0f;
        c.invMass[i] = 1.0f;
        c.color[i] = 0xFFFFFFFFu;
    }
    CpuSolver *s = cpu_solver_create(2);
    cpu_solver_configure_grid(s, kWidth, kHeight, 10.0f);
    SimParams p = default_params();
    p.gravity = 0.0f;
    for (int step = 0; step < 120; step++) {
        cpu_solver_step(s, &c, 2, &p, 1.0f / 120.0f, 1, 1, NULL);
    }
    float dx = c.pos[2] - c.pos[0], dy = c.pos[3] - c.pos[1];
    float dist = sqrtf(dx * dx + dy * dy);
    CHECK(dist > 19.0f, "coincident circles still overlapping: distance %.3f", dist);

    cpu_solver_destroy(s);
    circles_free(&c);
}

int main(void)
{
    test_grid_matches_brute_force();
    test_thread_count_determinism();
    test_coincident_circles_separate();
    test_pile_settles(500, 8, 2);
    test_pile_settles(5000, 8, 2);
    test_pile_settles(20000, 16, 1);

    if (g_failures) {
        fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    printf("all physics tests passed\n");
    return 0;
}
