#ifndef CONTACT_H
#define CONTACT_H

/*
 * Per-body solver math shared by the CPU (physics_cpu.c) and GPU
 * (physics_gpu.cu) paths, so both modes run the exact same algorithm.
 *
 * The solver is Jacobi-style: every body gathers the corrections it receives
 * from all of its contacts and writes only its own state. Nothing writes to a
 * neighbour, so one thread (GPU) or loop iteration (CPU) per body needs no
 * atomics or locks, and the result does not depend on thread count or
 * scheduling order.
 */

#include <math.h>

#if defined(__CUDACC__)
#define CS_FN static __host__ __device__ __forceinline__
#else
#define CS_FN static inline
#endif

#ifndef CS_SLOP
#define CS_SLOP          0.01f  /* penetration (px) left uncorrected to avoid jitter */
#endif
#ifndef CS_POS_PERCENT
#define CS_POS_PERCENT   0.8f   /* fraction of remaining overlap removed per iteration */
#endif
#ifndef CS_BIAS
#define CS_BIAS          0.1f   /* Baumgarte velocity bias factor */
#endif
#ifndef CS_REST_SPEED
#define CS_REST_SPEED    1.0f   /* approach speeds below this get no restitution */
#endif
#ifndef CS_RELAXATION
#define CS_RELAXATION    1.0f   /* Jacobi relaxation (omega) applied to averaged corrections */
#endif
#define CS_MAX_SPEED     600.0f
#define CS_SLEEP_SPEED   0.05f
#define CS_WALL_FRICTION 0.99f

typedef struct {
    float dpx, dpy;    /* summed position correction */
    float dvx, dvy;    /* summed velocity impulse (already divided by mass) */
    float maxPenetration;
    int   contacts;
} ContactAccum;

CS_FN void contact_accum_reset(ContactAccum *acc)
{
    acc->dpx = acc->dpy = acc->dvx = acc->dvy = 0.0f;
    acc->maxPenetration = 0.0f;
    acc->contacts = 0;
}

/* Clamp a body to the walls, reflecting the normal velocity component. */
CS_FN void apply_walls(float *x, float *y, float *vx, float *vy, float r,
                       float width, float height, float elasticity)
{
    if (*x < r) {
        *x = r;
        if (*vx < 0.0f) { *vx = -*vx * elasticity; *vy *= CS_WALL_FRICTION; }
    } else if (*x > width - r) {
        *x = width - r;
        if (*vx > 0.0f) { *vx = -*vx * elasticity; *vy *= CS_WALL_FRICTION; }
    }
    if (*y < r) {
        *y = r;
        if (*vy < 0.0f) { *vy = -*vy * elasticity; *vx *= CS_WALL_FRICTION; }
    } else if (*y > height - r) {
        *y = height - r;
        if (*vy > 0.0f) { *vy = -*vy * elasticity; *vx *= CS_WALL_FRICTION; }
    }
}

/* Gravity, damping, speed clamp, explicit integration, then walls. */
CS_FN void integrate_body(float *x, float *y, float *vx, float *vy, float r,
                          float gravity, float dampingStep, float dt,
                          float width, float height, float elasticity)
{
    *vy += gravity * dt;
    *vx *= dampingStep;
    *vy *= dampingStep;

    float speedSq = *vx * *vx + *vy * *vy;
    if (speedSq > CS_MAX_SPEED * CS_MAX_SPEED) {
        float scale = CS_MAX_SPEED / sqrtf(speedSq);
        *vx *= scale;
        *vy *= scale;
    }

    *x += *vx * dt;
    *y += *vy * dt;
    apply_walls(x, y, vx, vy, r, width, height, elasticity);
}

/*
 * Accumulates the correction body `a` receives from its contact with `b`.
 * `ia`/`ib` are the bodies' indices; they only break the tie when the two
 * centres coincide, so the two bodies push in opposite directions.
 */
CS_FN void contact_accumulate(ContactAccum *acc,
                              int ia, float ax, float ay, float avx, float avy, float ar, float aw,
                              int ib, float bx, float by, float bvx, float bvy, float br, float bw,
                              float elasticity, float invDt)
{
    float dx = bx - ax;
    float dy = by - ay;
    float minDist = ar + br;
    float distSq = dx * dx + dy * dy;
    if (distSq >= minDist * minDist) return;

    float wSum = aw + bw;
    if (wSum <= 0.0f) return;

    float dist = sqrtf(distSq);
    float nx, ny;
    if (dist > 1e-6f) {
        float invDist = 1.0f / dist;
        nx = dx * invDist;
        ny = dy * invDist;
    } else {
        nx = (ia < ib) ? 1.0f : -1.0f;
        ny = 0.0f;
        dist = 0.0f;
    }

    float penetration = minDist - dist;
    float share = aw / wSum;   /* lighter bodies take more of the correction */

    acc->contacts += 1;
    acc->maxPenetration = fmaxf(acc->maxPenetration, penetration);

    if (penetration > CS_SLOP) {
        float correction = (penetration - CS_SLOP) * CS_POS_PERCENT * share;
        acc->dpx -= correction * nx;
        acc->dpy -= correction * ny;
    }

    /* Relative velocity along the normal; negative means approaching. */
    float vn = (bvx - avx) * nx + (bvy - avy) * ny;
    if (vn >= 0.0f) return;

    float restitution = (vn < -CS_REST_SPEED) ? elasticity : 0.0f;
    float bias = CS_BIAS * fmaxf(penetration - CS_SLOP, 0.0f) * invDt;
    float impulse = (-(1.0f + restitution) * vn + bias) * share;
    acc->dvx -= impulse * nx;
    acc->dvy -= impulse * ny;
}

/* Averages the accumulated corrections over the contact count (Jacobi). */
CS_FN void contact_apply(const ContactAccum *acc, float *x, float *y, float *vx, float *vy)
{
    if (acc->contacts == 0) return;
    float invContacts = 1.0f / (float)acc->contacts;
    float posScale = CS_RELAXATION * invContacts;
    *x  += acc->dpx * posScale;
    *y  += acc->dpy * posScale;
    *vx += acc->dvx * invContacts;
    *vy += acc->dvy * invContacts;
}

CS_FN void sleep_body(float *vx, float *vy)
{
    if (*vx * *vx + *vy * *vy < CS_SLEEP_SPEED * CS_SLEEP_SPEED) {
        *vx = 0.0f;
        *vy = 0.0f;
    }
}

#endif /* CONTACT_H */
