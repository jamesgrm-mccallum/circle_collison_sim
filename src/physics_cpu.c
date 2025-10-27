#include "circle.h"
#include <stdlib.h>
#include <math.h>
#include <time.h>

void physics_cpu_init(Circle *circles, int count, float width, float height) {
    srand((unsigned int)time(NULL));
    
    for (int i = 0; i < count; i++) {
        circles[i].radius = 5.0f + (float)(rand() % 15);
        circles[i].x = circles[i].radius + (float)(rand() % (int)(width - 2 * circles[i].radius));
        circles[i].y = circles[i].radius + (float)(rand() % (int)(height - 2 * circles[i].radius));
        circles[i].vx = -100.0f + (float)(rand() % 200);
        circles[i].vy = -100.0f + (float)(rand() % 200);
        circles[i].mass = circles[i].radius * circles[i].radius * 3.14159f;
        
        // Random color
        circles[i].r = (float)rand() / RAND_MAX;
        circles[i].g = (float)rand() / RAND_MAX;
        circles[i].b = (float)rand() / RAND_MAX;
    }
}

static void resolve_collision(Circle *a, Circle *b, float elasticity, float dt) {
    // --- Tunable constants ---
    const float k_slop     = 0.003f;   // small tolerance for penetration (meters)
    const float k_percent  = 0.65f;    // percent of overlap corrected per iteration (0.5–0.8 good for piles)
    const float bias_factor = 0.25f;   // Baumgarte bias factor (0.2–0.3 typical)
    const float v_threshold = 0.02f;   // resting velocity threshold

    // --- Compute separation vector ---
    float dx = b->x - a->x;
    float dy = b->y - a->y;
    float dist_sq = dx * dx + dy * dy;
    float min_dist = a->radius + b->radius;

    // skip non-overlapping pairs early
    if (dist_sq >= min_dist * min_dist) return;

    float dist = sqrtf(dist_sq);
    float nx, ny;
    if (dist < 1e-8f) {
        // perfectly overlapping; choose arbitrary direction
        nx = 1.0f;
        ny = 0.0f;
        dist = 0.0f;
    } else {
        nx = dx / dist;
        ny = dy / dist;
    }

    float penetration = min_dist - dist;

    // --- Mass properties ---
    float invMassA = (a->mass > 0.0f) ? 1.0f / a->mass : 0.0f;
    float invMassB = (b->mass > 0.0f) ? 1.0f / b->mass : 0.0f;
    float invMassSum = invMassA + invMassB;
    if (invMassSum == 0.0f) return; // both static

    // --- Position correction (mass-weighted Baumgarte) ---
    float correction = 0.0f;
    if (penetration > k_slop)
        correction = (penetration - k_slop) * k_percent;

    float corrA = correction * (invMassA / invMassSum);
    float corrB = correction * (invMassB / invMassSum);

    a->x -= corrA * nx;
    a->y -= corrA * ny;
    b->x += corrB * nx;
    b->y += corrB * ny;

    // --- Relative velocity along normal ---
    float dvx = b->vx - a->vx;
    float dvy = b->vy - a->vy;
    float dvn = dvx * nx + dvy * ny;

    // --- Only handle approaching objects ---
    if (dvn > -v_threshold) return;

    // --- Baumgarte velocity bias (helps prevent sinking) ---
    float bias = bias_factor * fmaxf(penetration - k_slop, 0.0f) / dt;

    // --- Effective elasticity ---
    float effective_elasticity = elasticity * 0.6f;

    // --- Normal impulse ---
    float impulse = (-(1.0f + effective_elasticity) * dvn + bias) / invMassSum;

    // Apply impulse
    float jx = impulse * nx;
    float jy = impulse * ny;

    if (invMassA > 0.0f) {
        a->vx -= invMassA * jx;
        a->vy -= invMassA * jy;
    }
    if (invMassB > 0.0f) {
        b->vx += invMassB * jx;
        b->vy += invMassB * jy;
    }
}

void physics_cpu_update(Circle *circles, int count, const SimParams *params, float dt) {
    //---------------------------------------------------------
    // 1. Apply forces (gravity) and integrate velocities
    //---------------------------------------------------------
    // Use exponential damping (frame-rate independent)
    // If params->damping ~ 0.98 for 60 FPS, then per-step damping = damping^(dt * 60)
    float base_damping = powf(params->damping, dt * 60.0f);
    float gravity = params->gravity;

    for (int i = 0; i < count; i++) {
        Circle *c = &circles[i];

        // Apply gravity
        c->vy += gravity * dt;

        // Apply damping (smooth exponential)
        c->vx *= base_damping;
        c->vy *= base_damping;

        // Integrate position
        c->x += c->vx * dt;
        c->y += c->vy * dt;

        // Velocity clamping for numerical stability
        const float max_speed = 600.0f;
        float speed_sq = c->vx * c->vx + c->vy * c->vy;
        if (speed_sq > max_speed * max_speed) {
            float scale = max_speed / sqrtf(speed_sq);
            c->vx *= scale;
            c->vy *= scale;
        }
    }

    //---------------------------------------------------------
    // 2. Collision resolution (iterative constraint solver)
    //---------------------------------------------------------
    // Increase iterations for stacking stability
    int iterations = 10;
    if (count > 500) iterations = 15;
    if (count > 2000) iterations = 20;

    const float friction = 0.99f;

    for (int iter = 0; iter < iterations; iter++) {
        //-----------------------
        // Circle–circle collisions
        //-----------------------
        for (int i = 0; i < count; i++) {
            Circle *a = &circles[i];
            for (int j = i + 1; j < count; j++) {
                Circle *b = &circles[j];
                resolve_collision(a, b, params->elasticity, dt);
            }
        }

        //-----------------------
        // Boundary constraints
        //-----------------------
        for (int i = 0; i < count; i++) {
            Circle *c = &circles[i];

            // Left wall
            if (c->x < c->radius) {
                c->x = c->radius;
                if (c->vx < 0) {
                    c->vx = -c->vx * params->elasticity;
                    c->vy *= friction;
                }
            }
            // Right wall
            else if (c->x > params->width - c->radius) {
                c->x = params->width - c->radius;
                if (c->vx > 0) {
                    c->vx = -c->vx * params->elasticity;
                    c->vy *= friction;
                }
            }

            // Bottom wall
            if (c->y < c->radius) {
                c->y = c->radius;
                if (c->vy < 0) {
                    c->vy = -c->vy * params->elasticity;
                    c->vx *= friction;
                }
            }
            // Top wall
            else if (c->y > params->height - c->radius) {
                c->y = params->height - c->radius;
                if (c->vy > 0) {
                    c->vy = -c->vy * params->elasticity;
                    c->vx *= friction;
                }
            }
        }
    }

    //---------------------------------------------------------
    // 3. Optional: Sleep low-motion circles (resting optimization)
    //---------------------------------------------------------
    // Helps reduce jitter for piles by zeroing small velocities
    const float sleep_threshold_sq = 0.05f * 0.05f;
    for (int i = 0; i < count; i++) {
        float speed_sq = circles[i].vx * circles[i].vx + circles[i].vy * circles[i].vy;
        if (speed_sq < sleep_threshold_sq) {
            circles[i].vx = 0.0f;
            circles[i].vy = 0.0f;
        }
    }
}