#ifndef CIRCLE_H
#define CIRCLE_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float x, y;           // Position
    float vx, vy;         // Velocity
    float radius;         // Radius
    float mass;           // Mass
    float r, g, b;        // Color
} Circle;

typedef struct {
    float gravity;
    float damping;
    float elasticity;
    float width;
    float height;
} SimParams;

// CPU Physics functions
void physics_cpu_init(Circle *circles, int count, float width, float height);
void physics_cpu_update(Circle *circles, int count, const SimParams *params, float dt);

// GPU Physics functions
#ifdef __cplusplus
}

// CUDA functions (C++ only)
void physics_gpu_init(Circle **d_circles, int count);
void physics_gpu_cleanup(Circle *d_circles);
void physics_gpu_update(Circle *h_circles, Circle *d_circles, int count, 
                        const SimParams *params, float dt);

// CUDA-OpenGL Interop
void physics_gpu_update_vbo(unsigned int vbo, int count, 
                            const SimParams *params, float dt);
void physics_gpu_register_vbo(unsigned int vbo, int count);
void physics_gpu_sync_persistent(const Circle *h_circles, int count);
void physics_gpu_unregister_vbo();
bool physics_gpu_interop_available();  

extern "C" {
#endif

#ifdef __cplusplus
}
#endif

#endif // CIRCLE_H