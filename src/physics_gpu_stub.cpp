// Used when the project is built without CUDA: every GPU entry point reports
// "unavailable", so the app runs in CPU mode only.
#include "circle.h"

bool physics_gpu_init(int, bool) { return false; }
void physics_gpu_shutdown() {}
const char *physics_gpu_device_name() { return "none (built without CUDA)"; }
bool physics_gpu_register_vbo(unsigned int) { return false; }
void physics_gpu_upload(const CircleArrays *, int, float, float, float) {}
void physics_gpu_download(CircleArrays *, int) {}
void physics_gpu_step(const SimParams *, float, GpuOutput) {}
const CircleInstance *physics_gpu_host_instances() { return nullptr; }
bool physics_gpu_poll_stats(GpuFrameStats *) { return false; }
void physics_gpu_synchronize() {}
