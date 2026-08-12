#include <cstdio>
#include <cstdlib>
#include <mc_runtime.h>
#include <mc_common.h>

#define CHECK(call) do {  mcError_t err = call;  if (err != mcSuccess) {  std::fprintf(stderr, "%s failed at %s:%d: %s\n", #call, __FILE__, __LINE__, mcGetErrorString(err));  std::exit(1);  }  } while (0)

int main() {
  int count = 0;
  CHECK(mcGetDeviceCount(&count));
  std::printf("mc_device_count=%d\n", count);
  for (int i = 0; i < count; ++i) {
    mcDeviceProp_t p;
    CHECK(mcGetDeviceProperties(&p, i));
    std::printf("device=%d\n", i);
    std::printf("name=%s\n", p.name);
    std::printf("totalGlobalMem=%zu\n", (size_t)p.totalGlobalMem);
    std::printf("sharedMemPerBlock=%zu\n", (size_t)p.sharedMemPerBlock);
    std::printf("regsPerBlock=%d\n", p.regsPerBlock);
    std::printf("warpSize=%d\n", p.warpSize);
    std::printf("memPitch=%zu\n", (size_t)p.memPitch);
    std::printf("maxThreadsPerBlock=%d\n", p.maxThreadsPerBlock);
    std::printf("maxThreadsDim=%d,%d,%d\n", p.maxThreadsDim[0], p.maxThreadsDim[1], p.maxThreadsDim[2]);
    std::printf("maxGridSize=%d,%d,%d\n", p.maxGridSize[0], p.maxGridSize[1], p.maxGridSize[2]);
    std::printf("clockRate=%d\n", p.clockRate);
    std::printf("totalConstMem=%zu\n", (size_t)p.totalConstMem);
    std::printf("major=%d\n", p.major);
    std::printf("minor=%d\n", p.minor);
    std::printf("textureAlignment=%zu\n", (size_t)p.textureAlignment);
    std::printf("deviceOverlap=%d\n", p.deviceOverlap);
    std::printf("multiProcessorCount=%d\n", p.multiProcessorCount);
    std::printf("kernelExecTimeoutEnabled=%d\n", p.kernelExecTimeoutEnabled);
    std::printf("integrated=%d\n", p.integrated);
    std::printf("canMapHostMemory=%d\n", p.canMapHostMemory);
    std::printf("computeMode=%d\n", p.computeMode);
    std::printf("concurrentKernels=%d\n", p.concurrentKernels);
    std::printf("ECCEnabled=%d\n", p.ECCEnabled);
    std::printf("pciBusID=%d\n", p.pciBusID);
    std::printf("pciDeviceID=%d\n", p.pciDeviceID);
    std::printf("pciDomainID=%d\n", p.pciDomainID);
    std::printf("memoryClockRate=%d\n", p.memoryClockRate);
    std::printf("memoryBusWidth=%d\n", p.memoryBusWidth);
    std::printf("l2CacheSize=%d\n", p.l2CacheSize);
    std::printf("maxThreadsPerMultiProcessor=%d\n", p.maxThreadsPerMultiProcessor);
    std::printf("isMultiGpuBoard=%d\n", p.isMultiGpuBoard);
    std::printf("multiGpuBoardGroupID=%d\n", p.multiGpuBoardGroupID);
  }
  return 0;
}
