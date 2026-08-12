#include <cstdio>
#include <cstdlib>
#include <mc_runtime.h>
#include <mc_common.h>

#define CHECK(call) do {  mcError_t err = call;  if (err != mcSuccess) {  std::fprintf(stderr, "%s failed at %s:%d: %s\n", #call, __FILE__, __LINE__, mcGetErrorString(err));  std::exit(1);  }  } while (0)

__global__ void add1_kernel(int *x) { x[0] += 1; }

int main() {
  int h = 41;
  int *d = nullptr;
  CHECK(mcMalloc((void**)&d, sizeof(int)));
  CHECK(mcMemcpy(d, &h, sizeof(int), mcMemcpyHostToDevice));
  add1_kernel<<<1, 1, 0, 0>>>(d);
  CHECK(mcGetLastError());
  CHECK(mcDeviceSynchronize());
  CHECK(mcMemcpy(&h, d, sizeof(int), mcMemcpyDeviceToHost));
  CHECK(mcFree(d));
  std::printf("add1_result=%d\n", h);
  return h == 42 ? 0 : 2;
}
