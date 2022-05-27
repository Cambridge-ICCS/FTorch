#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "ctorch.h"

int main(int argc, const char* argv[])
{
  if (argc != 3) {
    printf("Usage: %s <path-to-exported-script-module> <batch_size>\n",
           argv[0]);
    exit(1);
  }

  int batch_size = atoi(argv[2]);
  const int ndim = 4;
  int64_t* shape = (int64_t*)malloc(ndim * sizeof(int64_t));
  shape[0] = batch_size;
  shape[1] = 3;
  shape[2] = shape[3] = 224;
  torch_jit_script_module_t model = torch_jit_load(argv[1]);
  float* data = (float*)malloc(224 * 224 * 3 * sizeof(float));
  for (int i = 0; i < 224 * 224 * 3; i++)
    data[i] = 1.0;

  if (model) {
    torch_tensor_t input
        = torch_from_blob(data, ndim, shape, torch_kFloat32, torch_kCPU);
    // torch_tensor_t input = torch_ones(ndim, shape, torch_kFloat32,
    // torch_kCPU);
    torch_tensor_t output = torch_jit_module_forward(model, input);
    torch_jit_module_delete(model);
    torch_tensor_delete(input);
    torch_tensor_delete(output);
  }

  free(shape);
}
