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

  /* We assume you already know which model your are deploying */
  /* In this case, a CNN model trained on Imagenet, so the input shape
     is (1, 3, 224, 224) and the output shape is (1, 1000) */
  const int input_ndim = 4;
  int64_t* input_shape = (int64_t*)malloc(input_ndim * sizeof(int64_t));
  input_shape[0] = batch_size;
  input_shape[1] = 3;
  input_shape[2] = input_shape[3] = 224;

  const int output_ndim = 2;
  int64_t* output_shape = (int64_t*)malloc(output_ndim * sizeof(int64_t));
  output_shape[0] = batch_size;
  output_shape[1] = 1000;

  torch_jit_script_module_t model = torch_jit_load(argv[1]);
  int64_t input_size
      = input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3];
  float* input_data = (float*)malloc(input_size * sizeof(float));
  for (int i = 0; i < input_size; i++)
    input_data[i] = 1.0;

  int64_t output_size = output_shape[0] * output_shape[1];
  float* output_data = (float*)malloc(output_size * sizeof(float));

  if (model) {
    torch_tensor_t input = torch_from_blob(input_data, input_ndim, input_shape,
                                           torch_kFloat32, torch_kCPU);
    torch_tensor_t output = torch_from_blob(
        output_data, output_ndim, output_shape, torch_kFloat32, torch_kCPU);
    torch_jit_module_forward(model, input, output);
    torch_tensor_print(output);
    torch_jit_module_delete(model);
    torch_tensor_delete(input);
    torch_tensor_delete(output);
  }

  // Cleanup
  free(input_shape);
  free(output_shape);
  free(input_data);
  free(output_data);
}
