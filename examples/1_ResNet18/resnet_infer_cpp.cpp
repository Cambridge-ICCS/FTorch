#include <torch/script.h>
#include <torch/torch.h>

#include <chrono>
#include <iostream>
#include <memory>

using namespace std::chrono;

int main(int argc, const char* argv[])
{
  if (argc != 3) {
    std::cerr << "Usage: ts_infer_cpp <path-to-exported-script-module> "
                 "<batch_size>\n";
    return -1;
  }

  int batch_size = atoi(argv[2]);

  torch::jit::script::Module module;
  try {
    // Deserialize the Module from a file using torch::jit::load()
    module = torch::jit::load(argv[1]);
  } catch (const c10::Error& e) {
    std::cerr << "Error loading the module\n";
    return -1;
  }

  // Create a vector of input tensors
  std::vector<torch::jit::IValue> inputs;
  // at::Tensor input_tensor = at::randn({batch_size, 3, 224, 224});
  at::Tensor input_tensor = at::ones({batch_size, 3, 224, 224});

  // Execute the model on CPU and turn its output into a tensor
  // IValue forward(std::vector<IValue> inputs, const Kwargs &kwargs = Kwargs())
  std::cout << "Deploying on CPU" << std::endl;
  inputs.push_back(input_tensor);

  auto start = high_resolution_clock::now();
  at::Tensor output = module.forward(inputs).toTensor();
  auto stop = high_resolution_clock::now();
  auto cpu_time = duration_cast<milliseconds>(stop - start);
  std::cout << "CPU time: " << cpu_time.count() << " ms" << std::endl;

  if (torch::cuda::is_available()) {
    std::cout << "Deploying on GPU" << std::endl;
    torch::DeviceType device_type = torch::kCUDA;
    torch::Device device(device_type);
    // Move your model to the GPU
    // void to(at::Device device, bool non_blocking = false)
    // Recursively moves all parameters to the given device.
    // If non_blocking is true and the source is in pinned memory and
    // destination is on the GPU or vice versa, the copy is performed
    // asynchronously with respect to the host. Otherwise, the argument
    // has no effect.
    module.to(at::kCUDA);
    inputs.clear();
    inputs.push_back(input_tensor.to(at::kCUDA));
    auto start = high_resolution_clock::now();
    at::Tensor output = module.forward(inputs).toTensor();
    torch::cuda::synchronize();
    auto stop = high_resolution_clock::now();
    auto gpu_time = duration_cast<milliseconds>(stop - start);
    std::cout << "GPU time: " << gpu_time.count() << " ms" << std::endl;
  }
}
