import torch
import torchvision

def main():
    
    # Load a pre-trained PyTorch model
    cpu_model = torchvision.models.resnet101(pretrained=True)

    # Option 1
    # Generate a torch.jit.ScriptModule via scripting
    annotated_script_model_cpu = torch.jit.script(cpu_model)
    annotated_script_model_cpu.save('annotated_cpu.pt')

    # Option 2
    # Generate a torch.jit.ScriptModule via tracing with representative input
    input = torch.rand(1, 3, 224, 224)
    traced_script_model_cpu = torch.jit.trace(cpu_model, input)
    traced_script_model_cpu.save('traced_cpu.pt')

    # GPU versions
    device = torch.device('cuda')
    gpu_model = cpu_model.to(device)
    gpu_input = input.to(device)
    traced_script_model_gpu = torch.jit.trace(gpu_model, gpu_input)
    traced_script_model_gpu.save('traced_gpu.pt')
    annotated_script_model_gpu = torch.jit.script(gpu_model)
    annotated_script_model_gpu.save('annotated_gpu.pt')
  
if __name__ == '__main__':
    main()
