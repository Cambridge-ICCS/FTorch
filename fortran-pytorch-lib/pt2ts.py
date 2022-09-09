import torch
import torchvision

def main():

    # Load a pre-trained PyTorch model
    print("Loading pre-trained restnet101 model...")
    cpu_model = torchvision.models.resnet101(pretrained=True)

    print("\nHow would you like to construct the model?")
    print(" 1. CPU")
    print(" 2. CPU optimised")
    print(" 3. GPU")

    # Get input from the user, sanitise it and focus on the first character
    user_input = input("Enter 1, 2, or 3: ")
    user_input = user_input.strip()[0]
    if user_input == "1":
        # Option 1
        # Generate a torch.jit.ScriptModule via scripting
        model_name = "annotated_cpu.pt"
        print("Generating saved model on the CPU...")
        annotated_script_model_cpu = torch.jit.script(cpu_model)
        annotated_script_model_cpu.save(model_name)
        print("Wrote " + model_name)

    elif user_input == "2":
        # Option 2
        # Generate a torch.jit.ScriptModule via tracing (i.e. one that is optimised)
        # with representative input
        model_name = 'traced_cpu.pt'
        repr_input = torch.rand(1, 3, 224, 224)
        print("Generating an optimised saved model on the CPU (using tracing)...")
        traced_script_model_cpu = torch.jit.trace(cpu_model, repr_input)
        traced_script_model_cpu.save(model_name)
        print("Wrote " + model_name)

    elif user_input == "3":
        # GPU versions
        print("Generating saved model on the GPU...")
        device = torch.device('cuda')
        gpu_model = cpu_model.to(device)
        gpu_input = input.to(device)

        model_name = 'traced_gpu.pt'
        traced_script_model_gpu = torch.jit.trace(gpu_model, gpu_input)
        traced_script_model_gpu.save(model_name)
        print("Wrote " + model_name)

        model_name = 'annotated_gpu.pt'
        annotated_script_model_gpu = torch.jit.script(gpu_model)
        annotated_script_model_gpu.save('annotated_gpu.pt')
        print("Wrote " + model_name)

    else:
        print("Invalid input, please type 1, 2 or 3")

if __name__ == '__main__':
    main()
