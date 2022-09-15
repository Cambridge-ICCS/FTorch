import torch
import torch.nn.functional as F
import torchvision

def main():

    # Load a pre-trained PyTorch model
    print("Loading pre-trained ResNet-18 model...")
    model = torchvision.models.resnet18(pretrained=True)
    # Switch-off some specific layers/parts of the model that behave
    # differently during training and inference
    model.eval()
    dummy_input = torch.ones(1, 3, 224, 224)
    output = model(dummy_input)
    top5 = F.softmax(output, dim=1).topk(5).indices
    print('PyTorch model top 5 results:\n  {}'.format(top5))

    print("\nHow would you like to construct the model?")
    print(" 1. CPU scripted")
    print(" 2. CPU traced")
    print(" 3. GPU scripted")
    print(" 4. GPU traced")

    # Get input from the user, sanitise it and focus on the first character
    user_input = input("Enter 1, 2, 3, or 4: ")
    user_input = user_input.strip()[0]
    if user_input == "1":
        # Generate a TorchScript CPU model via scripting
        model_name = "scripted_cpu.pt"
        print("Generating a TorchScript model on the CPU using scripting...")
        scripted_model_cpu = torch.jit.optimize_for_inference(torch.jit.script(model))
        scripted_model_cpu.save(model_name)
        print("Wrote " + model_name)
        output = scripted_model_cpu(dummy_input)
        top5 = F.softmax(output, dim=1).topk(5).indices
        print('TorchScript model top 5 results:\n  {}'.format(top5))

    elif user_input == "2":
        # Generate a TorchScript CPU model via tracing with dummy input
        model_name = 'traced_cpu.pt'
        print("Generating a TorchScript model on the CPU using tracing...")
        traced_model_cpu = torch.jit.trace(model, dummy_input)
        traced_model_cpu.save(model_name)
        print("Wrote " + model_name)
        output = traced_model_cpu(dummy_input)
        top5 = F.softmax(output, dim=1).topk(5).indices
        print('TorchScript model top 5 results:\n  {}'.format(top5))

    elif user_input == "3":
        device = torch.device('cuda')
        model_gpu = model.to(device)
        model_gpu.eval()
        dummy_input_gpu = dummy_input.to(device)

        # Generate a TorchScript GPU model via scripting
        print("Generating a TorchScript model on the GPU using scripting...")
        model_name = 'scripted_gpu.pt'
        scripted_model_gpu = torch.jit.optimize_for_inference(torch.jit.script(model_gpu))
        scripted_model_gpu.save(model_name)
        print("Wrote " + model_name)
        output = scripted_model_gpu(dummy_input_gpu)
        top5 = F.softmax(output, dim=1).topk(5).indices
        print('TorchScript model top 5 results:\n  {}'.format(top5))

    elif user_input == "4":
        device = torch.device('cuda')
        model_gpu = model.to(device)
        model_gpu.eval()
        dummy_input_gpu = dummy_input.to(device)

        print("Generating a TorchScript model on the GPU using tracing...")
        model_name = 'traced_gpu.pt'
        traced_model_gpu = torch.jit.optimize_for_inference(torch.jit.trace(model_gpu, dummy_input_gpu))
        traced_model_gpu.save(model_name)
        print("Wrote " + model_name)
        output = traced_model_gpu(dummy_input_gpu)
        top5 = F.softmax(output, dim=1).topk(5).indices
        print('TorchScript model top 5 results:\n  {}'.format(top5))

    else:
        print("Invalid input, please type 1, 2, 3 or 4")

if __name__ == '__main__':
    main()
