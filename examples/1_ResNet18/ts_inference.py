import torch
import argparse

def deploy(device, batch_size):

    input = torch.ones(batch_size, 3, 224, 224)
    #input = torch.rand(batch_size, 3, 224, 224)

    if device == 'cpu':
        # Load model as a ScriptModule
        model = torch.jit.load('./annotated_cpu.pt')
        # Inference
        output = model.forward(input)
    elif device == 'cuda':
        # All previously saved modules, no matter their device, are first
        # loaded onto CPU, and then are moved to the devices they were saved
        # from, so we don't need to manually transfer the model to the GPU
        model = torch.jit.load('./annotated_gpu.pt') 
        input_gpu = input.to(torch.device('cuda'))
        output_gpu = model.forward(input_gpu)
        output = output_gpu.to(torch.device('cpu'))

    print(output[:,0:5])

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", help="Device to run on (cpu, cuda)", required=True)
    parser.add_argument("-b", "--batch-size", type=int, default=1, help="Batch size for inference")

    args = parser.parse_args()
    deploy(args.device, args.batch_size)
