import torch
from torch.autograd import Variable

from models import create_model
from options.test_options import TestOptions
from seg_utils.utils import dict_to_opts


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def time_inference(model, input_shape=(1, 256, 256)):
    dummy_input = Variable(torch.randn(1, *input_shape)).cuda()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    model.eval()
    start.record()
    _ = model(dummy_input)
    end.record()

    # Waits for everything to finish running
    torch.cuda.synchronize()
    return start.elapsed_time(end)


base_options = dict(
    dataroot="~/",
    experiment="network_testing",
    name="network_testing",
    output_nc=2,
    model="unet_256",
    dataset_mode="segmentation",
    batch_size=16,
    num_threads=0,
    restore_file="network_testing",
)
opt_str = dict_to_opts(base_options)

# Load the parameters from json file
opt = TestOptions().parse(opt_str)  # get training options
# Define the models and optimizer
model = create_model(opt)

print(model)
print('params = {}'.format(count_parameters(model)))
print('inference time = {}'.format(time_inference(model, (1, 256, 256))))
