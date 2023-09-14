import torch
import torch.nn as nn
import torch.nn.functional as F
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import matplotlib.pyplot as plt
from torchvision import transforms as T, utils

results_path = 'checkpoints/noise_predictor'
# Load your pretrained U-Net model
data = torch.load('checkpoints/diffusion20230913_191837/model-12.pt')

model = Unet(
    dim=64,
    dim_mults=(1, 2, 4),
    channels=1,
    full_attn = (False, False, True), # use full attention on the final U-Net layer.
    flash_attn=True,  # use attention on all U-Net levels.
)

# from torchinfo import summary
# batch_size = 16
# summary(unet, input_size=[(batch_size, 1, 28, 28), (batch_size, )])

diffusion = GaussianDiffusion(
    model,
    image_size=28,
    objective="pred_v",
    timesteps=1000,  # number of steps
    sampling_timesteps=999,
    # sampling_timesteps = 250,    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    # objective="pred_noise",

)
diffusion.load_state_dict(data['model'])

# Freeze the parameters of the U-Net model
for param in model.parameters():
    param.requires_grad = False

# Define a dictionary to store the intermediate layer outputs and their shapes
intermediate_outputs = {}


# Define the hook function to store the output and shape of the desired layer
def hook_fn(module, input, output, layer_name):
    intermediate_outputs[layer_name] = {
        'output': output,
        # 'shape': output.shape
    }


# Define the list of layer names for which you want to capture the outputs
layer_names = ['conv1', 'conv2', 'conv3'] 

# Register the hook on the desired layers
for idx, down_block in enumerate(model._modules["downs"]):
    # desired_layer = unet._modules.get(layer_name)
    desired_layer = down_block[2]
    desired_layer.register_forward_hook(
        lambda module, input, output, layer_name="Residual" + str(idx): hook_fn(module, input, output, layer_name))


# Define the regression network
class NoiseLevelPredictor(nn.Module):
    def __init__(self):
        super(NoiseLevelPredictor, self).__init__()

        # Linear layers for lowest features
        self.res0_linear = nn.Linear(64 * 28 * 28, 56)
        # Linear layers for intermediate features
        self.res1_linear = nn.Linear(64 * 14 * 14, 28)
        # Layers for high level features
        self.res2_linear = nn.Linear(128 * 7 * 7, 14)

        # Global average pooling
        self.pooling = nn.AdaptiveAvgPool1d(1)

        # Final linear layer
        self.final_linear = nn.Linear(3, 1)

    def forward(self, res0, res1, res2):
        # Process conv1 features
        res0 = self.res0_linear(res0.view(res0.size(0), -1))
        res0 = self.pooling(F.relu(res0))

        # Process conv2 features
        res1 = self.res1_linear(res1.view(res1.size(0), -1))
        res1 = self.pooling(F.relu(res1))

        # Process conv3 features
        res2 = self.res2_linear(res2.view(res2.size(0), -1))
        res2 = self.pooling(F.relu(res2))

        # Concatenate processed features
        features = torch.cat((res0, res1, res2), dim=1)

        # Final linear layer
        output = self.final_linear(features)

        return output


# Create an instance of the noise level predictor model
noise_level_predictor = NoiseLevelPredictor()

# Define your loss function (e.g., Mean Squared Error)
criterion = nn.MSELoss()

# Define your optimizer
optimizer = torch.optim.Adam(noise_level_predictor.parameters(), lr=0.001)

# Assuming you have a dataset with input images and corresponding noise levels
# Define your data loading and preprocessing steps

# Training loop
step = 0
train_num_steps = 1000

while step < train_num_steps:
    # img, _ = next(self.dl)
    # data = img.to(device)
    optimizer.zero_grad()
    # Write function sample & predict
    # Sample stuff
    img = diffusion.ddim_learn_noise([16, 1, 28, 28], noise_model=noise_level_predictor, loss_fn=criterion,
                                   optimizer=optimizer,
                                   intermediate_outputs=intermediate_outputs)
    
    utils.save_image(img, str("sampled_ddim_" + str(step) + ".png"))
    # Compute the loss
    # loss = criterion(predicted_noise_levels, noise_levels)

    # Backward pass and optimization
    # loss.backward()
    # optimizer.step()

    # Print the loss for monitoring
    # print('Step [{}/{}], Loss: {:.4f}'.format(step + 1, train_num_steps, loss.item()))

    step += 1
