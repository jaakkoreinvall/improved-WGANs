import argparse
import torch
from torch.distributions import normal

import batch_generator
import models

parser = argparse.ArgumentParser(description='Wasserstein GAN with Gradient Penalty - MNIST')
parser.add_argument('--batch_size', type=int, default=50, help='batch size (default: 50)')
parser.add_argument('--datapath', required=True, help='path to dataset')
parser.add_argument('--latent_dim', type=int, default=128, help='dimensionality of the latent variable, z (default: 128)')
parser.add_argument('--model_dim', type=int, default=64, help='dimensionality of the model (default: 64)')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
args = parser.parse_args()

batch_size = args.batch_size
datapath = args.datapath
latent_dim = args.latent_dim
model_dim = args.model_dim
seed = args.seed

output_dim = 784 # 784 = 28 * 28, number of pixels in an MNIST image

torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

generator = models.Generator(output_dim, latent_dim, model_dim)
#generator.load_state_dict(torch.load('generator.pth.tar'))
generator.to(device)
critic = models.Critic(model_dim)
#critic.load_state_dict(torch.load('critic.pth.tar'))
critic.to(device)

# test models
# 1. latent variable to generator
normal_dist = normal.Normal(0.0, 1.0)
with torch.no_grad():
    latent_variable_z = normal_dist.sample((batch_size, latent_dim))
latent_variable_z = latent_variable_z.to(device)
generator_output = generator(latent_variable_z)
print("generator's output size: ", generator_output.size())

# 2. generator output to critic
critic_output = critic(generator_output)
print("critic's output size: ", critic_output.size())

# 3. real data to critic
for phase in ['train', 'test']:
    data_generator = batch_generator.batch_generator(batch_size, phase, datapath)
    i, sampled_batch = next(data_generator)
    real_data_x = sampled_batch[0]
    real_data_x = real_data_x.to(device)
    print("real_data's size: ", real_data_x.size())
    critic_output = critic(real_data_x)
    print("critic's output size: ", critic_output.size())