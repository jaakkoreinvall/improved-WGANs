"""
A PyTorch implementation of Wasserstein GAN with gradient penalty (WGAN-GP) 
based on https://arxiv.org/abs/1704.00028 that is applied to MNIST data 
"""

import argparse
import torch
from torch.distributions import normal, uniform
import torch.optim as optim

import batch_generator
import models

parser = argparse.ArgumentParser(description='Wasserstein GAN with Gradient Penalty - MNIST')
parser.add_argument('--batch_size', type=int, default=50, help='batch size (default: 50)')
parser.add_argument('--datapath', required=True, help='path to dataset')
parser.add_argument('--grad_penalty_coef', type=float, default=10, help='the gradient penalty coefficient')
parser.add_argument('--latent_dim', type=int, default=128, help='dimensionality of the latent variable, z (default: 128)')
parser.add_argument('--load_models', action='store_true', help='load pretrained models (default: False)')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
parser.add_argument('--model_dim', type=int, default=64, help='dimensionality of the model (default: 64)')
parser.add_argument('--n_epochs', type=int, default=25, help='number of epochs (default: 25)')
parser.add_argument('--n_critic', type=int, default=5, help='number of critic iterations per generator iteration (default: 5)')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
args = parser.parse_args()

batch_size = args.batch_size
datapath = args.datapath
grad_penalty_coef = args.grad_penalty_coef
latent_dim = args.latent_dim
load_models = args.load_models
lr = args.lr
model_dim = args.model_dim
n_epochs = args.n_epochs
n_critic = args.n_critic
seed = args.seed

output_dim = 784 # 784 = 28 * 28, number of pixels in an MNIST image

torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

generator = models.Generator(output_dim, latent_dim, model_dim)
critic = models.Critic(model_dim)
if load_models == True:
    generator.load_state_dict(torch.load('generator.pth.tar'))
    critic.load_state_dict(torch.load('critic.pth.tar'))
generator.to(device)
critic.to(device)

critic_optimizer = optim.Adam(critic.parameters(), lr=lr, betas=(0.5, 0.9))
generator_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.9))

# set distributions for later use
normal_dist = normal.Normal(0.0, 1.0)
uniform_dist = uniform.Uniform(0.0, 1.0)

# WGAN with gradient penalty algorithm
generator_iteration = 0
losses = []
for epoch in range(n_epochs):
    print('Epoch {}/{}'.format(epoch, n_epochs - 1))
    print('-' * 10)
    
    for phase in ['train', 'test']:
        data_generator, n_batches = batch_generator.batch_generator(batch_size, phase, datapath)
        print("number of batches: ", n_batches)
        if phase == 'train':
            critic.train()
            generator.train()
        else:
            critic.eval()
            generator.eval()

        batch_count = 0
        while batch_count < n_batches:
            for p in critic.parameters():
                p.requires_grad_(True)
            for p in generator.parameters():
                p.requires_grad_(False)

            critic_iter_count = 0
            while critic_iter_count < n_critic and batch_count < n_batches:

                critic_optimizer.zero_grad()

                _, sampled_batch = next(data_generator)

                # line 4 (1/3): sample real data, x
                real_data_x = sampled_batch[0]

                # line 4 (2/3): sample latent variable, z
                latent_variable_z = normal_dist.sample((batch_size, latent_dim))

                # line 4 (3/3): sample a random number, e
                eps = uniform_dist.sample((batch_size, 1, 1, 1))
                eps = eps.expand_as(real_data_x)

                real_data_x, latent_variable_z, eps = real_data_x.to(device), latent_variable_z.to(device), eps.to(device)

                with torch.set_grad_enabled(phase == 'train'):
                    # line 5: generate fake data
                    fake_data_xtilde = generator(latent_variable_z)
                    fake_data_xtilde = fake_data_xtilde.view(batch_size, 1, 28, 28)
                    
                    # line 7 (1/3): compute the negative critic loss
                    negative_critic_loss = - (critic(fake_data_xtilde) - critic(real_data_x)).mean()
    
                    if phase == 'train':
                        # line 6: define sampling distribution so that it samples uniformly between data distribution and generator distribution
                        interpolated_data_xhat = torch.tensor((eps * real_data_x + (1-eps) * fake_data_xtilde), device=device, requires_grad=True)
    
                        # line 7 (2/3): compute the gradient penalty
                        critic_output = critic(interpolated_data_xhat)
                        weight = torch.ones(critic_output.size()).to(device)
                        gradients = torch.autograd.grad(outputs = critic_output,
                                                        inputs = interpolated_data_xhat,
                                                        grad_outputs = weight,
                                                        create_graph=True)
                        critic_grad_l2_norm = torch.norm(gradients[0].view(batch_size, -1), p=2, dim=1)
                        grad_penalty = (critic_grad_l2_norm - 1) ** 2
    
                        # line 7 (3/3): compute loss for the optimization of the critic
                        loss = - negative_critic_loss + (grad_penalty_coef * grad_penalty).mean()
                        # line 9: compute gradients and update weights
                        loss.backward()
                        critic_optimizer.step()

                critic_iter_count += 1
                batch_count += 1

            print('{} Loss: {:.4f}'.format(batch_count, negative_critic_loss.item()))
            losses.append(negative_critic_loss.item())

            for p in critic.parameters():
                p.requires_grad_(False)
            for p in generator.parameters():
                p.requires_grad_(True)
            # line 11: sample latent variable, z
            latent_variable_z = normal_dist.sample((batch_size, latent_dim)).to(device)
            generator_optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                # line 12 (1/3): generate fake data
                fake_data = generator(latent_variable_z)
                # line 12 (2/3): compute loss for the optimization of the generator
                generator_loss = -critic(fake_data).mean()
                
                if phase == 'train':
                    # line 12 (3/3): compute gradients and update weights
                    generator_loss.backward()
                    generator_optimizer.step()

            if (generator_iteration+1) % 1000 == 0:
                torch.save(critic.state_dict(), 'critic.pth.tar')
                torch.save(generator.state_dict(), 'generator.pth.tar')
                with open("losses.txt", "a") as losses_file:
                    for loss in losses:
                        losses_file.write("%s\n" % loss)
                losses = []

            generator_iteration += 1