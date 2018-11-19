import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, output_dim, latent_dim, model_dim):
        super(Generator, self).__init__()
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.model_dim = model_dim
        self.linear = nn.Linear(self.latent_dim, 4*4*4*self.model_dim)
        self.convtransposed1 = nn.ConvTranspose2d(4*self.model_dim, 2*self.model_dim, 4, stride=2, padding=1, bias=False)
        self.convtransposed2 = nn.ConvTranspose2d(2*self.model_dim, self.model_dim, 4, stride=2, padding=1, bias=False)
        self.convtransposed3 = nn.ConvTranspose2d(self.model_dim, 1, 4, stride=2, padding=1, bias=False)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, z):
        z = self.relu(self.linear(z))
        z = z.view(-1, 4*self.model_dim, 4, 4)
        z = self.relu(self.convtransposed1(z))
        z = z[:, :, :7, :7]
        z = self.relu(self.convtransposed2(z))
        z = self.sigmoid(self.convtransposed3(z))
        return z.view(-1, self.output_dim)

class Critic(nn.Module):
    def __init__(self, model_dim):
        super(Critic, self).__init__()
        self.model_dim = model_dim
        self.conv1 = nn.Conv2d(1, self.model_dim, 4, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(self.model_dim, 2*self.model_dim, 4, stride=2, padding=1, bias=False)
        self.conv3 = nn.Conv2d(2*self.model_dim, 4*self.model_dim, 4, stride=2, padding=2, bias=False)
        self.linear = nn.Linear(4*4*4*self.model_dim, 1)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.leakyrelu(self.conv1(x))
        x = self.leakyrelu(self.conv2(x))
        x = self.leakyrelu(self.conv3(x))
        x = x.view(-1, 4*4*4*self.model_dim)
        x = self.linear(x)
        return x