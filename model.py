from torch import nn, optim
import torch
from parameters import *


def KL4Norm(mu1, log_var1, mu2, log_var2):
    kl_divergence = -0.5 * torch.sum(
        1 + log_var1 - log_var2 - log_var1.exp() / log_var2.exp() - (mu1 - mu2).pow(2) / log_var2.exp())
    return kl_divergence
# 上面KL4Norm函数是用来计算两个高斯分布之间的KL散度的，在本项目中，用来计算两个隐变量之间的KL散度

class MLP(nn.ModuleList):
    def __init__(self, channels, skips=None, use_bn=True, act=nn.LeakyReLU, dropout=0.):
        super().__init__()
        self.num_layers = len(channels) - 1
        if skips is None:
            skips = {}
        self.skips = skips
        self.channels = channels
        for i in range(1, self.num_layers + 1):
            in_channels = channels[i - 1] + (channels[skips[i]] if i in skips else 0)
            layers = [nn.Linear(in_channels, channels[i])]
            if i < self.num_layers:
                if use_bn:
                    layers.append(nn.BatchNorm1d(channels[i]))
                layers.append(act())
            if i + 1 == self.num_layers and dropout > 0:
                layers.append(nn.Dropout(dropout, inplace=True))
            self.append(nn.Sequential(*layers))  # 输入的list必须添加*进行转化

    def forward(self, x):
        xs = [x]  # xs创建为一个list，用于保存每层的输出，方便于进行Resnet
        for i in range(self.num_layers):
            if i + 1 in self.skips:
                x = torch.cat([xs[self.skips[i + 1]], x], dim=-1)
            x = self[i](x)
            xs.append(x)
        return x


def vae_loss(x, x_reconstructed, mu, log_var, beta):
    # reconstruction_loss = nn.functional.binary_cross_entropy(x_reconstructed, x, reduction='sum')
    reconstruction_loss = nn.functional.mse_loss(x_reconstructed, x, reduction='sum')
    kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    return (reconstruction_loss + beta * kl_divergence), reconstruction_loss, (beta * kl_divergence)


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()

        # # Encoder
        # self.encoder = nn.Sequential(
        #     nn.Linear(input_dim, hidden_dim[0]),
        #     nn.LeakyReLU(),
        #     # nn.ReLU(),
        #     nn.Linear(hidden_dim[0], hidden_dim[1]),
        #     nn.LeakyReLU(),
        #     # nn.ReLU(),
        #     nn.Linear(hidden_dim[1], latent_dim * 2)
        # )
        #
        # # Decoder
        # self.decoder = nn.Sequential(
        #     nn.Linear(latent_dim, hidden_dim[1]),
        #     nn.LeakyReLU(),
        #     # nn.ReLU(),
        #     nn.Linear(hidden_dim[1], hidden_dim[0]),
        #     nn.LeakyReLU(),
        #     # nn.ReLU(),
        #     nn.Linear(hidden_dim[0], input_dim),
        #     nn.Tanh()
        # )

        # Encoder
        self.encoder = MLP(
            # [input_dim, 32, 32, 64, 64, 128, 128, 256, 256, 512, 512, 2048, 2048, out_dim],
            # [input_dim, 128, 256, 256, 256, 128, 128, 128, 128, 128, 128, latent_dim * 2],    # 加skips的结构
            [input_dim, 128, 256, 256, 256, 128, latent_dim * 2],
            # [input_dim, 32, 32, 64, 64, 128, 128, 256, 256, 512, 512, out_dim],
            # [input_dim, 4, 8, 16, 32, 64, 128, 512, 1024, 1024, 2048, 2048, out_dim],
            # [input_dim, 4, 8, 16, 32, 64, 128, 512, 1024, 2048, 4096, out_dim],
            # skips={4: 2, 7: 5, 10: 8},  # 一个类似于树的用法，查表
            # skips={3: 1, 5: 3, 7: 5, 9: 7},  # 一个类似于树的用法，查表
            act=nn.LeakyReLU,
            # dropout=0.1
        )

        # Decoder
        self.decoder = MLP(
            # [input_dim, 32, 32, 64, 64, 128, 128, 256, 256, 512, 512, 2048, 2048, out_dim],
            # [latent_dim, 64, 64, 64, 64, 128, 128, 128, 256, 256, 256, input_dim],            # 加skips的结构
            [latent_dim, 64, 128, 256, 256, 256, input_dim],
            # [input_dim, 32, 32, 64, 64, 128, 128, 256, 256, 512, 512, out_dim],
            # [input_dim, 4, 8, 16, 32, 64, 128, 512, 1024, 1024, 2048, 2048, out_dim],
            # [input_dim, 4, 8, 16, 32, 64, 128, 512, 1024, 2048, 4096, out_dim],
            # skips={4: 2, 7: 5, 10: 8},  # 一个类似于树的用法，查表
            # skips={3: 1, 5: 3, 7: 5, 9: 7},  # 一个类似于树的用法，查表
            act=nn.LeakyReLU,
            # dropout=0.1
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encode the input
        h = self.encoder(x.view(-1, input_dim_channel))
        mu, log_var = h.chunk(2, dim=-1)    # h.chunk

        # Reparameterize
        z = self.reparameterize(mu, log_var)

        # Decode the latent representation
        x_reconstructed = self.decoder(z)

        return x_reconstructed, mu, log_var

    def inference(self, mu, log_var, n_samples):
        results = []
        for _ in range(n_samples):
            z = self.reparameterize(mu, log_var)
            x_generated = self.decoder(z).view(-1, input_dim_channel)
            results.append(x_generated)
        results = torch.cat(results, dim=0)
        return results


class VAE_precoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE_precoder, self).__init__()

        # # Encoder
        # self.encoder = nn.Sequential(
        #     nn.Linear(input_dim, hidden_dim[0]),
        #     nn.LeakyReLU(),
        #     # nn.ReLU(),
        #     nn.Linear(hidden_dim[0], hidden_dim[1]),
        #     nn.LeakyReLU(),
        #     # nn.ReLU(),
        #     nn.Linear(hidden_dim[1], latent_dim * 2)
        # )
        #
        # # Decoder
        # self.decoder = nn.Sequential(
        #     nn.Linear(latent_dim, hidden_dim[1]),
        #     nn.LeakyReLU(),
        #     # nn.ReLU(),
        #     nn.Linear(hidden_dim[1], hidden_dim[0]),
        #     nn.LeakyReLU(),
        #     # nn.ReLU(),
        #     nn.Linear(hidden_dim[0], input_dim),
        #     nn.Tanh()
        # )

        self.encoder = MLP(
            # [input_dim, 32, 32, 64, 64, 128, 128, 256, 256, 512, 512, 2048, 2048, out_dim],
            # [input_dim, 128, 256, 256, 256, 128, 128, 128, 128, 128, 128, latent_dim * 2],    # 加skips的结构
            [input_dim, 32, 64, 128, 128, 64, latent_dim * 2],
            # [input_dim, 32, 32, 64, 64, 128, 128, 256, 256, 512, 512, out_dim],
            # [input_dim, 4, 8, 16, 32, 64, 128, 512, 1024, 1024, 2048, 2048, out_dim],
            # [input_dim, 4, 8, 16, 32, 64, 128, 512, 1024, 2048, 4096, out_dim],
            # skips={4: 2, 7: 5, 10: 8},  # 一个类似于树的用法，查表
            # skips={3: 1, 5: 3, 7: 5, 9: 7},  # 一个类似于树的用法，查表
            act=nn.LeakyReLU,
            # dropout=0.1
        )

        # Decoder
        self.decoder = MLP(
            # [input_dim, 32, 32, 64, 64, 128, 128, 256, 256, 512, 512, 2048, 2048, out_dim],
            # [latent_dim, 64, 64, 64, 64, 128, 128, 128, 256, 256, 256, input_dim],            # 加skips的结构
            [latent_dim, 32, 64, 128, 128, 64, input_dim],
            # [input_dim, 32, 32, 64, 64, 128, 128, 256, 256, 512, 512, out_dim],
            # [input_dim, 4, 8, 16, 32, 64, 128, 512, 1024, 1024, 2048, 2048, out_dim],
            # [input_dim, 4, 8, 16, 32, 64, 128, 512, 1024, 2048, 4096, out_dim],
            # skips={4: 2, 7: 5, 10: 8},  # 一个类似于树的用法，查表
            # skips={3: 1, 5: 3, 7: 5, 9: 7},  # 一个类似于树的用法，查表
            act=nn.LeakyReLU,
            # dropout=0.1
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encode the input
        h = self.encoder(x.view(-1, input_dim_precoder))
        mu, log_var = h.chunk(2, dim=-1)

        # Reparameterize
        z = self.reparameterize(mu, log_var)

        # Decode the latent representation
        x_reconstructed = self.decoder(z)

        return x_reconstructed, mu, log_var

    def inference(self, mu, log_var, n_samples):
        results = []
        for _ in range(n_samples):
            z = self.reparameterize(mu, log_var)
            x_generated = self.decoder(z).view(-1, input_dim_precoder)
            results.append(x_generated)
        results = torch.cat(results, dim=0)
        return results
