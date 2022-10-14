import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

from einops.layers.torch import Rearrange
from jukebox.vqvae.bottleneck import BottleneckBlock

class RCBlock(nn.Module):
    def __init__(self, output_dim, ks, dilation, num_groups):
        super().__init__()
        ksm1 = ks-1
        mfd = output_dim
        di = dilation
        self.num_groups = num_groups

        self.relu = nn.LeakyReLU()
        self.conv = nn.Conv1d(mfd, mfd, ks, 1, ksm1*di//2, dilation=di, groups=num_groups)
        self.gn = nn.GroupNorm(num_groups, mfd)

    def init_hidden(self, batch_size, hidden_size):
        num_layers = 1
        num_directions = 2
        hidden = torch.zeros(num_layers*num_directions, batch_size, hidden_size)
        hidden.normal_(0, 1)
        return hidden

    def forward(self, x):
        bs, mfd, nf = x.size()
        r = x.clone()
        c = self.relu(self.gn(self.conv(r)))
        x = x+r+c

        return x


class BodyGBlock(nn.Module):
    def __init__(self, input_dim, output_dim, middle_dim, num_groups):
        super().__init__()

        ks = 3  # kernel size
        mfd = middle_dim

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mfd = mfd
        self.num_groups = num_groups

        # ### Main body ###
        block = [
            nn.Conv1d(input_dim, mfd, 3, 1, 1),
            nn.GroupNorm(num_groups, mfd),
            nn.LeakyReLU(),
            RCBlock(mfd, ks, dilation=1, num_groups=num_groups),
            nn.Conv1d(mfd, output_dim, 3, 1, 1),
        ]
        self.block = nn.Sequential(*block)

    def forward(self, x):

        # ### Main ###
        x = self.block(x)

        return x


class Sampler(nn.Module):
    def __init__(self, input_dim, output_dim, z_scale_factors):
        super().__init__()

        mfd = 512
        num_groups = 4
        self.num_groups = num_groups
        self.mfd = mfd

        self.output_dim = output_dim
        self.input_dim = input_dim
        self.z_scale_factors = z_scale_factors

        # ### Main body ###
        self.block0 = BodyGBlock(input_dim, mfd, mfd, num_groups)
        self.head0 = nn.Conv1d(mfd, output_dim, 3, 1, 1)

        blocks = []
        heads = []
        for scale_factor in z_scale_factors:
            block = BodyGBlock(mfd, mfd, mfd, num_groups)
            blocks.append(block)

            head = nn.Conv1d(mfd, output_dim, 3, 1, 1)
            heads.append(head)

        self.blocks = nn.ModuleList(blocks)
        self.heads = nn.ModuleList(heads)

    def forward(self, z):

        # SBlock0
        z_scale_factors = self.z_scale_factors
        x_body = self.block0(z)
        x_head = self.head0(x_body)

        for ii, (block, head, scale_factor) in enumerate(zip(self.blocks, self.heads, z_scale_factors)):
            x_body = F.interpolate(x_body, scale_factor=scale_factor, mode='nearest')
            x_head = F.interpolate(x_head, scale_factor=scale_factor, mode='nearest')

            x_body = x_body + block(x_body)
            x_head = x_head + head(x_body)

        return x_head

class VQVAE(nn.Module):
    def __init__(self, codebook_size, encoder=None, decoder=None, device=None):
        super().__init__()
        self.vq = BottleneckBlock(codebook_size, 64, 0.99, device)
        self.encoder = encoder 
        self.decoder = decoder 

    def forward(self, x):
        """
        Args:
        x shape: (bs, dimension, frame_length)
        """
        x = self.encoder(x)
        x_l, x_d, commit_loss, metric = self.vq(x)
        x = self.decoder(x_d)
        return x, commit_loss, metric
    
    def encode(self, x):
        # encode as tokens (LongTensor)
        self.eval()
        x = self.encoder(x)
        x_l = self.vq.encode(x)
        return x_l
    
    def decode(self, x_l):
        # [bs, T]
        self.eval()

        N, T = x_l.shape
        x_d = self.vq.dequantise(x_l)
        x_d = x_d.view(N, T, -1).permute(0, 2, 1).contiguous()
        return self.decoder(x_d) 
    
    def restore_from_ckpt(self, hps, device):
        self.seq_len = 4096 // np.prod(hps.upsample_ratios)
        hps_name = f'{hps.vq_name}_target.pkl'
        ckpt = torch.load(
            os.path.join(hps.ckpt_dir, hps_name), 
            map_location=lambda storage, loc: storage
        )
        self.load_state_dict(ckpt['model'])
        self.to(device)
        print(f'resume from: {hps_name}')
        return torch.FloatTensor(ckpt['mean']).to(device), torch.FloatTensor(ckpt['std']).to(device)



        