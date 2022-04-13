import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class Conv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.weight)
        nn.init.zeros_(self.bias)


class RFF_MLP_Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.RFF_freq = nn.Parameter(
            16 * torch.randn([1, 32]), requires_grad=False)
        self.MLP = nn.ModuleList([
            nn.Linear(64, 128),
            nn.Linear(128, 256),
            nn.Linear(256, 512),
        ])

    def forward(self, sigma):
        """
        Arguments:
          sigma:
              (shape: [B, 1], dtype: float32)

        Returns:
          x: embedding of sigma
              (shape: [B, 512], dtype: float32)
        """
        x = self._build_RFF_embedding(sigma)
        for layer in self.MLP:
            x = F.relu(layer(x))
        return x

    def _build_RFF_embedding(self, sigma):
        """
        Arguments:
          sigma:
              (shape: [B, 1], dtype: float32)
        Returns:
          table:
              (shape: [B, 64], dtype: float32)
        """
        freqs = self.RFF_freq
        freqs = freqs.to(device=torch.device("cuda"))
        table = 2 * np.pi * sigma * freqs
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table


class Film(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.output_layer = nn.Linear(512, 2 * output_dim)

    def forward(self, sigma_encoding):
        sigma_encoding = self.output_layer(sigma_encoding)
        sigma_encoding = sigma_encoding.unsqueeze(-1)
        gamma, beta = torch.chunk(sigma_encoding, 2, dim=1)
        return gamma, beta


class UBlock(nn.Module):
    def __init__(self, input_size, hidden_size, factor, dilation, args):
        super().__init__()
        assert isinstance(dilation, (list, tuple))
        assert len(dilation) == 4
        self.args=args

        self.factor = factor
        self.residual_dense = Conv1d(input_size, hidden_size, 1)
        self.convs = nn.ModuleList([
            Conv1d(2 * input_size, hidden_size, 3,
                   dilation=dilation[0], padding=dilation[0],padding_mode=self.args.unet.padding),
            Conv1d(hidden_size, hidden_size, 3,
                   dilation=dilation[1], padding=dilation[1],padding_mode=self.args.unet.padding),
            Conv1d(hidden_size, hidden_size, 3,
                   dilation=dilation[2], padding=dilation[2],padding_mode=self.args.unet.padding),
            Conv1d(hidden_size, hidden_size, 3,
                   dilation=dilation[3], padding=dilation[3],padding_mode=self.args.unet.padding)
        ])
        self.cropcatblock=CropConcatBlock()

    def forward(self, x, x_dblock):
        height_diff = (x_dblock.shape[2] - x.shape[2])
        pad=(math.floor(height_diff/2), math.ceil(height_diff/2))
        x=torch.nn.functional.pad(x, pad, mode='circular')

        size = x_dblock.shape[-1] * self.factor

        residual = F.interpolate(x, size=size)
        residual = self.residual_dense(residual)
        x = torch.cat((x, x_dblock),1)
        x = F.leaky_relu(x, 0.2)
        x = F.interpolate(x, size=size)
        for layer in self.convs:
            x = F.leaky_relu(x, 0.2)
            x = layer(x)
        return x + residual

class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        x_len,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels

        self.use_checkpoint = use_checkpoint
        #self.norm = normalization(channels)
        self.qkv = nn.Conv1d( channels, channels * 3, 1)
        # split qkv before split heads
        self.attention = QKVAttention(self.num_heads)

        self.Posencoding= PositionalEncoding(d_model=channels,dropout=0,max_len=x_len)

        self.proj_out = nn.Conv1d( channels, channels, 1)

    #def forward(self, x):
    #    return checkpoint(self._forward, (x,), self.parameters(), True)

    def forward(self, x):
        b, c,t  = x.shape
        self.Posencoding(x) 
        x = x.reshape(b, c, -1)
        qkv = self.qkv(x)
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, t)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x=x.permute(2,0,1)
        x = x + self.pe[:x.size(0)]
        x=x.permute(1,2,0)
        return self.dropout(x)

class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)

class DBlock(nn.Module):
    def __init__(self, input_size, hidden_size, factor, args):
        super().__init__()
        self.args=args
        self.factor = factor
        self.residual_dense = Conv1d(input_size, hidden_size, 1)
        self.layer_1 = Conv1d(input_size, hidden_size,
                              3, dilation=1, padding=1, padding_mode=self.args.unet.padding)
        self.convs = nn.ModuleList([
            Conv1d(hidden_size, hidden_size, 3, dilation=2, padding=2, padding_mode=self.args.unet.padding),
            Conv1d(hidden_size, hidden_size, 3, dilation=4, padding=4,padding_mode=self.args.unet.padding),
            Conv1d(hidden_size, hidden_size, 3, dilation=8, padding=8,padding_mode=self.args.unet.padding),

        ])

    def forward(self, x, gamma, beta):
        size = x.shape[-1] // self.factor

        residual = self.residual_dense(x)
        residual = F.interpolate(residual, size=size)

        x = F.interpolate(x, size=size)
        x = F.leaky_relu(x, 0.2)
        x = self.layer_1(x)
        x = gamma * x + beta
        for layer in self.convs:
            x = F.leaky_relu(x, 0.2)
            x = layer(x)

        return x + residual


class UNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args=args
        self.conv_1 = Conv1d(1, 32, 5, padding=2,padding_mode=self.args.unet.padding )
        self.embedding = RFF_MLP_Block()
        f=[2,2,4,4,8]
        lens=args.audio_len/np.cumprod(f)
        self.downsample = nn.ModuleList([
            DBlock(32, 128, f[0], args=self.args),
            DBlock(128, 128, f[1], args=self.args),
            DBlock(128, 256, f[2], args=self.args),
            DBlock(256, 512, f[3], args=self.args),
            DBlock(512, 512, f[4], args=self.args),
        ])
        if self.args.unet.use_attention:
        
            self.attention=nn.ModuleList([
                None,
                None,
                AttentionBlock(256, int(lens[2]), num_heads=self.args.unet.num_att_heads),
                AttentionBlock(512, int(lens[3]), num_heads=self.args.unet.num_att_heads),
                AttentionBlock(512, int(lens[4]), num_heads=self.args.unet.num_att_heads)
            ])
        
        self.gamma_beta = nn.ModuleList([
            Film(128),
            Film(128),
            Film(256),
            Film(512),
            Film(512),
        ])
        self.upsample = nn.ModuleList([
            UBlock(512, 512, f[4], [1, 2, 4, 8], args=self.args),
            UBlock(512, 256, f[3], [1, 2, 4, 8], args=self.args),
            UBlock(256, 128, f[2], [1, 2, 4, 8], args=self.args),
            UBlock(128, 128, f[1], [1, 2, 4, 8], args=self.args),
            UBlock(128, 128, f[0], [1, 2, 4, 8], args=self.args),
        ])
        self.last_conv = Conv1d(128, 1, 3, padding=1)

    def forward(self, audio, sigma):
        x = audio.unsqueeze(1)
        x = self.conv_1(x)
        downsampled = []
        sigma_encoding = self.embedding(sigma)

        if self.args.unet.use_attention:
            for film, layer, att  in zip(self.gamma_beta, self.downsample, self.attention):
                gamma, beta = film(sigma_encoding)
                x = layer(x, gamma, beta)
                if not att is None:
                    x=att(x)
                downsampled.append(x)
        else:
            for film, layer  in zip(self.gamma_beta, self.downsample):
                gamma, beta = film(sigma_encoding)
                x = layer(x, gamma, beta)
                downsampled.append(x)

        for layer, x_dblock in zip(self.upsample, reversed(downsampled)):
            x = layer(x, x_dblock)

        height_diff = (audio.shape[1] - x.shape[2])
        pad=(math.floor(height_diff/2), math.ceil(height_diff/2))
        x=torch.nn.functional.pad(x, pad, mode='circular')

        x = self.last_conv(x)
        x = x.squeeze(1)
        return x

class CropConcatBlock(nn.Module):
    def forward(self, down_layer, x, **kwargs):
        x1_shape = down_layer.shape
        x2_shape = x.shape
        height_diff = (x2_shape[2] - x1_shape[2])
        pad=(math.floor(height_diff/2), math.ceil(height_diff/2))
         
        down_layer_padded=torch.nn.functional.pad(down_layer, pad, mode='circular')
        #down_layer_cropped = down_layer[:,
        #                                :,
        #                                height_diff: (x2_shape[2] + height_diff)]
        x = torch.cat((down_layer_padded, x),1)
        return x


