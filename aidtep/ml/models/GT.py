import torch
import numpy as np
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from torchinfo import summary
from torch.nn.init import xavier_uniform_, constant_, xavier_normal_
from torch.nn.parameter import Parameter
import torch.fft as fft
import copy
import math


def attention(query, key, value,
              mask=None, dropout=None, weight=None,
              attention_type='softmax'):
    '''
    Simplified from
    https://nlp.seas.harvard.edu/2018/04/03/attention.html
    Compute the Scaled Dot Product Attention
    '''

    d_k = query.size(-1)

    if attention_type == 'cosine':
        p_attn = F.cosine_similarity(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
    else:
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        seq_len = scores.size(-1)

        if attention_type == 'softmax':
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            p_attn = F.softmax(scores, dim=-1)
        elif attention_type in ['fourier', 'integral', 'local']:
            if mask is not None:
                scores = scores.masked_fill(mask == 0, 0)
            p_attn = scores / seq_len

    if dropout is not None:
        p_attn = F.dropout(p_attn)

    out = torch.matmul(p_attn, value)

    return out, p_attn

def linear_attention(query, key, value,
                     mask=None, dropout=None,
                     attention_type='galerkin'):
    '''
    Adapted from lucidrains' implementaion
    https://github.com/lucidrains/linear-attention-transformer/blob/master/linear_attention_transformer/linear_attention_transformer.py
    to https://nlp.seas.harvard.edu/2018/04/03/attention.html template
    linear_attn function
    Compute the Scaled Dot Product Attention globally
    '''

    seq_len = query.size(-2)
    if attention_type in ['linear', 'global']:
        query = query.softmax(dim=-1)
        key = key.softmax(dim=-2)
    scores = torch.matmul(key.transpose(-2, -1), value)

    if mask is not None:
        raise RuntimeError("linear attention does not support casual mask.")

    p_attn = scores / seq_len

    if dropout is not None:
        p_attn = F.dropout(p_attn)

    out = torch.matmul(query, p_attn)
    return out, p_attn

def causal_linear_attn(query, key, value, kv_mask=None, dropout=None, eps=1e-7):
    '''
    Modified from https://github.com/lucidrains/linear-attention-transformer
    '''
    bsz, n_head, seq_len, d_k, dtype = *query.shape, query.dtype

    key /= seq_len

    if kv_mask is not None:
        mask = kv_mask[:, None, :, None]
        key = key.masked_fill_(~mask, 0.)
        value = value.masked_fill_(~mask, 0.)
        del mask

    b_q, b_k, b_v = [x.reshape(bsz, n_head, -1, 1, d_k) for x in (query, key, value)]

    b_k_sum = b_k.sum(dim=-2)
    b_k_cumsum = b_k_sum.cumsum(dim=-2).type(dtype)

    p_attn = torch.einsum('bhund,bhune->bhude', b_k, b_v)
    p_attn = p_attn.cumsum(dim=-3).type(dtype)
    if dropout is not None:
        p_attn = F.dropout(p_attn)

    D_inv = 1. / torch.einsum('bhud,bhund->bhun', b_k_cumsum + eps, b_q)
    attn = torch.einsum('bhund,bhude,bhun->bhune', b_q, p_attn, D_inv)
    return attn.reshape(*query.shape), p_attn

#===============================================================================
def default(value, d):
    '''
    helper taken from https://github.com/lucidrains/linear-attention-transformer
    '''
    return d if value is None else value

class Identity(nn.Module):
    '''
    a placeholder layer similar to tensorflow.no_op():
    https://github.com/pytorch/pytorch/issues/9160#issuecomment-483048684
    not used anymore as
    https://pytorch.org/docs/stable/generated/torch.nn.Identity.html
    edge and grid are dummy inputs
    '''

    def __init__(self, in_features=None, out_features=None,
                 *args, **kwargs):
        super(Identity, self).__init__()

        if in_features is not None and out_features is not None:
            self.id = nn.Linear(in_features, out_features)
        else:
            self.id = nn.Identity()

    def forward(self, x, edge=None, grid=None):
        return self.id(x)

class Shortcut2d(nn.Module):
    '''
    (-1, in, S, S) -> (-1, out, S, S)
    Used in SimpleResBlock
    '''

    def __init__(self, in_features=None,
                 out_features=None,):
        super(Shortcut2d, self).__init__()
        self.shortcut = nn.Linear(in_features, out_features)

    def forward(self, x, edge=None, grid=None):
        x = x.permute(0, 2, 3, 1)
        x = self.shortcut(x)
        x = x.permute(0, 3, 1, 2)
        return x

class Conv2dResBlock(nn.Module):
    '''
    Conv2d + a residual block
    https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    Modified from ResNet's basic block, one conv less, no batchnorm
    No batchnorm
    '''

    def __init__(self, in_dim, out_dim,
                 kernel_size=3,
                 padding=1,
                 dilation=1,
                 dropout=0.1,
                 stride=1,
                 bias=False,
                 residual=False,
                 basic_block=False,
                 activation_type='silu'):
        super(Conv2dResBlock, self).__init__()

        activation_type = default(activation_type, 'silu')
        self.activation = nn.SiLU() if activation_type == 'silu' else nn.ReLU()
        self.add_res = residual
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim,
                      kernel_size=kernel_size,
                      padding=padding,
                      dilation=dilation,
                      stride=stride,
                      bias=bias),
            nn.Dropout(dropout),
        )
        self.basic_block = basic_block
        if self.basic_block:
            self.conv1 = nn.Sequential(
                self.activation,
                nn.Conv2d(out_dim, out_dim,
                          kernel_size=kernel_size,
                          padding=padding,
                          bias=bias),
                nn.Dropout(dropout),
            )
        self.apply_shortcut = (in_dim != out_dim)

        if self.add_res:
            if self.apply_shortcut:
                self.res = Shortcut2d(in_dim, out_dim)
            else:
                self.res = Identity()

    def forward(self, x):
        if self.add_res:
            h = self.res(x)

        x = self.conv(x)

        if self.basic_block:
            x = self.conv1(x)

        if self.add_res:
            return self.activation(x + h)
        else:
            return self.activation(x)

class Interp2dEncoder(nn.Module):
    r'''
    Using Interpolate instead of avg pool
    interp dim hard coded or using a factor
    old code uses lambda and cannot be pickled
    '''

    def __init__(self, in_dim: int,
                 out_dim: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 dilation: int = 1,
                 interp_size=None,
                 residual=False,
                 activation_type='silu',
                 dropout=0.1,
                 debug=False):
        super(Interp2dEncoder, self).__init__()

        conv_dim0 = out_dim // 3
        conv_dim1 = out_dim // 3
        conv_dim2 = int(out_dim - conv_dim0 - conv_dim1)
        padding1 = padding//2 if padding//2 >= 1 else 1
        padding2 = padding//4 if padding//4 >= 1 else 1
        activation_type = default(activation_type, 'silu')
        self.interp_size = interp_size
        self.is_scale_factor = isinstance(
            interp_size[0], float) and isinstance(interp_size[1], float)
        self.conv0 = Conv2dResBlock(in_dim, out_dim, kernel_size=kernel_size,
                                    padding=padding, activation_type=activation_type,
                                    dropout=dropout,
                                    residual=residual)
        self.conv1 = Conv2dResBlock(out_dim, conv_dim0, kernel_size=kernel_size,
                                    padding=padding1,
                                    stride=stride, residual=residual,
                                    dropout=dropout,
                                    activation_type=activation_type,)
        self.conv2 = Conv2dResBlock(conv_dim0, conv_dim1, kernel_size=kernel_size,
                                    dilation=dilation,
                                    padding=padding2, residual=residual,
                                    dropout=dropout,
                                    activation_type=activation_type,)
        self.conv3 = Conv2dResBlock(conv_dim1, conv_dim2,
                                    kernel_size=kernel_size,
                                    residual=residual,
                                    dropout=dropout,
                                    activation_type=activation_type,)
        self.activation = nn.SiLU() if activation_type == 'silu' else nn.ReLU()
        self.add_res = residual
        self.debug = debug

    def forward(self, x):
        x = self.conv0(x)
        if self.is_scale_factor:
            x = F.interpolate(x, scale_factor=self.interp_size[0],
                              mode='bilinear',
                              recompute_scale_factor=True,
                              align_corners=True)
        else:
            x = F.interpolate(x, size=self.interp_size[0],
                              mode='bilinear',
                              align_corners=True)
        x = self.activation(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        out = torch.cat([x1, x2, x3], dim=1)
        if self.add_res:
            out += x

        if self.is_scale_factor:
            out = F.interpolate(out, scale_factor=self.interp_size[1],
                                mode='bilinear',
                                recompute_scale_factor=True,
                                align_corners=True,)
        else:
            out = F.interpolate(out, size=self.interp_size[1],
                              mode='bilinear',
                              align_corners=True)
        out = self.activation(out)
        return out

class Interp2dUpsample(nn.Module):
    '''
    interpolate then Conv2dResBlock
    old code uses lambda and cannot be pickled
    temp hard-coded dimensions
    '''

    def __init__(self, in_dim: int,
                 out_dim: int,
                 kernel_size: int = 3,
                 padding: int = 1,
                 residual=False,
                 conv_block=True,
                 interp_mode='bilinear',
                 interp_size=None,
                 activation_type='silu',
                 dropout=0.1,
                 debug=False):
        super(Interp2dUpsample, self).__init__()
        activation_type = default(activation_type, 'silu')
        self.activation = nn.SiLU() if activation_type == 'silu' else nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        if conv_block:
            self.conv = nn.Sequential(Conv2dResBlock(
                in_dim, out_dim,
                kernel_size=kernel_size,
                padding=padding,
                residual=residual,
                dropout=dropout,
                activation_type=activation_type),
                self.dropout,
                self.activation)
        self.conv_block = conv_block
        self.interp_size = interp_size
        self.interp_mode = interp_mode
        self.debug = debug

    def forward(self, x):
        x = F.interpolate(x, size=self.interp_size[0],
                          mode=self.interp_mode,
                          align_corners=True)
        if self.conv_block:
            x = self.conv(x)
        x = F.interpolate(x, size=self.interp_size[1],
                          mode=self.interp_mode,
                          align_corners=True)
        return x

class SpectralConv2d(nn.Module):
    def __init__(self, in_dim,
                 out_dim,
                 modes: int,  # number of fourier modes
                 n_grid=None,
                 dropout=0.1,
                 norm='ortho',
                 activation='silu',
                 return_freq=False,  # whether to return the frequency target
                 debug=False):
        super(SpectralConv2d, self).__init__()

        '''
        Modified Zongyi Li's SpectralConv2d PyTorch 1.6 code
        using only real weights
        https://github.com/zongyi-li/fourier_neural_operator/blob/master/fourier_2d.py
        '''
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = nn.Linear(in_dim, out_dim)  # for residual
        self.modes = modes
        activation = default(activation, 'silu')
        self.activation = nn.SiLU() if activation == 'silu' else nn.ReLU()
        self.n_grid = n_grid  # just for debugging
        self.fourier_weight = nn.ParameterList([Parameter(
            torch.FloatTensor(in_dim, out_dim,
                                                modes, modes, 2)) for _ in range(2)])
        for param in self.fourier_weight:
            xavier_normal_(param, gain=1/(in_dim*out_dim)
                           * np.sqrt(in_dim+out_dim))
        self.dropout = nn.Dropout(dropout)
        self.norm = norm
        self.return_freq = return_freq
        self.debug = debug

    @staticmethod
    def complex_matmul_2d(a, b):
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        op = partial(torch.einsum, "bixy,ioxy->boxy")
        return torch.stack([
            op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
            op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
        ], dim=-1)

    def forward(self, x):
        '''
        Input: (-1, n_grid**2, in_features) or (-1, n_grid, n_grid, in_features)
        Output: (-1, n_grid**2, out_features) or (-1, n_grid, n_grid, out_features)
        '''
        batch_size = x.size(0)
        n_dim = x.ndim
        if n_dim == 4:
            n = x.size(1)
            assert x.size(1) == x.size(2)
        elif n_dim == 3:
            n = int(x.size(1)**(0.5))
        else:
            raise ValueError("Dimension not implemented")
        in_dim = self.in_dim
        out_dim = self.out_dim
        modes = self.modes

        x = x.view(-1, n, n, in_dim)
        res = self.linear(x)
        x = self.dropout(x)

        x = x.permute(0, 3, 1, 2)
        x_ft = fft.rfft2(x, s=(n, n), norm=self.norm)
        x_ft = torch.stack([x_ft.real, x_ft.imag], dim=-1)

        out_ft = torch.zeros(batch_size, out_dim, n, n //
                             2+1, 2, device=x.device)
        out_ft[:, :, :modes, :modes] = self.complex_matmul_2d(
            x_ft[:, :, :modes, :modes], self.fourier_weight[0])
        out_ft[:, :, -modes:, :modes] = self.complex_matmul_2d(
            x_ft[:, :, -modes:, :modes], self.fourier_weight[1])
        out_ft = torch.complex(out_ft[..., 0], out_ft[..., 1])

        x = fft.irfft2(out_ft, s=(n, n), norm=self.norm)
        x = x.permute(0, 2, 3, 1)
        x = self.activation(x + res)

        if n_dim == 3:
            x = x.view(batch_size, n**2, out_dim)

        if self.return_freq:
            return x, out_ft
        else:
            return x

class DownScaler(nn.Module):
    def __init__(self, in_dim=1,
                 out_dim=128,
                 dropout=0.05,
                 activation_type='silu',
                 interp_size=[0.7071, 0.7072],
                 debug=False):
        super(DownScaler, self).__init__()
        '''
        A wrapper for conv2d/interp downscaler
        '''

        self.downsample = Interp2dEncoder(in_dim=in_dim,
                                          out_dim=out_dim,
                                          interp_size=interp_size,
                                          activation_type=activation_type,
                                          dropout=dropout,
                                          debug=debug)

        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x):
        '''
        2D:
            Input: (-1, n, n, in_dim)
            Output: (-1, n_s, n_s, out_dim)
        '''
        n_grid = x.size(1)
        bsz = x.size(0)
        x = x.view(bsz, n_grid, n_grid, self.in_dim)
        x = x.permute(0, 3, 1, 2)
        x = self.downsample(x)
        x = x.permute(0, 2, 3, 1)
        return x

class UpScaler(nn.Module):
    def __init__(self, in_dim=128,
                 out_dim=128,
                 dropout=0.1,
                 activation_type='silu',
                 interp_mode='bilinear',
                 interp_size=None,
                 debug=False):
        super(UpScaler, self).__init__()
        '''
        A wrapper for DeConv2d upscaler or interpolation upscaler
        Deconv: Conv1dTranspose
        Interp: interp->conv->interp
        '''

        self.upsample = Interp2dUpsample(in_dim=in_dim,
                                         out_dim=out_dim,
                                         interp_mode=interp_mode,
                                         interp_size=interp_size,
                                         dropout=dropout,
                                         activation_type=activation_type,
                                         debug=debug)


    def forward(self, x):
        '''
        2D:
            Input: (-1, n_s, n_s, in_dim)
            Output: (-1, n, n, out_dim)
        '''
        x = x.permute(0, 3, 1, 2)
        x = self.upsample(x)
        x = x.permute(0, 2, 3, 1)
        return x

class SpectralRegressor(nn.Module):
    def __init__(self, in_dim=128,
                 n_hidden=32,
                 freq_dim=32,
                 out_dim=1,
                 modes=12,
                 num_spectral_layers: int = 2,
                 n_grid=None,
                 dim_feedforward=None,
                 spacial_fc=True,
                 spacial_dim=2,
                 return_freq=False,
                 return_latent=False,
                 normalizer=None,
                 activation='silu',
                 last_activation=True,
                 dropout=0,
                 debug=False):
        super(SpectralRegressor, self).__init__()
        '''
        A wrapper for both SpectralConv1d and SpectralConv2d
        Ref: Li et 2020 FNO paper
        https://github.com/zongyi-li/fourier_neural_operator/blob/master/fourier_2d.py
        A new implementation incoporating all spacial-based FNO
        in_dim: input dimension, (either n_hidden or spacial dim)
        n_hidden: number of hidden features out from attention to the fourier conv
        '''
        if spacial_dim == 2:  # 2d, function + (x,y)
            spectral_conv = SpectralConv2d
        else:
            raise NotImplementedError("3D not implemented.")
        activation = default(activation, 'silu')
        self.activation = nn.SiLU() if activation == 'silu' else nn.ReLU()
        dropout = default(dropout, 0.1)
        self.spacial_fc = spacial_fc  # False in Transformer
        if self.spacial_fc:
            self.fc = nn.Linear(in_dim + spacial_dim, n_hidden)
        self.spectral_conv = nn.ModuleList([spectral_conv(in_dim=n_hidden,
                                                          out_dim=freq_dim,
                                                          n_grid=n_grid,
                                                          modes=modes,
                                                          dropout=dropout,
                                                          activation=activation,
                                                          return_freq=return_freq,
                                                          debug=debug)])
        for _ in range(num_spectral_layers - 1):
            self.spectral_conv.append(spectral_conv(in_dim=freq_dim,
                                                    out_dim=freq_dim,
                                                    n_grid=n_grid,
                                                    modes=modes,
                                                    dropout=dropout,
                                                    activation=activation,
                                                    return_freq=return_freq,
                                                    debug=debug))
        if not last_activation:
            self.spectral_conv[-1].activation = Identity()

        self.n_grid = n_grid  # dummy for debug
        self.dim_feedforward = default(dim_feedforward, 2*spacial_dim*freq_dim)
        self.regressor = nn.Sequential(
            nn.Linear(freq_dim, self.dim_feedforward),
            self.activation,
            nn.Linear(self.dim_feedforward, out_dim),
        )
        self.normalizer = normalizer
        self.return_freq = return_freq
        self.return_latent = return_latent
        self.debug = debug

    def forward(self, x, grid=None):
        '''
        2D:
            Input: (-1, n, n, in_features)
            Output: (-1, n, n, n_targets)
        1D:
            Input: (-1, n, in_features)
            Output: (-1, n, n_targets)
        '''
        x_latent = []
        x_fts = []

        if self.spacial_fc:
            x = torch.cat([x, grid], dim=-1)
            x = self.fc(x)

        for layer in self.spectral_conv:
            if self.return_freq:
                x, x_ft = layer(x)
                x_fts.append(x_ft.contiguous())
            else:
                x = layer(x)

            if self.return_latent:
                x_latent.append(x.contiguous())

        x = self.regressor(x)

        if self.normalizer:
            x = self.normalizer.inverse_transform(x)

        if self.return_freq or self.return_latent:
            return x, dict(preds_freq=x_fts, preds_latent=x_latent)
        else:
            return x



class SimpleAttention(nn.Module):
    def __init__(self, n_head, d_model,
                 pos_dim: int = 1,
                 attention_type='fourier',
                 dropout=0.1,
                 xavier_init=1e-4,
                 diagonal_weight=1e-2,
                 symmetric_init=False,
                 norm=False,
                 norm_type='layer',
                 eps=1e-5,
                 debug=False):
        super(SimpleAttention, self).__init__()
        assert d_model % n_head == 0
        self.attention_type = attention_type
        self.d_k = d_model // n_head
        self.n_head = n_head
        self.pos_dim = pos_dim
        self.linears = nn.ModuleList(
            [copy.deepcopy(nn.Linear(d_model, d_model)) for _ in range(3)])
        self.xavier_init = xavier_init
        self.diagonal_weight = diagonal_weight
        self.symmetric_init = symmetric_init
        if self.xavier_init > 0:
            self._reset_parameters()
        self.add_norm = norm
        self.norm_type = norm_type
        if norm:
            self._get_norm(eps=eps)

        if pos_dim > 0:
            self.fc = nn.Linear(d_model + n_head*pos_dim, d_model)

        self.attn_weight = None
        self.dropout = nn.Dropout(dropout)
        self.debug = debug

    def forward(self, query, key, value, pos=None, mask=None, weight=None):
        if mask is not None:
            mask = mask.unsqueeze(1)

        bsz = query.size(0)
        if weight is not None:
            query, key = weight*query, weight*key

        query, key, value = \
            [layer(x).view(bsz, -1, self.n_head, self.d_k).transpose(1, 2)
             for layer, x in zip(self.linears, (query, key, value))]

        if self.add_norm:
            if self.attention_type in ['linear', 'galerkin', 'global']:
                if self.norm_type == 'instance':
                    key, value = key.transpose(-2, -1), value.transpose(-2, -1)

                key = torch.stack(
                    [norm(x) for norm, x in
                     zip(self.norm_K, (key[:, i, ...] for i in range(self.n_head)))], dim=1)
                value = torch.stack(
                    [norm(x) for norm, x in
                     zip(self.norm_V, (value[:, i, ...] for i in range(self.n_head)))], dim=1)

                if self.norm_type == 'instance':
                    key, value = key.transpose(-2, -1), value.transpose(-2, -1)
            else:
                if self.norm_type == 'instance':
                    key, query = key.transpose(-2, -1), query.transpose(-2, -1)

                key = torch.stack(
                    [norm(x) for norm, x in
                     zip(self.norm_K, (key[:, i, ...] for i in range(self.n_head)))], dim=1)
                query = torch.stack(
                    [norm(x) for norm, x in
                     zip(self.norm_Q, (query[:, i, ...] for i in range(self.n_head)))], dim=1)

                if self.norm_type == 'instance':
                    key, query = key.transpose(-2, -1), value.transpose(-2, -1)

        if pos is not None and self.pos_dim > 0:
            assert pos.size(-1) == self.pos_dim
            pos = pos.unsqueeze(1)
            pos = pos.repeat([1, self.n_head, 1, 1])
            query, key, value = [torch.cat([pos, x], dim=-1)
                                 for x in (query, key, value)]

        if self.attention_type in ['linear', 'galerkin', 'global']:
            x, self.attn_weight = linear_attention(query, key, value,
                                                   mask=mask,
                                                   attention_type=self.attention_type,
                                                   dropout=self.dropout)
        elif self.attention_type == 'causal':
            assert mask is not None
            x, self.attn_weight = causal_linear_attn(query, key, value,
                                                   mask=mask,
                                                   dropout=self.dropout)
        else:
            x, self.attn_weight = attention(query, key, value,
                                            mask=mask,
                                            attention_type=self.attention_type,
                                            dropout=self.dropout)

        out_dim = self.n_head * self.d_k if pos is None else self.n_head * \
            (self.d_k + self.pos_dim)
        att_output = x.transpose(1, 2).contiguous().view(bsz, -1, out_dim)

        if pos is not None and self.pos_dim > 0:
            att_output = self.fc(att_output)

        return att_output, self.attn_weight

    def _reset_parameters(self):
        for param in self.linears.parameters():
            if param.ndim > 1:
                xavier_uniform_(param, gain=self.xavier_init)
                if self.diagonal_weight > 0.0:
                    param.data += self.diagonal_weight * \
                        torch.diag(torch.ones(
                            param.size(-1), dtype=torch.float))
                if self.symmetric_init:
                    param.data += param.data.T
                    # param.data /= 2.0
            else:
                constant_(param, 0)

    def _get_norm(self, eps):
        if self.attention_type in ['linear', 'galerkin', 'global']:
            if self.norm_type == 'instance':
                self.norm_K = self._get_instancenorm(self.d_k, self.n_head,
                                                     eps=eps,
                                                     affine=True)
                self.norm_V = self._get_instancenorm(self.d_k, self.n_head,
                                                     eps=eps,
                                                     affine=True)
            elif self.norm_type == 'layer':
                self.norm_K = self._get_layernorm(self.d_k, self.n_head,
                                                  eps=eps)
                self.norm_V = self._get_layernorm(self.d_k, self.n_head,
                                                  eps=eps)
        else:
            if self.norm_type == 'instance':
                self.norm_K = self._get_instancenorm(self.d_k, self.n_head,
                                                     eps=eps,
                                                     affine=True)
                self.norm_Q = self._get_instancenorm(self.d_k, self.n_head,
                                                     eps=eps,
                                                     affine=True)
            elif self.norm_type == 'layer':
                self.norm_K = self._get_layernorm(self.d_k, self.n_head,
                                                  eps=eps)
                self.norm_Q = self._get_layernorm(self.d_k, self.n_head,
                                                  eps=eps)

    @staticmethod
    def _get_layernorm(normalized_dim, n_head, **kwargs):
        return nn.ModuleList(
            [copy.deepcopy(nn.LayerNorm(normalized_dim, **kwargs)) for _ in range(n_head)])

    @staticmethod
    def _get_instancenorm(normalized_dim, n_head, **kwargs):
        return nn.ModuleList(
            [copy.deepcopy(nn.InstanceNorm1d(normalized_dim, **kwargs)) for _ in range(n_head)])

class FeedForward(nn.Module):
    def __init__(self, in_dim=256,
                 dim_feedforward: int = 1024,
                 out_dim=None,
                 batch_norm=False,
                 activation='relu',
                 dropout=0.1):
        super(FeedForward, self).__init__()
        out_dim = default(out_dim, in_dim)
        n_hidden = dim_feedforward
        self.lr1 = nn.Linear(in_dim, n_hidden)

        if activation == 'silu':
            self.activation = nn.SiLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()

        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bn = nn.BatchNorm1d(n_hidden)
        self.lr2 = nn.Linear(n_hidden, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.activation(self.lr1(x))
        x = self.dropout(x)
        if self.batch_norm:
            x = x.permute((0, 2, 1))
            x = self.bn(x)
            x = x.permute((0, 2, 1))
        x = self.lr2(x)
        return x

class FourierTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=96,
                 pos_dim=1,
                 n_head=2,
                 dim_feedforward=512,
                 attention_type='fourier',
                 pos_emb=False,
                 layer_norm=True,
                 attn_norm=None,
                 norm_type='layer',
                 norm_eps=None,
                 batch_norm=False,
                 attn_weight=False,
                 xavier_init: float=1e-2,
                 diagonal_weight: float=1e-2,
                 symmetric_init=False,
                 residual_type='add',
                 activation_type='relu',
                 dropout=0.1,
                 ffn_dropout=None,
                 debug=False,
                 ):
        super(FourierTransformerEncoderLayer, self).__init__()

        dropout = default(dropout, 0.05)
        if attention_type in ['linear', 'softmax']:
            dropout = 0.1
        ffn_dropout = default(ffn_dropout, dropout)
        norm_eps = default(norm_eps, 1e-5)
        attn_norm = default(attn_norm, not layer_norm)
        if (not layer_norm) and (not attn_norm):
            attn_norm = True
        norm_type = default(norm_type, 'layer')

        self.attn = SimpleAttention(n_head=n_head,
                                    d_model=d_model,
                                    attention_type=attention_type,
                                    diagonal_weight=diagonal_weight,
                                    xavier_init=xavier_init,
                                    symmetric_init=symmetric_init,
                                    pos_dim=pos_dim,
                                    norm=attn_norm,
                                    norm_type=norm_type,
                                    eps=norm_eps,
                                    dropout=dropout)
        self.d_model = d_model
        self.n_head = n_head
        self.pos_dim = pos_dim
        self.add_layer_norm = layer_norm
        if layer_norm:
            self.layer_norm1 = nn.LayerNorm(d_model, eps=norm_eps)
            self.layer_norm2 = nn.LayerNorm(d_model, eps=norm_eps)
        dim_feedforward = default(dim_feedforward, 2*d_model)
        self.ff = FeedForward(in_dim=d_model,
                              dim_feedforward=dim_feedforward,
                              batch_norm=batch_norm,
                              activation=activation_type,
                              dropout=ffn_dropout,
                              )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.residual_type = residual_type  # plus or minus

        self.debug = debug
        self.attn_weight = attn_weight
        self.__name__ = attention_type.capitalize() + 'TransformerEncoderLayer'

    def forward(self, x, pos=None, weight=None):
        '''
        - x: node feature, (batch_size, seq_len, n_feats)
        - pos: position coords, needed in every head

        Remark:
            - for n_head=1, no need to encode positional
            information if coords are in features
        '''

        if pos is not None and self.pos_dim > 0:
            att_output, attn_weight = self.attn(
                x, x, x, pos=pos, weight=weight)  # encoder no mask
        else:
            att_output, attn_weight = self.attn(x, x, x, weight=weight)

        if self.residual_type in ['add', 'plus'] or self.residual_type is None:
            x = x + self.dropout1(att_output)
        else:
            x = x - self.dropout1(att_output)
        if self.add_layer_norm:
            x = self.layer_norm1(x)

        x1 = self.ff(x)
        x = x + self.dropout2(x1)

        if self.add_layer_norm:
            x = self.layer_norm2(x)

        if self.attn_weight:
            return x, attn_weight
        else:
            return x

class GT(nn.Module):
    def __init__(self, R_dic):
        super(GT, self).__init__()
        self.downsample = DownScaler(in_dim=R_dic['in_dim'],interp_size=R_dic['down_size'])
        self.upsample = UpScaler(interp_size=R_dic['up_size'])
        self.fno = SpectralRegressor()

        self.pos_f = torch.tensor(self.get_grid(R_dic['res_input']), dtype=torch.float32)
        self.pos_c = torch.tensor(self.get_grid(R_dic['res_att']), dtype=torch.float32)

        if 'y_norm' in R_dic:
            self.y_norm = R_dic['y_norm']
        else:
            self.y_norm = None

        encoder_layer = FourierTransformerEncoderLayer(d_model=128,
                                                       n_head=4,
                                                       attention_type='galerkin',
                                                       dim_feedforward=256,
                                                       layer_norm=False,
                                                       attn_norm=True,
                                                       batch_norm=False,
                                                       pos_dim=2,
                                                       xavier_init=0.01,
                                                       diagonal_weight=0.02,
                                                       symmetric_init=False,
                                                       attn_weight=False,
                                                       dropout=0.05,
                                                       ffn_dropout=0.05,
                                                       norm_eps=0.0000001,
                                                       debug=False)
        self.encoder_layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(6)])

        self.dpo = nn.Dropout(0.0)

    def forward(self, x):
        x = x.permute(0,2,3,1)
        self.pos_f = self.pos_f.to(x.device)
        self.pos_c = self.pos_c.to(x.device)
        bsz = x.shape[0]
        pos_f = self.pos_f.unsqueeze(0)
        pos_f = pos_f.repeat(bsz, 1, 1, 1)
        res_c = self.pos_c.shape[1]
        pos_c = self.pos_c.unsqueeze(0)
        pos_c = pos_c.repeat(bsz, 1, 1, 1).view(bsz, -1, 2)

        x = self.downsample(x)
        x = x.view(bsz, -1, 128)
        x = self.dpo(x)
        for encoder in self.encoder_layers:
            x = encoder(x, pos=pos_c)

        x = x.view(bsz, res_c, res_c, 128)
        x = self.upsample(x)
        x = self.dpo(x)

        x = self.fno(x, pos_f)

        if self.y_norm is not None:
            x = torch.squeeze(x, dim=-1)
            x = self.y_norm.decode(x)
            x = torch.unsqueeze(x, dim=-1)

        return x

    def get_grid(self, n_grid, subsample=1):
        x = np.linspace(0, 1, n_grid)
        y = np.linspace(0, 1, n_grid)
        x, y = np.meshgrid(x, y)
        s = subsample

        x = x[::s, ::s]
        y = y[::s, ::s]
        grid = np.stack([x, y], axis=-1)
        return grid
#===============================================================================
