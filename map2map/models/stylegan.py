import math
import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal

from .style import ModulatedConv3d
from .resample import Resampler
from .narrow import narrow_by

def normalization(channels):
    # For channels <= 32, use GroupNorm with channels/4 groups
    # This ensures each group has at least 4 channels
    if channels <= 32:
        num_groups = max(1, channels // 4)
    else:
        num_groups = 32
    
    return nn.GroupNorm(num_groups=num_groups, num_channels=channels, eps=1e-5, affine=True)


# def normalization(channels):
#     return nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-16, affine=True)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def narrow_as(x, y):
    """
    _summary_

    Args:
        x (tensor): tensor to narrow, shape must be larger than y
        y (tensor): tensor to narrow to

    Returns:
        _type_: new x that is narrow to y
    """
    # x N C D H W
    # y N C d h w
    if x.size() == y.size():
        return x
    else:
        edge = (x.size()[-1] - y.size()[-1]) // 2
        for d in range(2, x.dim()):
            x = x.narrow(d, edge, x.size()[d] - 2 * edge)
        return x


class NoiseInjection(nn.Module):
    """Add or concatenate noise.

    Add noise if `cat=False`.
    The number of channels `chan` should be 1 (StyleGAN2)
    or that of the input (StyleGAN).
    """

    def __init__(self):
        super().__init__()

        self.std = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x, noise=None):
        batch, channels, height, width, depth = x.size()
        if noise is None:
            noise = torch.randn_like(x[:, :1]).to(x.device)
        std_shape = (-1,) + (1,) * (x.dim() - 2)
        noise = self.std.view(std_shape) * noise
        return x + noise
 

class HBlock(nn.Module):
    """The "H" block of the StyleGAN2 generator.

        x_p                     y_p
         |                       |
    convolution           linear upsample
         |                       |
          >--- projection ------>+
         |                       |
         v                       v
        x_n                     y_n

    See Fig. 7 (b) upper in https://arxiv.org/abs/1912.04958
    Upsampling are all linear, not transposed convolution.

    Parameters
    ----------
    prev_chan : number of channels of x_p
    next_chan : number of channels of x_n
    out_chan : number of channels of y_p and y_n
    embedding_size : size of embedding

    Notes
    -----
    next_size = 2 * prev_size - 6
    """

    def __init__(
        self,
        prev_chan,
        next_chan,
        out_chan,
        embedding_size,
        inject_noise=True,
        use_normalize=False,
        use_attention=False,
        use_pixel_shuffle=False,
        which_attention="linear",
    ):
        super().__init__()

        self.embedding_size = embedding_size
        self.inject_noise = inject_noise
        self.upsample = Resampler(ndim=3, scale_factor=2, narrow=True)
        self.use_normalize = use_normalize
        self.use_attention = use_attention
        self.use_pixel_shuffle = use_pixel_shuffle
        
        
        if self.use_pixel_shuffle:
            self.x_upconv = nn.Conv3d(
                prev_chan,
                prev_chan * 8,
                kernel_size=3,
                padding=1,
            )
            self.pixel_shuffle = PixelShuffle3D(upscale_factor=2)

        if self.inject_noise:
            self.noise1 = NoiseInjection()
            self.noise2 = NoiseInjection()

        if self.use_normalize:
            self.normalize1 = normalization(next_chan)
            self.normalize2 = normalization(next_chan)

        self.conv1 = ModulatedConv3d(
            in_chan=prev_chan,
            out_chan=next_chan,
            embedding_size=embedding_size,
            kernel_size=3,
            demodulation=True,
        )
        self.act1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        if use_attention:
            if which_attention == "self":
                self.attention = SelfAttention3D(next_chan)
            elif which_attention == "linear":
                self.attention = LinearAttention3D(next_chan)

        self.conv2 = ModulatedConv3d(
            in_chan=next_chan,
            out_chan=next_chan,
            embedding_size=embedding_size,
            kernel_size=3,
            demodulation=True,
        )
        self.act2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.proj = ModulatedConv3d(
            in_chan=next_chan,
            out_chan=out_chan,
            embedding_size=embedding_size,
            kernel_size=1,
            demodulation=True,
            )
        
        self.proj_act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x, y, s):
        # left branch:
        if self.use_pixel_shuffle:
            x = self.x_upconv(x, s)
            x = self.pixel_shuffle(x)
            x = narrow_by(x, 1)
        else:
            x = self.upsample(x)
        # block 1
        # ---------------------
        x = self.conv1(x, s)
        if self.inject_noise:
            x = self.noise1(x)
        if self.use_normalize:
            x = self.normalize1(x)
        x = self.act1(x)
        # ---------------------
        # block 2
        # ---------------------
        x = self.conv2(x, s)
        if self.inject_noise:
            x = self.noise2(x)
        if self.use_normalize:
            x = self.normalize2(x)
        x = self.act2(x)
        if self.use_attention:
            x = self.attention(x)

        # ---------------------
        # right branch
        y = self.upsample(y)
        y = narrow_as(y, x)
        y = y + self.proj_act(self.proj(x, s))
        return x, y


class PixelShuffle3D(nn.Module):
    """3D pixel shuffle layer for upsampling."""
    def __init__(self, upscale_factor):
        super().__init__()
        self.upscale_factor = upscale_factor
        
    def forward(self, x):
        batch_size, channels, depth, height, width = x.size()
        r = self.upscale_factor
        
        # Reshape to prepare for shuffling
        out_channels = channels // (r**3)
        x = x.view(batch_size, out_channels, r, r, r, depth, height, width)
        
        # Permute and reshape to get the upscaled output
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        x = x.view(batch_size, out_channels, depth*r, height*r, width*r)
        
        return x


class SelfAttention3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv3d(channels, channels // 8, 1)
        self.key = nn.Conv3d(channels, channels // 8, 1)
        self.value = nn.Conv3d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch, c, d, h, w = x.size()

        # Flatten spatial dimensions for attention calculation
        q = self.query(x).view(batch, -1, d * h * w).permute(0, 2, 1)  # B, DHW, C/8
        k = self.key(x).view(batch, -1, d * h * w)  # B, C/8, DHW
        v = self.value(x).view(batch, -1, d * h * w)  # B, C, DHW

        # Calculate attention scores
        attention = torch.bmm(q, k)  # B, DHW, DHW
        attention = torch.softmax(attention, dim=2)

        # Apply attention to values
        out = torch.bmm(v, attention.permute(0, 2, 1))  # B, C, DHW
        out = out.view(batch, c, d, h, w)

        return self.gamma * out + x  # Residual connection with learnable weight
    
    
class LinearAttention3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv3d(channels, channels//8, 1)
        self.key = nn.Conv3d(channels, channels//8, 1)
        self.value = nn.Conv3d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        batch, c, d, h, w = x.size()
        
        # Generate feature maps
        q = self.query(x)    # B, C/8, D, H, W
        k = self.key(x)      # B, C/8, D, H, W
        v = self.value(x)    # B, C, D, H, W
        
        # Apply softmax along channel dimension
        q = q.view(batch, -1, d*h*w)           # B, C/8, DHW
        k = self.softmax(k.view(batch, -1, d*h*w))  # B, C/8, DHW
        v = v.view(batch, -1, d*h*w)           # B, C, DHW
        
        # Linear attention computation (avoid explicit DHW×DHW matrix)
        context = torch.bmm(v, k.transpose(1, 2))   # B, C, C/8
        attention = torch.bmm(context, q)           # B, C, DHW
        
        # Reshape and combine with input
        attention = attention.view(batch, c, d, h, w)
        return x + self.gamma * attention


class G(nn.Module):
    def __init__(
        self,
        in_chan,
        out_chan,
        style_size,
        embedding_size=16,
        scale_factor=8,
        previous_scale_factor=2,
        chan_base=512,
        chan_min=64,
        chan_max=512,
        inject_noise=True,
        use_normalize=False,
        use_attention=False,
        **kwargs
    ):
        super().__init__()

        self.in_chan = in_chan
        self.out_chan = out_chan
        self.style_size = style_size
        self.embedding_size = embedding_size
        self.scale_factor = scale_factor
        num_blocks = round(math.log2(self.scale_factor))
        self.num_blocks = num_blocks
        print(f"num_blocks: {num_blocks}")
        self.inject_noise = inject_noise
        self.use_normalize = use_normalize
        assert chan_min <= chan_max

        def chan(b):
            c = chan_base >> b
            c = max(c, chan_min)
            c = min(c, chan_max)
            return c

        self.head = ModulatedConv3d(
            in_chan=in_chan,
            out_chan=chan(0),
            embedding_size=embedding_size,
            kernel_size=1,
            demodulation=True,
        )
        self.head_act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.style_embed = nn.Sequential(
            nn.Linear(style_size, embedding_size * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(embedding_size * 2, embedding_size * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(embedding_size * 2, embedding_size),
        )

        self.blocks = nn.ModuleList()
        for b in range(num_blocks):
            prev_chan, next_chan = chan(b), chan(b + 1)
            use_attn_in_this_block = False
            if use_attention:
                if num_blocks <= 3 and b == 1:  # Middle block for small networks
                    use_attn_in_this_block = True
                elif num_blocks > 3 and (b == 1 or b == 2):  # Middle blocks for larger networks
                    use_attn_in_this_block = True

            self.blocks.append(
                HBlock(
                    prev_chan=prev_chan,
                    next_chan=next_chan,
                    out_chan=out_chan,
                    embedding_size=embedding_size,
                    inject_noise=inject_noise,
                    use_normalize=use_normalize,
                    use_attention=use_attn_in_this_block
                )
            )
            
            
        # freeze the previous scale factor blocks
        if previous_scale_factor > 0:
            # Calculate how many blocks to freeze
            prev_num_blocks = round(math.log2(previous_scale_factor))
            print(f"prev_num_blocks: {prev_num_blocks}")
            # Freeze only the EARLY blocks, not head/style embedding
            for b in range(prev_num_blocks):
                for param in self.blocks[b].parameters():
                    param.requires_grad = False
                    
            # Keep head and style_embed trainable - they need to adapt to new layers
            # self.head.requires_grad_(True)  # This is default
            # self.style_embed.requires_grad_(True)  # This is default
            
            print(f"Frozen {prev_num_blocks} blocks for progressive training")

    def forward(self, x, style):
        s = self.style_embed(style)

        y = x  # direct from the input without toRGB
        x = self.head(x, s)  # shallow feature extraction
        x = self.head_act(x)

        for block in self.blocks:
            x, y = block(x, y, s)

        return y

class G_T(nn.Module):
    def __init__(
        self,
        in_chan,
        out_chan,
        style_size,
        embedding_size=16,
        scale_factor=8,
        chan_base=512,
        chan_min=64,
        chan_max=512,
        inject_noise=True,
        **kwargs
    ):
        super().__init__()

        self.in_chan = in_chan
        self.out_chan = out_chan
        self.style_size = style_size
        self.embedding_size = embedding_size
        self.scale_factor = scale_factor
        num_blocks = round(math.log2(self.scale_factor))
        self.num_blocks = num_blocks
        self.inject_noise = inject_noise

        assert chan_min <= chan_max

        def chan(b):
            c = chan_base >> b
            c = max(c, chan_min)
            c = min(c, chan_max)
            return c

        self.head = ModulatedConv3d(
            in_chan=in_chan,
            out_chan=chan(0),
            embedding_size=embedding_size,
            kernel_size=1,
            demodulation=True,
        )
        self.head_act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.style_embed = nn.Sequential(
            nn.Linear(style_size, embedding_size),
            nn.SiLU(),
            nn.Linear(embedding_size, embedding_size),
        )

        self.blocks = nn.ModuleList()
        for b in range(num_blocks):
            prev_chan, next_chan = chan(b), chan(b + 1)
            self.blocks.append(
                HBlock(
                    prev_chan=prev_chan,
                    next_chan=next_chan,
                    out_chan=out_chan,
                    embedding_size=embedding_size,
                    inject_noise=inject_noise,
                )
            )

    def forward(self, x, style):
        s = self.style_embed(style)

        y = x  # direct from the input without toRGB
        x = self.head(x, s)  # shallow feature extraction
        x = self.head_act(x)

        for block in self.blocks:
            x, y = block(x, y, s)

        return y



class ModulatedResidualBlock(nn.Module):
    def __init__(
        self,
        in_chan,
        out_chan,
        embedding_size,
        kernel_size=3,
        stride=1,
        use_normalize=False,
    ):
        super().__init__()
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.embedding_size = embedding_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.use_normalize = use_normalize
        
        if self.use_normalize:
            self.normalize1 = normalization(out_chan)
            self.normalize2 = normalization(out_chan)

        self.conv1 = ModulatedConv3d(
            in_chan=in_chan,
            out_chan=out_chan,
            embedding_size=embedding_size,
            kernel_size=kernel_size,
            stride=stride,
            demodulation=True,
        )
        self.act1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv2 = ModulatedConv3d(
            in_chan=out_chan,
            out_chan=out_chan,
            embedding_size=embedding_size,
            kernel_size=kernel_size,
            stride=stride,
            demodulation=True,
        )
        self.act2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        self.layer_scale = nn.Parameter(torch.ones(1) * 1e-5, requires_grad=True)

        self.skip = ModulatedConv3d(
            in_chan=in_chan,
            out_chan=out_chan,
            embedding_size=embedding_size,
            kernel_size=1,
            stride=1,
            demodulation=True,
        )

    def forward(self, x, style):
        # skip branch
        skip = self.skip(x, style)
        # main branch
        x = self.conv1(x, style)
        if self.use_normalize:
            x = self.normalize1(x)
        x = self.act1(x)
        x = self.conv2(x, style)
        if self.use_normalize:
            x = self.normalize2(x)
        x = self.act2(x)
        x = x * self.layer_scale
        skip = narrow_as(skip, x)

        return x + skip


class D(nn.Module):
    def __init__(
        self,
        in_chan,
        out_chan,
        style_size,
        embedding_size=16,
        scale_factor=8,
        chan_base=512,
        chan_min=64,
        chan_max=512,
        use_normalize=False,
        **kwargs
    ):
        super().__init__()

        self.in_chan = in_chan
        self.out_chan = out_chan
        self.style_size = style_size
        # self.scale_factor = scale_factor
        self.scale_factor = 8 # WARNING: hardcoded for fixed discriminator architecture
        num_blocks = round(math.log2(self.scale_factor))
        self.num_blocks = num_blocks
        self.embedding_size = embedding_size
        self.use_normalize = use_normalize
        assert chan_min <= chan_max
        def chan(b):
            if b >= 0:
                c = chan_base >> b
            else:
                c = chan_base << -b
            c = max(c, chan_min)
            c = min(c, chan_max)
            return c

        self.head = ModulatedConv3d(
            in_chan=in_chan
            + 8,  # FIXME here I hard coded the in_chan+8 to meet the dimension after eul_scale_factor 2
            out_chan=chan(num_blocks),
            embedding_size=embedding_size,
            kernel_size=1,
        )
        self.head_act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.style_embed = nn.Sequential(
            nn.Linear(style_size, embedding_size),
            nn.SiLU(),
            nn.Linear(embedding_size, embedding_size),
        )

        self.downsample = Resampler(ndim=3, scale_factor=0.5)

        self.blocks = nn.ModuleList()
        for b in reversed(range(num_blocks)):
            prev_chan, next_chan = chan(b + 1), chan(b)
            self.blocks.append(
                ModulatedResidualBlock(
                    in_chan=prev_chan,
                    out_chan=next_chan,
                    embedding_size=embedding_size,
                    use_normalize=use_normalize,
                )
            )

        self.conv1 = ModulatedConv3d(
            in_chan=chan(0),
            out_chan=chan(-1),
            embedding_size=embedding_size,
            kernel_size=1,
        )

        self.act1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.out = ModulatedConv3d(
            in_chan=chan(-1),
            out_chan=out_chan,
            embedding_size=embedding_size,
            kernel_size=1,
        )

    def forward(self, x, style):
        s = self.style_embed(style)

        x = self.head(x, s)
        x = self.head_act(x)

        for block in self.blocks:
            x = block(x, s)
            x = self.downsample(x)

        x = self.conv1(x, s)
        x = self.act1(x)
        x = self.out(x, s)

        return x


# class CrossChanelMixing(nn.Module):
#     def __init__(
#         self,
#         in_chan,
#         out_chan,
#     ):
#         super().__init__()
#         self.in_chan = in_chan
#         self.out_chan = out_chan
#         self.mix = nn.Conv3d(in_channels=in_chan, out_channels=out_chan, kernel_size=1, stride=1, padding=0,bias=False)
#         for param in self.mix.parameters():
#             param.requires_grad = False
#         nn.init.kaiming_normal_(self.mix.weight, a=0.2)
#
#     def forward(self, x, style=None):
#         # batch_size, channels, height, width, depth = x.shape
#         x = self.mix(x)
#         return x
#
# class Projector(nn.Module):
#     def __init__(
#             self,
#             in_chan,
#             out_chan,
#     )
