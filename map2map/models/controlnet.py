import math
import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal

from .style import ModulatedConv3d
from .resample import Resampler
from .narrow import narrow_by
from .stylegan import G, HBlock, normalization, zero_module, narrow_as


class ConditioningNetwork(nn.Module):
    """
    Conditioning network that processes control inputs.
    Similar to ControlNet's conditioning branch.
    """
    def __init__(
        self,
        control_in_chan,
        embedding_size=16,
        scale_factor=8,
        chan_base=512,
        chan_min=64,
        chan_max=512,
        use_normalize=False,
    ):
        super().__init__()
        
        self.control_in_chan = control_in_chan
        self.embedding_size = embedding_size
        self.scale_factor = scale_factor
        num_blocks = round(math.log2(self.scale_factor))
        self.num_blocks = num_blocks
        self.use_normalize = use_normalize
        
        def chan(b):
            c = chan_base >> b
            c = max(c, chan_min)
            c = min(c, chan_max)
            return c
        
        # Initial convolution to process control input
        self.head = nn.Conv3d(
            control_in_chan,
            chan(0),
            kernel_size=3,
            padding=1
        )
        self.head_act = nn.SiLU()
        
        if self.use_normalize:
            self.head_norm = normalization(chan(0))
        
        # Downsampling blocks to match generator's feature hierarchy
        self.downsample = Resampler(ndim=3, scale_factor=0.5)
        
        self.blocks = nn.ModuleList()
        for b in range(num_blocks):
            in_chan, out_chan = chan(b), chan(b + 1)
            
            block = nn.Sequential(
                nn.Conv3d(in_chan, out_chan, kernel_size=3, padding=1),
                normalization(out_chan) if use_normalize else nn.Identity(),
                nn.SiLU(),
                nn.Conv3d(out_chan, out_chan, kernel_size=3, padding=1),
                normalization(out_chan) if use_normalize else nn.Identity(),
                nn.SiLU(),
            )
            self.blocks.append(block)
    
    def forward(self, control):
        """
        Process control input and return features at multiple scales.
        Returns list of features matching generator's hierarchy.
        """
        features = []
        
        x = self.head(control)
        if self.use_normalize:
            x = self.head_norm(x)
        x = self.head_act(x)
        
        # Store initial features
        features.append(x)
        
        # Process through downsampling blocks
        for block in self.blocks:
            x = self.downsample(x)
            x = block(x)
            features.append(x)
        
        return features


class ControlledHBlock(nn.Module):
    """
    Modified HBlock that accepts control features.
    Integrates control information via zero convolutions.
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
        
        # Original HBlock
        self.hblock = HBlock(
            prev_chan=prev_chan,
            next_chan=next_chan,
            out_chan=out_chan,
            embedding_size=embedding_size,
            inject_noise=inject_noise,
            use_normalize=use_normalize,
            use_attention=use_attention,
            use_pixel_shuffle=use_pixel_shuffle,
            which_attention=which_attention,
        )
        
        # Zero convolutions for control feature injection
        self.control_conv1 = zero_module(nn.Conv3d(next_chan, next_chan, kernel_size=1))
        self.control_conv2 = zero_module(nn.Conv3d(next_chan, next_chan, kernel_size=1))
        self.control_proj = zero_module(nn.Conv3d(next_chan, out_chan, kernel_size=1))
    
    def forward(self, x, y, s, control_features=None):
        # Store original positions for control injection
        x_orig, y_orig = x, y
        
        # Forward through original HBlock but intercept at key points
        if hasattr(self.hblock, 'use_pixel_shuffle') and self.hblock.use_pixel_shuffle:
            x = self.hblock.x_upconv(x)
            x = self.hblock.pixel_shuffle(x)
        else:
            x = self.hblock.upsample(x)
        
        # Block 1
        x = self.hblock.conv1(x, s)
        if self.hblock.inject_noise:
            x = self.hblock.noise1(x)
        if self.hblock.use_normalize:
            x = self.hblock.normalize1(x)
        x = self.hblock.act1(x)
        
        # Inject control features after first conv
        if control_features is not None:
            control_feat = narrow_as(control_features, x)
            x = x + self.control_conv1(control_feat)
        
        # Block 2
        x = self.hblock.conv2(x, s)
        if self.hblock.inject_noise:
            x = self.hblock.noise2(x)
        if self.hblock.use_normalize:
            x = self.hblock.normalize2(x)
        x = self.hblock.act2(x)
        if self.hblock.use_attention:
            x = self.hblock.attention(x)
        
        # Inject control features after second conv
        if control_features is not None:
            control_feat = narrow_as(control_features, x)
            x = x + self.control_conv2(control_feat)
        
        # Right branch
        y = self.hblock.upsample(y)
        y = narrow_as(y, x)
        
        # Project with control influence
        proj_out = self.hblock.proj(x, s)
        if control_features is not None:
            control_feat = narrow_as(control_features, proj_out)
            proj_out = proj_out + self.control_proj(control_feat)
        
        y = y + self.hblock.proj_act(proj_out)
        
        return x, y


class ControlNet(nn.Module):
    """
    ControlNet for StyleGAN-based super-resolution generator.
    Accepts a pretrained generator and adds controllable conditioning.
    """
    def __init__(
        self,
        generator,
        control_in_chan,
        control_scale_factor=None,
        use_normalize=False,
        freeze_generator=True,
    ):
        super().__init__()
        
        self.generator = generator
        self.control_in_chan = control_in_chan
        self.freeze_generator = freeze_generator
        
        # Extract generator parameters
        self.in_chan = generator.in_chan
        self.out_chan = generator.out_chan
        self.style_size = generator.style_size
        self.embedding_size = generator.embedding_size
        self.scale_factor = generator.scale_factor
        self.num_blocks = generator.num_blocks
        
        # Control scale factor (how much to downsample control input)
        if control_scale_factor is None:
            control_scale_factor = self.scale_factor
        self.control_scale_factor = control_scale_factor
        
        # Freeze generator if specified
        if freeze_generator:
            for param in self.generator.parameters():
                param.requires_grad = False
        
        # Conditioning network
        self.conditioning_net = ConditioningNetwork(
            control_in_chan=control_in_chan,
            embedding_size=self.embedding_size,
            scale_factor=self.control_scale_factor,
            use_normalize=use_normalize,
        )
        
        # Create controlled versions of generator blocks
        self.controlled_blocks = nn.ModuleList()
        for i, block in enumerate(self.generator.blocks):
            controlled_block = ControlledHBlock(
                prev_chan=block.conv1.in_chan,
                next_chan=block.conv1.out_chan,
                out_chan=block.proj.out_chan,
                embedding_size=self.embedding_size,
                inject_noise=hasattr(block, 'inject_noise') and block.inject_noise,
                use_normalize=hasattr(block, 'use_normalize') and block.use_normalize,
                use_attention=hasattr(block, 'use_attention') and block.use_attention,
            )
            self.controlled_blocks.append(controlled_block)
    
    def forward(self, x, style, control):
        """
        Forward pass with control conditioning.
        
        Args:
            x: Input low-resolution data
            style: Style vector for generator
            control: Control conditioning input
        """
        # Process control input
        control_features = self.conditioning_net(control)
        
        # Generator head processing
        s = self.generator.style_embed(style)
        y = x
        x = self.generator.head(x, s)
        x = self.generator.head_act(x)
        
        # Forward through controlled blocks
        for i, controlled_block in enumerate(self.controlled_blocks):
            # Get appropriate control features for this scale
            if i < len(control_features):
                ctrl_feat = control_features[i]
            else:
                ctrl_feat = None
            
            x, y = controlled_block(x, y, s, ctrl_feat)
        
        return y
    
    def forward_without_control(self, x, style):
        """
        Forward pass without control (equivalent to original generator).
        """
        return self.generator(x, style)
    
    def set_control_strength(self, strength):
        """
        Scale the control influence by modifying zero conv weights.
        """
        for block in self.controlled_blocks:
            for module in [block.control_conv1, block.control_conv2, block.control_proj]:
                for param in module.parameters():
                    param.data *= strength


class MultiScaleControlNet(ControlNet):
    """
    Extended ControlNet that accepts multiple control inputs at different scales.
    """
    def __init__(
        self,
        generator,
        control_channels_list,
        control_scale_factors=None,
        use_normalize=False,
        freeze_generator=True,
    ):
        # Initialize with first control input
        super().__init__(
            generator=generator,
            control_in_chan=sum(control_channels_list),
            use_normalize=use_normalize,
            freeze_generator=freeze_generator,
        )
        
        self.control_channels_list = control_channels_list
        self.num_controls = len(control_channels_list)
        
        if control_scale_factors is None:
            control_scale_factors = [self.scale_factor] * self.num_controls
        self.control_scale_factors = control_scale_factors
        
        # Replace single conditioning network with multiple
        del self.conditioning_net
        self.conditioning_nets = nn.ModuleList()
        
        for i, (control_chan, scale_factor) in enumerate(
            zip(control_channels_list, control_scale_factors)
        ):
            conditioning_net = ConditioningNetwork(
                control_in_chan=control_chan,
                embedding_size=self.embedding_size,
                scale_factor=scale_factor,
                use_normalize=use_normalize,
            )
            self.conditioning_nets.append(conditioning_net)
    
    def forward(self, x, style, controls):
        """
        Forward pass with multiple control inputs.
        
        Args:
            x: Input low-resolution data
            style: Style vector for generator
            controls: List of control conditioning inputs
        """
        # Process each control input
        all_control_features = []
        for control, conditioning_net in zip(controls, self.conditioning_nets):
            control_features = conditioning_net(control)
            all_control_features.append(control_features)
        
        # Combine control features by addition
        combined_features = []
        max_scales = max(len(cf) for cf in all_control_features)
        
        for scale_idx in range(max_scales):
            scale_features = []
            for cf in all_control_features:
                if scale_idx < len(cf):
                    scale_features.append(cf[scale_idx])
            
            if scale_features:
                # Sum features at this scale
                combined = scale_features[0]
                for feat in scale_features[1:]:
                    combined = combined + narrow_as(feat, combined)
                combined_features.append(combined)
        
        # Generator head processing
        s = self.generator.style_embed(style)
        y = x
        x = self.generator.head(x, s)
        x = self.generator.head_act(x)
        
        # Forward through controlled blocks
        for i, controlled_block in enumerate(self.controlled_blocks):
            if i < len(combined_features):
                ctrl_feat = combined_features[i]
            else:
                ctrl_feat = None
            
            x, y = controlled_block(x, y, s, ctrl_feat)
        
        return y