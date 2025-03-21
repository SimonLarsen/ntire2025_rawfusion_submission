from typing import Sequence
from abc import ABC
from functools import partial
import torch
from torch import nn
from einops import rearrange, repeat


def transform_flow(
    x: torch.Tensor,
    flow: torch.Tensor,
    mode: str = "bilinear",
) -> torch.Tensor:
    b, _, h, w = x.shape

    if flow.shape[-2:] != (h, w):
        flow = torch.nn.functional.interpolate(
            input=flow,
            size=(h, w),
            align_corners=False,
            mode="bilinear",
        )

    offsets = rearrange(-flow, "b xy h w -> b h w xy")

    grid = torch.stack(
        torch.meshgrid(
            torch.arange(0, w, device=x.device, dtype=x.dtype),
            torch.arange(0, h, device=x.device, dtype=x.dtype),
            indexing="xy",
        ),
        dim=-1,
    )
    grid = repeat(grid, "... -> b ...", b=b)
    grid = grid + offsets

    grid_scale = torch.tensor([(w - 1) / 2, (h - 1) / 2], device=x.device, dtype=x.dtype)
    grid_normalized = grid / grid_scale - 1

    return torch.nn.functional.grid_sample(
        input=x,
        grid=grid_normalized,
        mode=mode,
        align_corners=True,
        padding_mode="zeros",
    )


def local_l2_norm(x: torch.Tensor, size: int):
    h, w = x.shape[-2:]
    if h <= size and w <= size:
        return torch.norm(x, p=2, dim=(2, 3), keepdim=True)
    
    s = x ** 2
    s = s.cumsum_(dim=-1).cumsum_(dim=-2)
    s = torch.nn.functional.pad(s, (1, 0, 1, 0))

    k1, k2 = min(h, size), min(w, size)
    s1, s2, s3, s4 = (
        s[:,:,:-k1,:-k2],
        s[:,:,:-k1,k2:],
        s[:,:,k1:,:-k2],
        s[:,:,k1:,k2:],
    )
    out = s4 + s1 - s2 - s3
    out = torch.sqrt(out)

    oh, ow = out.shape[-2:]
    pad2d = ((w - ow) // 2, (w - ow + 1) // 2, (h - oh) // 2, (h - oh + 1) // 2)
    out = torch.nn.functional.pad(out, pad2d, mode="replicate")
    return out


class LPLayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        module_device = x.device
        downcast_x = _cast_if_autocast_enabled(x)
        downcast_weight = _cast_if_autocast_enabled(self.weight) if self.weight is not None else self.weight
        downcast_bias = _cast_if_autocast_enabled(self.bias) if self.bias is not None else self.bias
        with torch.autocast(enabled=False, device_type=module_device.type):
            return nn.functional.layer_norm(
                downcast_x,
                self.normalized_shape,
                downcast_weight,
                downcast_bias,
                self.eps,
            )


class LPLayerNorm2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.norm = LPLayerNorm(*args, **kwargs)
    
    def forward(self, x):
        x = rearrange(x, "b c h w -> b h w c")
        x = self.norm(x)
        x = rearrange(x, "b h w c -> b c h w")
        return x
    

class GRN(nn.Module):
    def __init__(
        self,
        channels: int,
        train_size: int,
        tlc: bool = True,
    ):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.train_size = train_size
        self.tlc = tlc

    def forward(self, x):
        if self.tlc:
            Gx = local_l2_norm(x, self.train_size)
        else:
            Gx = torch.norm(x, p=2, dim=(2,3), keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x
    

class ConvolutionBlock(nn.Module, ABC):
    def __init__(
        self,
        channels: int,
        train_size: int,
        *args,
        **kwargs,
    ):
        super().__init__()


class ConvNextBlock(ConvolutionBlock):
    def __init__(
        self,
        channels: int,
        train_size: int,
        kernel_size: int = 3,
        expand_ratio: float = 4.0,
        act_layer: nn.Module = nn.SiLU,
        norm_layer: nn.Module = LPLayerNorm2d,
        dropout: float = 0.0,
        second_norm: bool = False,
        grn: bool = True,
    ):
        super().__init__(channels, train_size)

        hidden_dim = round(channels * expand_ratio)

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                padding="same",
                groups=channels,
                bias=False,
            ),
            norm_layer(channels),
            nn.Conv2d(
                in_channels=channels,
                out_channels=hidden_dim,
                kernel_size=1,
                bias=not second_norm,
            ),
            norm_layer(hidden_dim) if second_norm else nn.Identity(),
            act_layer(),
            GRN(hidden_dim, train_size) if grn else nn.Identity(),
            nn.Conv2d(
                in_channels=hidden_dim,
                out_channels=channels,
                kernel_size=1,
            ),
            nn.Dropout(dropout),
        )

        nn.init.zeros_(self.block[-2].weight)
        nn.init.zeros_(self.block[-2].bias)
    
    def forward(self, x):
        return self.block(x) + x


class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        downsample: bool = False,
        block_layer: ConvolutionBlock = ConvNextBlock,
    ):
        super().__init__()

        self.out_channels = out_channels
        self.downsample = downsample
        if downsample:
            self.proj_in = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2,
                stride=2,
            )
        else:
            if in_channels != out_channels:
                self.proj_in = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
            else:
                self.proj_in = nn.Identity()
        
        blocks = []
        for _ in range(depth):
            blocks.append(block_layer(out_channels))
        self.blocks = nn.Sequential(*blocks)
    
    def forward(self, x):
        x = self.proj_in(x)
        x = self.blocks(x)
        return x


class Upsample(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int = 2,
    ):
        super().__init__()

        self.proj = nn.Conv2d(in_channels, out_channels * patch_size ** 2, 1, 1, 0)
        self.shuffle = nn.PixelShuffle(patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = self.shuffle(x)
        return x
    

class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        depth: int,
        block_layer: ConvolutionBlock = ConvNextBlock,
        skip_method: str = "lerp",
    ):
        super().__init__()

        assert skip_method in ("lerp", "conv")

        self.upsample = Upsample(in_channels, skip_channels, 2)
        self.skip_channels = skip_channels
        self.skip_method = skip_method
        if skip_method == "lerp":
            self.skip_t = nn.Parameter(torch.tensor([0.0]))
        elif skip_method == "conv":
            self.skip_conv = nn.Conv2d(2 * skip_channels, skip_channels, 3, 1, 1)

        blocks = []
        for _ in range(depth):
            blocks.append(block_layer(skip_channels))
        self.blocks = nn.Sequential(*blocks)

        self.proj_out = None
        if out_channels != skip_channels:
            self.proj_out = nn.Conv2d(skip_channels, out_channels, 1, 1)
        
            nn.init.zeros_(self.proj_out.weight)
            nn.init.zeros_(self.proj_out.bias)
    
    def forward(self, x, x_skip):
        x = self.upsample(x)

        if self.skip_method == "lerp":
            skip_t = self.skip_t.sigmoid()
            x = skip_t * x + (1.0 - skip_t) * x_skip
        elif self.skip_method == "conv":
            x = self.skip_conv(torch.cat((x, x_skip), dim=1))

        x = self.blocks(x)

        if self.proj_out is not None:
            x = self.proj_out(x)

        return x
    

class UNet(nn.Module):
    def __init__(
        self,
        train_size: int,
        in_channels: int = 3,
        out_channels: int = 3,
        widths: Sequence[int] = (32, 64, 128, 256),
        depths: Sequence[int] = (2, 2, 3, 4),
        norm_layer: nn.Module = nn.BatchNorm2d,
        act_layer: nn.Module = nn.SiLU,
        block_layer: ConvolutionBlock = ConvNextBlock,
        skip_method: str = "conv",
    ):
        super().__init__()

        assert len(widths) == len(depths)
        self.widths = widths
        train_sizes = [train_size // 2**i for i in range(len(widths))]

        self.conv_in = nn.Conv2d(
            in_channels=in_channels,
            out_channels=widths[0],
            kernel_size=3,
            padding="same",
        )

        self.down = nn.ModuleList()
        prev_channels = widths[0]
        for i in range(len(widths)):
            self.down.append(
                DownBlock(
                    in_channels=prev_channels,
                    out_channels=widths[i],
                    depth=depths[i],
                    downsample=i > 0,
                    block_layer=partial(
                        block_layer,
                        train_size=train_sizes[i],
                    )
                )
            )
            prev_channels = widths[i]
        
        self.up = nn.ModuleList()
        for i in range(len(widths) - 1):
            self.up.append(
                UpBlock(
                    in_channels=widths[-1 - i],
                    skip_channels=widths[-2 - i],
                    out_channels=widths[-2 - i],
                    depth=depths[-2 - i],
                    block_layer=partial(
                        block_layer,
                        train_size=train_sizes[-2 - i],
                    ),
                    skip_method=skip_method,
                )
            )
        
        self.norm_out = norm_layer(widths[0])
        self.act_out = act_layer()
        self.conv_out = nn.Conv2d(
            in_channels=widths[0],
            out_channels=out_channels,
            kernel_size=3,
            padding="same",
        )
        
    def forward(self, x):
        x = self.conv_in(x)

        outputs = []
        for layer in self.down:
            x = layer(x)
            outputs.append(x)
        
        for i, layer in enumerate(self.up):
            x = layer(x, outputs[-2 - i])

        x = self.norm_out(x)
        x = self.act_out(x)
        x = self.conv_out(x)
        return x
    

def _cast_if_autocast_enabled(tensor):
    if torch.is_autocast_enabled():
        if tensor.device.type == 'cuda':
            dtype = torch.get_autocast_gpu_dtype()
        elif tensor.device.type == 'cpu':
            dtype = torch.get_autocast_cpu_dtype()
        else:
            raise NotImplementedError()
        return tensor.to(dtype=dtype)
    return tensor


class CNDB(nn.Module):
    def __init__(
        self,
        train_size: int,
        base_channels: int  = 64,
        grow_channels: int = 32,
        expand_ratio: float = 4.0,
        kernel_size: int = 3,
        depth: int = 4,
        norm_layer: nn.Module = LPLayerNorm2d,
        act_layer: nn.Module = nn.SiLU,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.base_channels = base_channels
        self.grow_channels = grow_channels

        self.blocks = nn.ModuleList()
        for i in range(depth):
            dim = base_channels + i * grow_channels
            self.blocks.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=dim,
                        out_channels=base_channels,
                        kernel_size=1,
                    ),
                    ConvNextBlock(
                        train_size=train_size,
                        channels=base_channels,
                        kernel_size=kernel_size,
                        expand_ratio=expand_ratio,
                        act_layer=act_layer,
                        norm_layer=norm_layer,
                        dropout=dropout,
                    ),
                    nn.Conv2d(
                        in_channels=base_channels,
                        out_channels=grow_channels,
                        kernel_size=1,
                    ),
                    act_layer(),
                )
            )
        
        self.conv_out = nn.Conv2d(base_channels + depth * grow_channels, base_channels, 3, 1, 1)
        self.dropout = nn.Dropout(dropout)

        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, x):
        outputs = [x]
        for block in self.blocks:
            h = block(torch.cat(outputs, dim=1))
            outputs.append(h)
        
        h = self.conv_out(torch.cat(outputs, dim=1))
        h = self.dropout(h)
        return 0.2 * h + x


class Stem(nn.Module):
    def __init__(
        self,
        train_size: int,
        in_channels: int = 3,
        width: int = 32,
        depth: int = 5,
        kernel_size: int = 3,
        expand_ratio: float = 3.0,
        norm_layer: nn.Module = LPLayerNorm2d,
        act_layer: nn.Module = nn.SiLU,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.patch = nn.Conv2d(
            in_channels=in_channels,
            out_channels=width,
            kernel_size=2,
            stride=2,
        )

        self.blocks = nn.Sequential()
        for _ in range(depth):
            self.blocks.append(
                ConvNextBlock(
                    train_size=train_size // 2,
                    channels=width,
                    kernel_size=kernel_size,
                    expand_ratio=expand_ratio,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    dropout=dropout,
                )
            )
    
    def forward(self, x):
        x = self.patch(x)
        x = self.blocks(x)
        return x
    
class RegistrationModel(nn.Module):
    def __init__(
        self,
        train_size: int,
        in_channels: int = 3,
        widths: Sequence[int] = (32, 64, 128, 256, 512),
        depths: Sequence[int] = (2, 3, 4, 5, 6),
        kernel_size: int = 5,
        expand_ratio: float = 2.5,
        norm_layer: nn.Module = LPLayerNorm2d,
        act_layer: nn.Module = nn.SiLU,
        dropout: float = 0.0,
    ):
        super().__init__()

        block_layer = partial(
            ConvNextBlock,
            kernel_size=kernel_size,
            expand_ratio=expand_ratio,
            norm_layer=norm_layer,
            act_layer=act_layer,
            dropout=dropout,
        )

        self.unet = UNet(
            train_size=train_size,
            in_channels=2 * in_channels,
            out_channels=2,
            widths=widths,
            depths=depths,
            norm_layer=norm_layer,
            act_layer=act_layer,
            block_layer=block_layer,
            skip_method="conv",
        )
        
        nn.init.zeros_(self.unet.conv_out.weight)
        nn.init.zeros_(self.unet.conv_out.bias)
    
    def forward(self, x, return_mask: bool = True):
        assert len(x.shape) == 5  # b n c h w
        b, n = x.shape[:2]

        x_ref = repeat(x[:, :1], "b 1 c ... -> b n c ...", n=n)
        x_flow = torch.cat((x_ref, x), dim=2)

        x_flow = rearrange(x_flow, "b n ... -> (b n) ...")
        flow = self.unet(x_flow)

        x = rearrange(x, "b n ... -> (b n) ...")
        x = transform_flow(x, flow)
        x = rearrange(x, "(b n) ... -> b n ...", b=b)

        mask = None
        if return_mask:
            mask = torch.ones_like(x[:, :, :1])
            mask = rearrange(mask, "b n ... -> (b n) ...")
            mask = transform_flow(mask, flow)
            mask = rearrange(mask, "(b n) ... -> b n ...", b=b)
        
        flow = rearrange(flow, "(b n) ... -> b n ...", b=b)
        return x, flow, mask
    

class MergeModel(nn.Module):
    def __init__(
        self,
        train_size: int,
        num_brackets: int = 9,
        in_channels: int = 3,
        base_channels: int = 48,
        grow_channels: int = 24,
        expand_ratio: float = 2.5,
        kernel_size: int = 5,
        feature_channels: int = 32,
        depth: int = 24,
        norm_layer: nn.Module = LPLayerNorm2d,
        act_layer: nn.Module = nn.SiLU,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(
            in_channels=num_brackets * in_channels,
            out_channels=base_channels,
            kernel_size=1,
            stride=1,
        )

        self.blocks = nn.Sequential()
        for _ in range(depth):
            self.blocks.append(
                CNDB(
                    train_size=train_size,
                    base_channels=base_channels,
                    grow_channels=grow_channels,
                    expand_ratio=expand_ratio,
                    kernel_size=kernel_size,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    dropout=dropout,
                )
            )
        
        self.unpatch = nn.Sequential(
            nn.Conv2d(base_channels, feature_channels, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(feature_channels, 4 * feature_channels, 3, 1, 1),
            nn.PixelShuffle(2),
        )

        self.conv_out = nn.Conv2d(feature_channels, 3, 3, 1, 1)

        nn.init.zeros_(self.conv_out.weight)
        nn.init.constant_(self.conv_out.bias, 0.5)
    
    def forward(self, x):
        x = self.conv_in(x)
        x = self.blocks(x)
        x = self.unpatch(x)
        x = self.conv_out(x)
        return x

class FlowCNRDBModel(nn.Module):
    def __init__(
        self,
        train_size: int = 256,
        num_brackets: int = 9,
        stem_width: int = 32,
        stem_depth: int = 5,
        stem_kernel_size: int = 5,
        stem_expand_ratio: float = 2.5,
        stem_dropout: float = 0.0,
        reg_widths: Sequence[int] = (32, 64, 128, 256, 512),
        reg_depths: Sequence[int] = (2, 3, 4, 5, 6),
        reg_kernel_size: int = 7,
        reg_expand_ratio: float = 2.5,
        reg_dropout: float = 0.0,
        merge_base_channels: int = 64,
        merge_grow_channels: int = 32,
        merge_expand_ratio: float = 2.0,
        merge_kernel_size: int = 7,
        merge_features: int = 32,
        merge_depth: int = 14,
        merge_dropout: float = 0.0,
        append_mask: bool = True,
    ):
        super().__init__()

        self.append_mask = append_mask
        self.stem_width = stem_width

        self.stem = Stem(
            train_size=train_size,
            in_channels=1,
            width=stem_width,
            depth=stem_depth,
            kernel_size=stem_kernel_size,
            expand_ratio=stem_expand_ratio,
            dropout=stem_dropout,
        )

        self.reg_model = RegistrationModel(
            train_size=train_size // 2,
            in_channels=stem_width,
            widths=reg_widths,
            depths=reg_depths,
            kernel_size=reg_kernel_size,
            expand_ratio=reg_expand_ratio,
            dropout=reg_dropout,
        )

        merge_inputs = stem_width + append_mask
        self.merge_model = MergeModel(
            train_size=train_size // 2,
            num_brackets=num_brackets,
            in_channels=merge_inputs,
            base_channels=merge_base_channels,
            grow_channels=merge_grow_channels,
            expand_ratio=merge_expand_ratio,
            kernel_size=merge_kernel_size,
            feature_channels=merge_features,
            depth=merge_depth,
            dropout=merge_dropout,
        )

    def forward(self, x: torch.FloatTensor):
        assert len(x.shape) == 4  # b n h w
        b = x.size(0)

        # Extract features
        x = rearrange(x, "b n ... -> (b n) 1 ...")
        x = self.stem(x)
        x = rearrange(x, "(b n) ... -> b n ...", b=b)

        # Warp other brackets to reference
        x, _, mask = self.reg_model(x, return_mask=self.append_mask)

        # Add warp mask as a separate channel
        if self.append_mask:
            x = torch.cat((x, mask), dim=2)

        # Merge brackets
        x = rearrange(x, "b n c ... -> b (n c) ...")
        x = self.merge_model(x)

        return x
