import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

# ----------------------------------------------------------------------------


def _init_conv_layer(conv, activation, mode="fan_out"):
    if isinstance(activation, nn.LeakyReLU):
        torch.nn.init.kaiming_uniform_(
            conv.weight,
            a=activation.negative_slope,
            nonlinearity="leaky_relu",
            mode=mode,
        )
    elif isinstance(activation, (nn.ReLU, nn.ELU)):
        torch.nn.init.kaiming_uniform_(conv.weight, nonlinearity="relu", mode=mode)
    else:
        pass
    if conv.bias != None:
        torch.nn.init.zeros_(conv.bias)


def output_to_image(out):
    out = (out[0].cpu().permute(1, 2, 0) + 1.0) * 127.5
    out = out.to(torch.uint8).numpy()
    return out


# ----------------------------------------------------------------------------

#################################
########### GENERATOR ###########
#################################


class GConv(nn.Module):
    """Implements the gated 2D convolution introduced in
    `Free-Form Image Inpainting with Gated Convolution`(Yu et al., 2019)
    """

    def __init__(
        self,
        cnum_in,
        cnum_out,
        ksize,
        stride=1,
        padding="auto",
        rate=1,
        activation=nn.ELU(),
        bias=True,
        gated=True,
    ):
        super().__init__()

        padding = rate * (ksize - 1) // 2 if padding == "auto" else padding
        self.activation = activation
        self.cnum_out = cnum_out
        num_conv_out = 2 * cnum_out if gated else cnum_out
        self.conv = nn.Conv2d(
            cnum_in,
            num_conv_out,
            kernel_size=ksize,
            stride=stride,
            padding=padding,
            dilation=rate,
            bias=bias,
        )

        _init_conv_layer(self.conv, activation=self.activation)

        self.ksize = ksize
        self.stride = stride
        self.rate = rate
        self.padding = padding
        self.gated = gated

    def forward(self, x):
        """
        Args:

        """
        if not self.gated:
            return self.conv(x)

        x = self.conv(x)
        x, y = torch.split(x, self.cnum_out, dim=1)
        x = self.activation(x)
        y = torch.sigmoid(y)
        x = x * y

        return x


# ----------------------------------------------------------------------------


class GDeConv(nn.Module):
    """Upsampling (x2) followed by convolution"""

    def __init__(self, cnum_in, cnum_out, padding=1):
        super().__init__()

        self.conv = GConv(cnum_in, cnum_out, ksize=3, stride=1, padding=padding)

    def forward(self, x):
        x = F.interpolate(
            x, scale_factor=2, mode="nearest", recompute_scale_factor=False
        )
        x = self.conv(x)
        return x


# ----------------------------------------------------------------------------


class GDownsamplingBlock(nn.Module):
    """Strided convolution (s=2) followed by convolution (s=1)"""

    def __init__(self, cnum_in, cnum_out, cnum_hidden=None):
        super().__init__()

        cnum_hidden = cnum_out if cnum_hidden == None else cnum_hidden
        self.conv1_downsample = GConv(cnum_in, cnum_hidden, ksize=3, stride=2)
        self.conv2 = GConv(cnum_hidden, cnum_out, ksize=3, stride=1)

    def forward(self, x):
        x = self.conv1_downsample(x)
        x = self.conv2(x)
        return x


# ----------------------------------------------------------------------------


class GUpsamplingBlock(nn.Module):
    """Upsampling (x2) followed by two convolutions"""

    def __init__(self, cnum_in, cnum_out, cnum_hidden=None):
        super().__init__()
        cnum_hidden = cnum_out if cnum_hidden == None else cnum_hidden
        self.conv1_upsample = GDeConv(cnum_in, cnum_hidden)
        self.conv2 = GConv(cnum_hidden, cnum_out, ksize=3, stride=1)

    def forward(self, x):
        x = self.conv1_upsample(x)
        x = self.conv2(x)
        return x


# ----------------------------------------------------------------------------


class CoarseGenerator(nn.Module):
    """Coarse Network (Stage I)"""

    def __init__(self, cnum_in, cnum_out, cnum):
        super().__init__()

        self.conv1 = GConv(cnum_in, cnum // 2, ksize=5, stride=1, padding=2)

        # downsampling
        self.down_block1 = GDownsamplingBlock(cnum // 2, cnum)
        self.down_block2 = GDownsamplingBlock(cnum, 2 * cnum)

        # bottleneck
        self.conv_bn1 = GConv(2 * cnum, 2 * cnum, ksize=3, stride=1)
        self.conv_bn2 = GConv(2 * cnum, 2 * cnum, ksize=3, rate=2, padding=2)
        self.conv_bn3 = GConv(2 * cnum, 2 * cnum, ksize=3, rate=4, padding=4)
        self.conv_bn4 = GConv(2 * cnum, 2 * cnum, ksize=3, rate=8, padding=8)
        self.conv_bn5 = GConv(2 * cnum, 2 * cnum, ksize=3, rate=16, padding=16)
        self.conv_bn6 = GConv(2 * cnum, 2 * cnum, ksize=3, stride=1)
        self.conv_bn7 = GConv(2 * cnum, 2 * cnum, ksize=3, stride=1)

        # upsampling
        self.up_block1 = GUpsamplingBlock(2 * cnum, cnum)
        self.up_block2 = GUpsamplingBlock(cnum, cnum // 4, cnum_hidden=cnum // 2)

        # to RGB
        self.conv_to_rgb = GConv(
            cnum // 4, cnum_out, ksize=3, stride=1, activation=None, gated=False
        )
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv1(x)

        # downsampling
        x = self.down_block1(x)
        x = self.down_block2(x)

        # bottleneck
        x = self.conv_bn1(x)
        x = self.conv_bn2(x)
        x = self.conv_bn3(x)
        x = self.conv_bn4(x)
        x = self.conv_bn5(x)
        x = self.conv_bn6(x)
        x = self.conv_bn7(x)

        # upsampling
        x = self.up_block1(x)
        x = self.up_block2(x)

        # to RGB
        x = self.conv_to_rgb(x)
        x = self.tanh(x)
        return x


# ----------------------------------------------------------------------------


class CoarseGANInpainter(nn.Module):
    """Inpainting network consisting of a coarse and a refinement network.
    Described in the paper
    `Free-Form Image Inpainting with Gated Convolution, Yu et. al`.
    """

    def __init__(self, cnum_in=5, cnum_out=3, cnum=48, checkpoint=None, device="cpu"):
        super().__init__()

        self.stage1 = CoarseGenerator(cnum_in, cnum_out, cnum).to(device)
        self.cnum_in = cnum_in
        self.device = device

        if checkpoint is not None:
            generator_state_dict = torch.load(checkpoint, map_location=device)
            self.load_state_dict(generator_state_dict, strict=True)

        self.eval()

    def forward(self, x, mask):
        """
        Args:
            x (Tensor): input of shape [batch, cnum_in, H, W]
            mask (Tensor): mask of shape [batch, 1, H, W]
        """
        # get coarse result
        x_stage1 = self.stage1(x)
        # inpaint input with coarse result
        # x = x_stage1*mask + x[:, :self.cnum_in-2]*(1.-mask)

        return x_stage1

    @torch.inference_mode()
    def predict(self, image, mask):
        """
        Args:
            image (Tensor): input image of shape [cnum_out, H, W]
            mask (Tensor): mask of shape [*, H, W]
            return_vals (str | List[str]): options: inpainted, stage1, stage2, flow
        """

        _, h, w = image.shape
        grid = 8

        next_h = math.ceil(h / grid) * grid
        next_w = math.ceil(w / grid) * grid

        delta_h = next_h - h
        delta_w = next_w - w

        # Change from crop to pad, then crop to original size after
        image = TF.pad(image, (0, 0, delta_w, delta_h))
        mask = TF.pad(mask, (0, 0, delta_w, delta_h))
        image = image[None, : self.cnum_in, :, :]
        mask = mask[None, :3, :, :].sum(1, keepdim=True)

        image = image * 2 - 1.0  # map image values to [-1, 1] range
        # 1.: masked 0.: unmasked
        mask = (mask > 0.0).to(dtype=torch.float32)

        image_masked = image * (1.0 - mask)  # mask image

        ones_x = torch.ones_like(image_masked)[:, :1]  # sketch channel
        x = torch.cat(
            [image_masked, ones_x, ones_x * mask], dim=1
        )  # concatenate channels

        x_stage1 = self.forward(x, mask)

        image_compl = image * (1.0 - mask) + x_stage1 * mask

        return output_to_image(image_compl)
