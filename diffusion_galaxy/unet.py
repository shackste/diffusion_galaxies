import torch
import torch.nn as nn
from torch.nn import functional as F


activation_functions = {
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    None: Identity,
}

class UNet(nn.Module):
    def __init__(self, c_in=1, c_out=1, time_dim=256, device='cuda', image_size=64, attention=True, activation=None):
        """
        It takes in input channel input and channel output, which by defualt are 3 because
        we work with RGB images, but you can use 1 for BW

        time_dim = dimension of timestep embedding

        """
        super().__init__()

        # Encoder
        self.device = device
        self.image_size = image_size
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, self.image_size)  # Wrap-up for 2 Conv Layers
        self.down1 = Down(self.image_size, self.image_size * 2)  # input and output channels
        self.sa1 = SelfAttention(self.image_size * 2,
                                 int(self.image_size / 2))  # 1st is channel dim, 2nd current image resolution
        self.down2 = Down(self.image_size * 2, self.image_size * 4)
        self.sa2 = SelfAttention(self.image_size * 4, int(self.image_size / 4))
        self.down3 = Down(self.image_size * 4, self.image_size * 4)
        self.sa3 = SelfAttention(self.image_size * 4, int(self.image_size / 8))

        # Bootleneck
        self.bot1 = DoubleConv(self.image_size * 4, self.image_size * 8)
        self.bot2 = DoubleConv(self.image_size * 8, self.image_size * 8)
        self.bot3 = DoubleConv(self.image_size * 8, self.image_size * 4)

        # Decoder: reverse of encoder
        self.up1 = Up(self.image_size * 8, self.image_size * 2)
        self.sa4 = SelfAttention(self.image_size * 2, int(self.image_size / 4))
        self.up2 = Up(self.image_size * 4, self.image_size)
        self.sa5 = SelfAttention(self.image_size, int(self.image_size / 2))
        self.up3 = Up(self.image_size * 2, self.image_size)
        self.sa6 = SelfAttention(self.image_size, self.image_size)
        self.outc = nn.Conv2d(self.image_size, c_out, kernel_size=1)  # projecting back to the output channel dimensions

        # Attention
        if not attention:
            self.sa1 = nn.Identity()
            self.sa2 = nn.Identity()
            self.sa3 = nn.Identity()
            self.sa4 = nn.Identity()
            self.sa5 = nn.Identity()
            self.sa6 = nn.Identity()

        # Activation
        self.activation = activation_functions[activation]()


    def pos_encoding(self, t, channels):
        """z
        Input noised images and the timesteps. The timesteps will only be
        a tensor with the integer timesteps values in it
        """
        inv_freq = 1.0 / (
                10000
                ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b],
                            dim=-1)  # Concatenates the given sequence of seq tensors in the given dimension.
        return pos_enc
        # Instead of giving the timesteps to the model in their plain form we'll make it
        # easier for it and encode them with the sinusoida embedding

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t,
                              self.time_dim)  # Encoding timesteps is HERE, we provide the dimension we want to encode

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)  # We note that upsampling box that in the skip connections from encoder
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        output = self.activation(output)
        return output


class IdentityLayer(nn.Module):
    def __init__(self):
        super(IdentityLayer, self).__init__()

    def forward(self, x):
        return x


class SelfAttention(nn.Module):
    """
    Pre Layer norm  -> multi-headed tension -> skip connections -> pass it to
    the feed forward layer (layer-norm -> 2 multiheadattention)
    """

    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)


class DoubleConv(nn.Module):
    """
    Normal convolution block, with 2d convolution -> Group Norm -> GeLU -> convolution -> Group Norm
    Possibility to add residual connection providing residual=True
    """

    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    """
    maxpool reduce size by half -> 2*DoubleConv -> Embedding layer

    """

    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(  # linear projection to bring the time embedding to the proper dimension
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])  # projection
        return x + emb


class Up(nn.Module):
    """
    We take the skip connection which comes from the encoder
    """

    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UNet_conditional(nn.Module):
    def __init__(self, c_in=1, c_out=1, time_dim=256, num_classes=None, device='cuda', image_size=64):
        """
        It takes in input channel input and channel output, which by default are 3 because
        we work with RGB images, but you can use 1 for BW

        time_dim = dimension of timestep embedding

        """
        super().__init__()

        # Encoder
        self.device = device
        self.image_size = image_size
        self.num_classes = num_classes
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, self.image_size)  # Wrap-up for 2 Conv Layers
        self.down1 = Down(self.image_size, self.image_size * 2)  # input and output channels
        self.sa1 = SelfAttention(self.image_size * 2,
                                 int(self.image_size / 2))  # 1st is channel dim, 2nd current image resolution
        self.down2 = Down(self.image_size * 2, self.image_size * 4)
        self.sa2 = SelfAttention(self.image_size * 4, int(self.image_size / 4))
        self.down3 = Down(self.image_size * 4, self.image_size * 4)
        self.sa3 = SelfAttention(self.image_size * 4, int(self.image_size / 8))

        # Bootleneck
        self.bot1 = DoubleConv(self.image_size * 4, self.image_size * 8)
        self.bot2 = DoubleConv(self.image_size * 8, self.image_size * 8)
        self.bot3 = DoubleConv(self.image_size * 8, self.image_size * 4)

        # Decoder: reverse of encoder
        self.up1 = Up(self.image_size * 8, self.image_size * 2)
        self.sa4 = SelfAttention(self.image_size * 2, int(self.image_size / 4))
        self.up2 = Up(self.image_size * 4, self.image_size)
        self.sa5 = SelfAttention(self.image_size, int(self.image_size / 2))
        self.up3 = Up(self.image_size * 2, self.image_size)
        self.sa6 = SelfAttention(self.image_size, self.image_size)
        self.outc = nn.Conv2d(self.image_size, c_out, kernel_size=1)  # projecting back to the output channel dimensions

        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)

    def pos_encoding(self, t, channels):
        """
        Input noised images and the timesteps. The timesteps will only be
        a tensor with the integer timesteps values in it
        """
        inv_freq = 1.0 / (
                10000
                ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc
        # Instead of giving the timesteps to the model in their plain form we'll make it

    # easier for it and encode them with the sinusoida embedding
    def forward(self, x, t, y):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t,
                              self.time_dim)  # Encoding timesteps is HERE, we provide the dimension we want to encode

        if y is not None:
            t += self.label_emb(y)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)  # We note that upsampling box that in the skip connections from encoder 2
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output
