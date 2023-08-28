import torch
import torch.nn as nn

class ZeroConvBatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ZeroConvBatchNorm, self).__init__()

        # Define a zero convolutional layer with batch normalization
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1) # TODO finalise the choice of kernel_size, stride and padding
        self.conv.weight.data.fill_(0)  # Initialize weights with zeros
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv(x)  # Apply the zero convolution operation
        out = self.bn(out)   # Apply batch normalization
        return out

class IdentityMappingModule(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(IdentityMappingModule, self).__init__()

        layers = []
        for _ in range(6):  # Create 6 pairs of zero conv and batch norm layers # TODO Determine number of layers needed
            zero_conv_bn = ZeroConvBatchNorm(hidden_channels, hidden_channels)
            layers.append(zero_conv_bn)
        self.identity_module = nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(-1).unsqueeze(-1)
        out = self.identity_module(x)  # Apply the sequence of zero conv and batch norm layers
        return out + x  # Implement skip connection by adding input tensor to output tensor