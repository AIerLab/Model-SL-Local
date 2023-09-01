import torch
import torch.nn as nn
from chatglm_block.modeling_chatglm import GLMBlock
import json
import os
from chatglm_block.configuration_chatglm import ChatGLMConfig

model_config_path = os.path.join('model', 'chatglm_6b_split_server', 'config.json')

# Open and load the JSON file into a Python dict
with open(model_config_path) as config_file:
    config_dict = json.load(config_file)

configuration = ChatGLMConfig(**config_dict)

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

        self.zeroconv1 = ZeroConvBatchNorm(hidden_channels, hidden_channels)
        self.glmblock = GLMBlock(configuration)
        self.zeroconv2 = ZeroConvBatchNorm(hidden_channels, hidden_channels)

    def forward(self, x_dict):
        x_input = x_dict["hidden_states"]
        
        out = x_input.unsqueeze(-1).unsqueeze(-1)
        out = self.zerocov1(out)  # Apply the sequence of zero conv and batch norm layers
        x_dict["hidden_states"] = out
        out = self.glmblock(**x_dict)[0]
        out = self.zeroconv2(out)

        return out + x_input  # Implement skip connection by adding input tensor to output tensor