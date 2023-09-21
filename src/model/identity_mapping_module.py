import torch
import torch.nn as nn
from model.chatglm_block import GLMBlock
from model.chatglm_block import ChatGLMConfig
import json
import os

# Define the path to the model configuration JSON file
model_config_path = os.path.join('model', 'chatglm_block', 'config.json')

# Open the configuration file and load it into a Python dictionary
with open(model_config_path) as config_file:
    config_dict = json.load(config_file)

# Create a configuration object using the dictionary values
configuration = ChatGLMConfig(**config_dict)

# Define a custom PyTorch module with a zero-initialized linear layer followed by batch normalization
class ZeroLinearBatchNorm(nn.Module):
    def __init__(self, in_features, out_features):
        super(ZeroLinearBatchNorm, self).__init__()

        # Define a linear layer with 'in_features' input features and 'out_features' output features
        # Initialize the weights of the linear layer to zero
        self.linear = nn.Linear(in_features, out_features)
        self.linear.weight.data.fill_(0) 

        # Define a batch normalization layer to normalize the outputs of the linear layer
        self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x):
        # Apply the linear layer to the input tensor 'x'
        out = self.linear(x)
        
        # Apply the batch normalization to the output of the linear layer
        # Note: The squeeze and unsqueeze operations are used to adjust the tensor dimensions as expected by the batch normalization layer
        out = self.bn(out.squeeze(1)) 
        return out.unsqueeze(1)

# Define a module that includes a sequence of zero-initialized linear layers, batch normalization, and a GLM block
class IdentityMappingModule(nn.Module):
    def __init__(self, in_features, hidden_features):
        super(IdentityMappingModule, self).__init__()

        # Define a sequence of layers: zero-initialized linear layer, GLM block, and another zero-initialized linear layer
        self.zerolinear1 = ZeroLinearBatchNorm(in_features, hidden_features)
        self.glmblock = GLMBlock(configuration.hidden_size, configuration.num_attention_heads, 
                                 configuration.layernorm_epsilon, configuration)
        self.zerolinear2 = ZeroLinearBatchNorm(hidden_features, in_features)

    def forward(self, x_dict):
        # Extract the 'hidden_states' tensor from the input dictionary
        x_input = x_dict["hidden_states"]
        
        # Print the shape of the input tensor for debugging purposes
        # print(f"[DEBUG]: x_input shape: {x_input.shape}")

        # Apply the first zero-initialized linear layer followed by batch normalization
        out = self.zerolinear1(x_input)  
        
        # Update the 'hidden_states' in the dictionary and apply the GLM block
        x_dict["hidden_states"] = out
        out = self.glmblock(**x_dict)[0]

        # Apply the second zero-initialized linear layer followed by batch normalization
        out = self.zerolinear2(out)
        
        # Print the output tensor added to the input tensor (skip connection) for debugging purposes
        # print(f"[DEBUG]: forward result: {out+x_input}")

        # Implement skip connection by adding the input tensor to the output tensor and return the result
        return out + x_input  
