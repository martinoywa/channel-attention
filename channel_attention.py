import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = self.sigmoid(y)
        return x * y

# Define a simple CNN architecture with Channel Attention
class SimpleCNNWithChannelAttention(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNNWithChannelAttention, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.channel_attention = ChannelAttention(64)  # Apply Channel Attention after the first convolutional layer
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(128  *7*  7, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.channel_attention(x)
        x = nn.ReLU(inplace=True)(x)
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)
        x = self.conv2(x)
        x = nn.ReLU(inplace=True)(x)
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Example usage
num_classes = 10

# Create an instance of the SimpleCNNWithChannelAttention architecture
cnn = SimpleCNNWithChannelAttention(num_classes)

# Generate a random input tensor
input_tensor = torch.randn(1, 3, 32, 32)

# Forward pass through the network
output_tensor = cnn(input_tensor)

print(output_tensor.shape)