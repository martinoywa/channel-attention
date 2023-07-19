# Channel Attention / Squeeze and Excitation
Implementation of Channel Attention in PyTorch.

# Introduction
Attention is simply the assumption that a network should learn to boost some information that is helpful for a given example, while at the same time decreasing the importance of that which is not.

Common types for Computer Vision:

- Spatial Attention: Focuses on specific spatial regions or locations within an image. It can be used for tasks like object detection or image captioning.
- Channel Attention / Squeeze and Excitation: Operates at the channel level. It allows the network to dynamically adjust the importance of different channels, emphasizing informative channels and suppressing less relevant ones.
- Self-Attention / intra-attention / non-local attention: It captures dependencies between different spatial locations within an image, i.e, who’s most likely to be related. It allows the network to attend to the spatial structure of the image and model long-range dependencies. Self-attention has been successfully applied in tasks like image classification, object detection, and image generation. Use in transformer-based architectures such as  the Vision Transformer (ViT).


# Code Notes
In this code, we define a ChannelAttention module that takes the number of input channels and a reduction ratio as parameters. The reduction ratio determines the number of hidden units in the fully connected layers of the channel attention block. The purpose of the reduction ratio is to reduce the dimensionality of the input channels before expanding them back to the original number of channels. This reduction can help in reducing the computational complexity of the channel attention mechanism. In some cases, you may choose not to use a reduction ratio and keep the number of hidden units the same as the number of input channels when you want to preserve the dimensionality of the channels and allow the channel attention mechanism to have a more direct impact on the input channels.

The forward method applies global average pooling (Squeeze) to the input tensor, followed by a fully connected network (Excitation) with ReLU activation. Finally, the sigmoid function is applied to the output to obtain attention weights which get multiplied by our channels, and depending on the score, a channel may either be boosted or diminished.

Next, we define a SimpleCNNWithChannelAttention class that extends a simple CNN architecture. The Channel Attention module is applied after the first convolutional layer. You can choose the specific layer(s) in your CNN architecture where you want to insert the Channel Attention module based on your requirements.

During the forward pass, the input tensor is passed through the layers of the CNN architecture, and the Channel Attention module is applied after the first convolutional layer. The output tensor is then passed through the remaining layers of the CNN architecture, and the classification layer is applied.

This way, the Channel Attention module becomes a part of the overall CNN architecture, enhancing the network's ability to focus on important channels and improving its performance.
