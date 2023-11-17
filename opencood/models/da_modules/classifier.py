import torch
import torch.nn.functional as F
from torch import nn

from opencood.models.da_modules.gradient_layer import GradientScalarLayer


# class DAFeatureHead(nn.Module):
#     """
#     Adds a simple Feature-level Domain Classifier head
#     """
#
#     def __init__(self, in_channels, grad_scale=-1):
#         """
#         Arguments:
#             in_channels (int): number of channels of the input feature
#             grad_scale (float): the scale of the gradient reversal layer
#         """
#         super(DAFeatureHead, self).__init__()
#
#         self.reverse_grad_layer = GradientScalarLayer(grad_scale)
#
#         self.conv1 = nn.Conv2d(in_channels, 512, kernel_size=1, stride=1)
#         self.conv2 = nn.Conv2d(512, 1, kernel_size=1, stride=1)
#
#         for l in [self.conv1, self.conv2]:
#             torch.nn.init.normal_(l.weight, std=0.001)
#             torch.nn.init.constant_(l.bias, 0)
#
#     def forward(self, x):
#         x = self.reverse_grad_layer(x)
#
#         x = F.relu(self.conv1(x))
#         x = self.conv2(x)
#         return x


# class DAInstanceHead(nn.Module):
#     """
#     Adds a simple Instance-level Domain Classifier head
#     """
#     def __init__(self, in_channels, grad_scale=-1):
#         """
#         Arguments:
#             in_channels (int): number of channels of the input feature
#             grad_scale (float): the scale of the gradient reversal layer
#         """
#         super(DAInstanceHead, self).__init__()
#
#         self.reverse_grad_layer = GradientScalarLayer(grad_scale)
#
#         self.dense1 = nn.Linear(in_channels, 1024)
#         self.dense2 = nn.Linear(1024, 1024)
#         self.dense3 = nn.Linear(1024, 1)
#
#         for l in [self.dense1, self.dense2]:
#             nn.init.normal_(l.weight, std=0.01)
#             nn.init.constant_(l.bias, 0)
#         nn.init.normal_(self.dense3.weight, std=0.05)
#         nn.init.constant_(self.dense3.bias, 0)
#
#     def forward(self, x):
#         x = self.reverse_grad_layer(x)
#
#         x = F.relu(self.dense1(x))
#         x = F.dropout(x, p=0.5, training=self.training)
#         x = F.relu(self.dense2(x))
#         x = F.dropout(x, p=0.5, training=self.training)
#
#         x = self.dense3(x)
#         return x


class DAImgHead(nn.Module):
    """
    A simple Image-level Domain Classifier head
    """

    def __init__(self, in_channels, grad_scale=-1):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(DAImgHead, self).__init__()

        self.reverse_grad_layer = GradientScalarLayer(grad_scale)
        self.conv1 = nn.Conv2d(in_channels, 512, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(512, 1, kernel_size=1, stride=1)

        for l in [self.conv1, self.conv2]:
            nn.init.normal_(l.weight, std=0.001)
            nn.init.constant_(l.bias, 0)

    def forward(self, x):
        # X : (b, c, h, w), b=cars_1 + cars_2 + ...
        x = self.reverse_grad_layer(x)

        x = F.relu(self.conv1(x))
        x = self.conv2(x)

        return x


class DAClsHead(nn.Module):
    """
    A domain classifier.
    """
    def __init__(self, in_channels, grad_scale=-1):
        super(DAClsHead, self).__init__()
        self.reverse_grad_layer = GradientScalarLayer(grad_scale)

        self.dense1 = nn.Linear(in_channels, 1024)
        self.dense2 = nn.Linear(1024, 1024)
        self.dense3 = nn.Linear(1024, 1)

        for l in [self.dense1, self.dense2]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)
        nn.init.normal_(self.dense3.weight, std=0.05)
        nn.init.constant_(self.dense3.bias, 0)

    def forward(self, x, weight=None):
        # X : (b, c, h, w), b=cars_1 + cars_2 + ...
        # weight : (h, w) or (b, 1, h, w) or None
        x = self.reverse_grad_layer(x)

        if weight is not None:
            if len(weight.shape) == 2:
                weight = weight.unsqueeze(0).unsqueeze(0)
            x = x * weight.sigmoid()

        x = torch.flatten(x, start_dim=2)  # (b, c, h*w)
        x = x.mean(dim=2)  # (b, c)

        x = F.relu(self.dense1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.dense2(x))
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.dense3(x)
        return x
