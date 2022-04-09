from collections import OrderedDict

import torch
import torch.nn as nn


def shape_to_num_params(shapes):
    return torch.sum(torch.tensor([torch.prod(s) for s in shapes])).int().item()


class WeightRegressor(nn.Module):
    """Regressing features to convolution weight kernel"""

    def __init__(self, input_dim, hidden_dim, kernel_size=3, out_channels=16, in_channels=16):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.in_channels = in_channels

        # Feature Transformer
        self.fusion = nn.Sequential(
            nn.Conv2d(2 * self.input_dim, self.input_dim, kernel_size=1, padding=0, stride=1, bias=True),
            nn.InstanceNorm2d(self.input_dim),
            nn.ReLU(),
        )
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(64, self.hidden_dim, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(),
        )

        # Linear Mapper
        self.w1 = nn.Parameter(torch.randn((self.hidden_dim, self.in_channels * self.hidden_dim)))
        self.b1 = nn.Parameter(torch.randn((self.in_channels * self.hidden_dim)))
        self.w2 = nn.Parameter(torch.randn((self.hidden_dim, self.out_channels * self.kernel_size * self.kernel_size)))
        self.b2 = nn.Parameter(torch.randn((self.out_channels * self.kernel_size * self.kernel_size)))

        self.weight_init()

    def weight_init(self):
        nn.init.kaiming_normal_(self.w1)
        nn.init.zeros_(self.b1)
        nn.init.kaiming_normal_(self.w2)
        nn.init.zeros_(self.b2)

    def forward(self, w_image_codes, w_bar_codes):
        bs = w_image_codes.size(0)

        # Feature Transformation
        out = self.fusion(torch.cat((w_image_codes, w_bar_codes), 1))
        out = self.feature_extractor(out)
        out = out.view(bs, -1)

        # Linear map to weights
        out = torch.matmul(out, self.w1) + self.b1
        out = out.view(bs, self.in_channels, self.hidden_dim)
        out = torch.matmul(out, self.w2) + self.b2
        kernel = out.view(bs, self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)

        return kernel


class Hypernetwork(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=64, target_shape=None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.target_shape = target_shape

        num_predicted_weights = 0
        weight_regressors = OrderedDict()
        for layer_name in target_shape:
            new_layer_name = "_".join(layer_name.split("."))
            shape = target_shape[layer_name]["shape"]

            # without consider bias
            if len(shape) == 4:
                out_channels, in_channels, kernel_size = shape[:3]
            else:
                out_channels, in_channels = shape
                kernel_size = 1

            num_predicted_weights += shape_to_num_params([torch.tensor(list(shape))])
            weight_regressors[new_layer_name] = WeightRegressor(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                kernel_size=kernel_size,
                out_channels=out_channels,
                in_channels=in_channels,
            )
        self.weight_regressors = nn.ModuleDict(weight_regressors)
        self.num_predicted_weights = num_predicted_weights

    def forward(self, w_image_codes, w_bar_codes):
        bs = w_image_codes.size(0)
        out_weights = {}

        for layer_name in self.weight_regressors:
            ori_layer_name = ".".join(layer_name.split("_"))
            w_idx = self.target_shape[ori_layer_name]["w_idx"]
            weights = self.weight_regressors[layer_name](
                w_image_codes[:, w_idx, :, :, :], w_bar_codes[:, w_idx, :, :, :]
            )
            out_weights[ori_layer_name] = weights.view(bs, *list(self.target_shape[ori_layer_name]["shape"]))

        return out_weights
