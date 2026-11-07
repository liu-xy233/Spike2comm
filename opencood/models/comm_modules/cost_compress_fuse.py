import numpy as np
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from opencood.models.cost_modules.deformable_detr import DeformableDETR


class CostCompressFuse(nn.Module):
    """CoST dynamic feature selection and optional reconstruction."""

    def __init__(self, args, max_cav=5):
        super().__init__()
        self.ego_id = 0
        self.smooth = bool(args.get("smooth", False))
        if self.smooth:
            kernel_size = args["gaussian_smooth"].get("k_size", 5)
            c_sigma = args["gaussian_smooth"].get("c_sigma", 1)
            self.gaussian_filter = nn.Conv2d(
                1, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2
            )
            self.init_gaussian_filter(kernel_size, c_sigma)
            self.gaussian_filter.requires_grad_(False)

        self.rescale_rate = args.get("rescale_rate", args.get("resacle_rate", 10))
        self.thresh = args.get("thresh", args.get("thre", 0.01))
        self.reconstruct = args.get("reconstruct", False)
        self.use_decoder = "use_decoder" in args

        if self.reconstruct:
            input_dim = args["mlp_dim"]
            if self.reconstruct == "deformable":
                self.renew_net = DeformableDETR(args["deformable_transformer"])
            elif self.reconstruct == "conv":
                self.conv_fuse = nn.Sequential(
                    Rearrange("b l c h w -> (b l) c h w", l=max_cav),
                    nn.Conv2d(2 * input_dim, input_dim, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(input_dim),
                    nn.GELU(),
                    Rearrange("(b l) c h w -> b l c h w", l=max_cav),
                )
            else:
                raise NotImplementedError("Unsupported CoST reconstruction mode: {}".format(self.reconstruct))

            hidden_dim = input_dim // 4
            if self.use_decoder:
                self.decoder = nn.Sequential(
                    Rearrange("b l c h w -> (b l) c h w", l=max_cav),
                    nn.Conv2d(input_dim, hidden_dim, kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU(),
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU(),
                    nn.Conv2d(hidden_dim, input_dim, kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(input_dim),
                    nn.ReLU(),
                    Rearrange("(b l) c h w -> b l c h w", l=max_cav),
                )
            else:
                self.decoder = nn.Identity()

    def init_gaussian_filter(self, k_size=5, sigma=1):
        center = k_size // 2
        x, y = np.mgrid[0 - center : k_size - center, 0 - center : k_size - center]
        g = 1 / (2 * np.pi * sigma) * np.exp(-(np.square(x) + np.square(y)) / (2 * np.square(sigma)))
        self.gaussian_filter.weight.data = torch.Tensor(g).unsqueeze(0).unsqueeze(0)
        self.gaussian_filter.bias.data.zero_()

    def forward(self, feat, confidence_map, prev_feat=None, pre_map=None, reset=False):
        batch_size, cav_num, _, height, width = confidence_map.shape

        if self.smooth:
            confidence_map = confidence_map.reshape(batch_size * cav_num, 1, height, width)
            communication_maps = self.gaussian_filter(confidence_map).reshape(batch_size, cav_num, 1, height, width)
            if pre_map is not None:
                pre_map = self.gaussian_filter(pre_map.reshape(batch_size * cav_num, 1, height, width))
                pre_map = pre_map.reshape(batch_size, cav_num, 1, height, width)
        else:
            communication_maps = confidence_map

        if reset or pre_map is None:
            dynamic_communication_maps = communication_maps
        else:
            dynamic_map = torch.abs(communication_maps - pre_map)
            dynamic_communication_maps = communication_maps * (
                1 / (self.rescale_rate + 1) + dynamic_map * self.rescale_rate
            )

        dynamic_communication_maps = torch.where(
            dynamic_communication_maps > self.thresh,
            torch.ones_like(dynamic_communication_maps),
            torch.zeros_like(dynamic_communication_maps),
        )
        dynamic_communication_maps[:, self.ego_id, ...] = 1

        cur_feat = feat * dynamic_communication_maps
        if self.reconstruct:
            if self.use_decoder:
                cur_feat = cur_feat + self.decoder(cur_feat)
            if not reset and prev_feat is not None:
                if self.reconstruct == "conv":
                    cur_feat = cur_feat + self.conv_fuse(torch.cat((cur_feat, prev_feat), dim=-3))
                elif self.reconstruct == "deformable":
                    cur_feat = cur_feat + self.renew_feat(cur_feat, prev_feat)

        communication_rate = (
            dynamic_communication_maps[:, self.ego_id + 1 :, ...].sum()
            / height
            / width
            / batch_size
        )
        return cur_feat, communication_rate

    def renew_feat(self, sparse_feat, prev_feat):
        batch_size, cav_num, channels, height, width = sparse_feat.shape
        fuse_feat = torch.stack((prev_feat, sparse_feat), dim=2).reshape(
            batch_size * cav_num, 2, channels, height, width
        )
        fuse_feat = self.renew_net(fuse_feat)
        return fuse_feat.reshape(batch_size, cav_num, channels, height, width)
