import argparse
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from configs.paths_config import model_paths
from models.encoders import fpn_encoders
from models.stylegan2_ada import Generator
from utils import common


class W_Encoder(nn.Module):
    def __init__(self, opts):
        super().__init__()

        # Configurations
        self.set_opts(opts)

        # Define and load architecture
        self.load_weights()

        # For visualization
        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))

    def load_weights(self):
        if self.opts.dataset_type == "ffhq_encode":
            w_encoder_path = model_paths["w_encoder_ffhq"]
        elif self.opts.dataset_type == "church_encode":
            w_encoder_path = model_paths["w_encoder_church"]
        ckpt = torch.load(w_encoder_path, map_location="cpu")
        print(f"Loaded pretrained W encoder from: {w_encoder_path}")

        opts = ckpt["opts"]
        opts = argparse.Namespace(**opts)

        if "ffhq" in self.opts.dataset_type or "celeb" in self.opts.dataset_type:
            # Using ResNet-IRSE50 for facial domain
            self.w_encoder = fpn_encoders.BackboneEncoderUsingLastLayerIntoW(50, "ir_se", opts)
        else:
            # Using ResNet34 pre-trained on ImageNet for other domains
            self.w_encoder = fpn_encoders.ResNetEncoderUsingLastLayerIntoW()

        self.w_encoder.load_state_dict(common.get_keys(ckpt, "encoder"), strict=True)
        self.w_encoder.to(self.opts.device).eval()
        common.toogle_grad(self.w_encoder, False)

        # Load pretrained StyleGAN2-ADA models
        if self.opts.dataset_type == "ffhq_encode":
            stylegan_ckpt_path = model_paths["stylegan2_ada_ffhq"]
        elif self.opts.dataset_type == "church_encode":
            stylegan_ckpt_path = model_paths["stylegan2_ada_church"]

        with open(stylegan_ckpt_path, "rb") as f:
            ckpt = pickle.load(f)
            # Generator
            G_original = ckpt["G_ema"]
            G_original = G_original.float()

        self.decoder = Generator(**G_original.init_kwargs)
        self.decoder.load_state_dict(G_original.state_dict())
        self.decoder.to(self.opts.device).eval()

        # Load latent average
        self.latent_avg = self.decoder.mapping.w_avg

    def forward(self, x, return_latents=False):
        num_ws = self.decoder.mapping.num_ws

        # Resize image to feed to encoder
        x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)

        # Obtain w code via W Encoder
        w_codes = self.w_encoder(x)

        # Normalize with respect to the center of an average face
        w_codes = w_codes + self.latent_avg.repeat(w_codes.shape[0], 1)
        w_codes = w_codes.unsqueeze(1).repeat([1, num_ws, 1])

        # Genenerate W-images
        w_images = self.decoder.synthesis(w_codes, added_weights=None, noise_mode="const")

        return_data = [w_images]
        if return_latents:
            return_data.append(w_codes)

        return tuple(return_data)

    def set_opts(self, opts):
        self.opts = opts
