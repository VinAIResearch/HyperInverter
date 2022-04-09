import math
import pickle

import torch
from configs.paths_config import model_paths
from models.encoders import fpn_encoders
from models.stylegan2_ada import Generator
from torch import nn
from utils.model_utils import RESNET_MAPPING


def get_keys(d, name):
    if "state_dict" in d:
        d = d["state_dict"]
    d_filt = {k[len(name) + 1 :]: v for k, v in d.items() if k[: len(name)] == name}
    return d_filt


class pSp(nn.Module):
    def __init__(self, opts):
        super(pSp, self).__init__()
        self.set_opts(opts)
        # compute number of style inputs based on the output resolution
        self.opts.n_styles = int(math.log(self.opts.output_size, 2)) * 2 - 2
        # Define architecture
        self.encoder = self.set_encoder()
        # self.decoder = Generator(self.opts.output_size, 512, 8)
        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        # Load weights if needed
        self.load_weights()

    def set_encoder(self):
        if self.opts.encoder_type == "GradualStyleEncoder":
            encoder = fpn_encoders.GradualStyleEncoder(50, "ir_se", self.opts)
        elif self.opts.encoder_type == "BackboneEncoderUsingLastLayerIntoW":
            encoder = fpn_encoders.BackboneEncoderUsingLastLayerIntoW(50, "ir_se", self.opts)
        elif self.opts.encoder_type == "ResNetEncoderUsingLastLayerIntoW":
            encoder = fpn_encoders.ResNetEncoderUsingLastLayerIntoW()
        elif self.opts.encoder_type == "ResNetGradualStyleEncoder":
            encoder = fpn_encoders.ResNetGradualStyleEncoder(n_styles=self.opts.n_styles)

        else:
            raise Exception("{} is not a valid encoders".format(self.opts.encoder_type))
        return encoder

    def __get_encoder_checkpoint(self):
        if "ffhq" in self.opts.dataset_type:
            print("Loading encoders weights from irse50!")
            encoder_ckpt = torch.load(model_paths["ir_se50"])
            return encoder_ckpt
        else:
            print("Loading encoders weights from resnet34!")
            encoder_ckpt = torch.load(model_paths["resnet34"])
            mapped_encoder_ckpt = dict(encoder_ckpt)
            for p, v in encoder_ckpt.items():
                for original_name, psp_name in RESNET_MAPPING.items():
                    if original_name in p:
                        mapped_encoder_ckpt[p.replace(original_name, psp_name)] = v
                        mapped_encoder_ckpt.pop(p)
            return encoder_ckpt

    def load_weights(self):
        print("Loading decoder weights from pretrained!")
        with open(self.opts.stylegan_weights, "rb") as f:
            G_original = pickle.load(f)["G_ema"]
            G_original = G_original.float()

        decoder = Generator(**G_original.init_kwargs)
        decoder.load_state_dict(G_original.state_dict())
        decoder.to(self.opts.device)
        decoder = decoder.float()
        if self.opts.train_decoder:
            self.decoder = decoder.train()
        else:
            self.decoder = decoder.eval()

        if self.opts.checkpoint_path is not None:
            print("Loading pSp from checkpoint: {}".format(self.opts.checkpoint_path))
            ckpt = torch.load(self.opts.checkpoint_path, map_location="cpu")
            self.encoder.load_state_dict(get_keys(ckpt, "encoder"), strict=True)
            self.encoder.to(self.opts.device)
            self.encoder.train()
            self.__load_latent_avg(ckpt)
        else:
            encoder_ckpt = self.__get_encoder_checkpoint()
            self.encoder.load_state_dict(encoder_ckpt, strict=False)
            self.encoder.train()
            self.latent_avg = None

    def forward(self, x, resize=True, return_latents=False):

        # Encoding image to latent code
        codes = self.encoder(x)

        # normalize with respect to the center of an average face
        if self.opts.start_from_latent_avg:
            if self.opts.learn_in_w:
                codes = codes + self.latent_avg.repeat(codes.shape[0], 1)
                codes = codes.unsqueeze(1).repeat([1, self.decoder.mapping.num_ws, 1])
            else:
                codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)

        images = self.decoder.synthesis(codes, noise_mode="const", force_fp32=True)

        if resize:
            images = self.face_pool(images)

        if return_latents:
            return images, codes
        else:
            return images

    def set_opts(self, opts):
        self.opts = opts

    def __load_latent_avg(self, ckpt, repeat=None):
        if "latent_avg" in ckpt:
            self.latent_avg = ckpt["latent_avg"].to(self.opts.device)
            if repeat is not None:
                self.latent_avg = self.latent_avg.repeat(repeat, 1)
        else:
            self.latent_avg = None
