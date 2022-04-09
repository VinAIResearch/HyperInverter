import argparse

import torch
from configs import paths_config
from evaluation.latent_creators.base_latent_creator import BaseLatentCreator
from models.hyper_inverter import HyperInverter
from torchvision.transforms import transforms


class HyperInverterLatentCreator(BaseLatentCreator):
    def __init__(self, domain=None):
        if domain == "human_faces":
            im_size = (1024, 1024)
            model_path = paths_config.model_paths["hyper_inverter_ffhq"]
        elif domain == "churches":
            im_size = (256, 256)
            model_path = paths_config.model_paths["hyper_inverter_church"]

        self.inversion_pre_process = transforms.Compose(
            [
                transforms.Resize(im_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        super().__init__("hyper_inverter", self.inversion_pre_process)

        ckpt = torch.load(model_path, map_location="cpu")
        opts = ckpt["opts"]
        opts["batch_size"] = 1
        opts["checkpoint_path"] = model_path
        opts = argparse.Namespace(**opts)
        self.inversion_net = HyperInverter(opts)
        self.inversion_net.eval()
        self.inversion_net = self.inversion_net.cuda()

    def run_projection(self, image):
        _, _, predicted_weights, latent = self.inversion_net(image, return_latents=True)
        added_weights = {}
        for key in predicted_weights:
            added_weights[key] = predicted_weights[key][0]
        return latent, added_weights


if __name__ == "__main__":
    latent_creator = HyperInverterLatentCreator()
    latent_creator.create_latents()
