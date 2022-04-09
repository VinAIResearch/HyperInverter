import argparse

import torch
from configs import paths_config
from evaluation.latent_creators.base_latent_creator import BaseLatentCreator
from models.psp.model import pSp
from torchvision.transforms import transforms


class PSPLatentCreator(BaseLatentCreator):
    def __init__(self, domain=None):
        self.inversion_pre_process = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        super().__init__("psp", self.inversion_pre_process)

        if domain == "human_faces":
            model_path = paths_config.model_paths["official_psp_ffhq"]
        elif domain == "churches":
            model_path = paths_config.model_paths["official_psp_church"]

        ckpt = torch.load(model_path, map_location="cpu")
        opts = ckpt["opts"]
        opts["batch_size"] = 1
        opts["checkpoint_path"] = model_path
        opts = argparse.Namespace(**opts)
        self.inversion_net = pSp(opts)
        self.inversion_net.eval()
        self.inversion_net = self.inversion_net.cuda()

    def run_projection(self, image):
        _, latent = self.inversion_net(image, return_latents=True, resize=False)

        return latent, None


if __name__ == "__main__":
    latent_creator = PSPLatentCreator()
    latent_creator.create_latents()
