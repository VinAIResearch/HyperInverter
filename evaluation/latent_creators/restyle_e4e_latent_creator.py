import argparse

import torch
from configs import paths_config
from evaluation.latent_creators.base_latent_creator import BaseLatentCreator
from models.restyle.e4e import e4e
from models.restyle.inference_utils import get_average_image, run_on_batch
from torchvision.transforms import transforms


class ReStyle_E4ELatentCreator(BaseLatentCreator):
    def __init__(self, domain=None):
        self.inversion_pre_process = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        super().__init__("restyle_e4e", self.inversion_pre_process)

        if domain == "human_faces":
            model_path = paths_config.model_paths["official_restyle_e4e_ffhq"]
        elif domain == "churches":
            model_path = paths_config.model_paths["official_restyle_e4e_church"]

        ckpt = torch.load(model_path, map_location="cpu")
        opts = ckpt["opts"]
        opts["batch_size"] = 1
        opts["checkpoint_path"] = model_path
        opts["n_iters_per_batch"] = 5
        opts["resize_outputs"] = False
        opts = argparse.Namespace(**opts)
        self.opts = opts
        self.net = e4e(self.opts)
        self.net.eval()
        self.net = self.net.cuda()

    def run_projection(self, image):
        avg_image = get_average_image(self.net, self.opts)
        _, latent = run_on_batch(image, self.net, self.opts, avg_image)
        latent = latent[0][-1].unsqueeze(0)

        return latent, None


if __name__ == "__main__":
    latent_creator = ReStyle_E4ELatentCreator()
    latent_creator.create_latents()
