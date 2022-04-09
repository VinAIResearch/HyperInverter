from argparse import Namespace

import torch
from configs import paths_config
from evaluation.latent_creators.base_latent_creator import BaseLatentCreator
from models.w_encoder import W_Encoder
from torchvision.transforms import transforms


class WEncoderLatentCreator(BaseLatentCreator):
    def __init__(self, domain=None):
        self.inversion_pre_process = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        super().__init__("w_encoder", self.inversion_pre_process)

        if domain == "human_faces":
            model_path = paths_config.model_paths["w_encoder_ffhq"]
        elif domain == "churches":
            model_path = paths_config.model_paths["w_encoder_church"]

        ckpt = torch.load(model_path, map_location="cpu")
        opts = ckpt["opts"]
        opts["batch_size"] = 1
        opts["checkpoint_path"] = model_path
        opts = Namespace(**opts)
        self.inversion_net = W_Encoder(opts)
        self.inversion_net.eval()
        self.inversion_net = self.inversion_net.cuda()

    def run_projection(self, image):
        _, latent = self.inversion_net(image, return_latents=True)

        return latent, None


if __name__ == "__main__":
    latent_creator = WEncoderLatentCreator()
    latent_creator.create_latents()
