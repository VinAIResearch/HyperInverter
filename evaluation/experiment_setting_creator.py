import glob
import os
import pickle
import sys


sys.path.append(".")
sys.path.append("..")

import torch  # noqa: E402
from configs import paths_config  # noqa: E402
from evaluation.latent_creators import (  # noqa: E402
    E4ELatentCreator,
    HyperInverterLatentCreator,
    PSPLatentCreator,
    ReStyle_E4ELatentCreator,
    SG2LatentCreator,
    SG2PlusLatentCreator,
    WEncoderLatentCreator,
)
from models.stylegan2_ada import Generator  # noqa: E402
from utils.common import toogle_grad  # noqa: E402


class ExperimentRunner:
    def __init__(self, args):
        self.args = args
        self.images_paths = sorted(glob.glob(f"{self.args.input_data_dir}/*"))
        self.target_paths = sorted(glob.glob(f"{self.args.input_data_dir}/*"))
        self.sampled_ws = None

        # Load pretrained Generator
        if self.args.domain == "human_faces":
            model_path = paths_config.model_paths["stylegan2_ada_ffhq"]
        elif self.args.domain == "churches":
            model_path = paths_config.model_paths["stylegan2_ada_church"]
        else:
            raise Exception("Not defined!")

        print(f"Load generator from {model_path}")
        with open(model_path, "rb") as f:
            G = pickle.load(f)["G_ema"]
            G = G.float()
        self.G = Generator(**G.init_kwargs)
        self.G.load_state_dict(G.state_dict())
        self.G.cuda().eval()
        toogle_grad(self.G, False)

    def run_experiment(self):
        for method in self.args.methods:
            if method == "psp":
                latent_creator = PSPLatentCreator(domain=self.args.domain)
                latent_creator.create_latents(self.args)
            elif method == "e4e":
                latent_creator = E4ELatentCreator(domain=self.args.domain)
                latent_creator.create_latents(self.args)
            elif method == "SG2_plus":
                latent_creator = SG2PlusLatentCreator(G=self.G, domain=self.args.domain)
                latent_creator.create_latents(self.args)
            elif method == "SG2":
                latent_creator = SG2LatentCreator(G=self.G, domain=self.args.domain)
                latent_creator.create_latents(self.args)
            elif method == "w_encoder":
                latent_creator = WEncoderLatentCreator(domain=self.args.domain)
                latent_creator.create_latents(self.args)
            elif method == "hyper_inverter":
                latent_creator = HyperInverterLatentCreator(domain=self.args.domain)
                latent_creator.create_latents(self.args)
            elif method == "restyle_e4e":
                latent_creator = ReStyle_E4ELatentCreator(domain=self.args.domain)
                latent_creator.create_latents(self.args)
            else:
                raise ("Not implemented!")

        torch.cuda.empty_cache()


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = 1

    runner = ExperimentRunner()
    runner.run_experiment(True)
