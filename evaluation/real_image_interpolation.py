import argparse
import os
import pickle
import sys


sys.path.append(".")
sys.path.append("..")

import numpy as np  # noqa: E402
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
from PIL import Image  # noqa: E402
from utils.common import convert_predicted_weights_to_dict, tensor2im  # noqa: E402
from utils.log_utils import get_concat_h  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", type=str, default="hyper_inverter")
    parser.add_argument("--domain", type=str, default="human_faces")
    parser.add_argument("--left_image_path", type=str)
    parser.add_argument("--right_image_path", type=str)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--saved_image_size", type=int, default=256)
    parser.add_argument("--saved_dir", type=str)
    parser.add_argument("--saved_file_name", type=str)
    parser.add_argument("--save_interpolated_images", action="store_true")
    parser.add_argument("--gif_speed", type=int, default=2)
    args = parser.parse_args()
    return args


args = parse_args()
saved_image_size = (args.saved_image_size, args.saved_image_size)


def linear_interpolate_weights(aw1, aw2, steps=10):
    aws_interps = []
    linspace = np.linspace(0, 1, steps)
    layer_names = aw1.keys()
    for l in linspace:
        aw_interp = {}
        for ln in layer_names:
            aw_interp[ln] = aw1[ln] * l + (1 - l) * aw2[ln]
        aws_interps.append(aw_interp)
    return aws_interps


def linear_interpolate(w1, w2, steps=10):
    linspace = np.linspace(0, 1, steps)
    interps = []
    for l in linspace:
        interps.append(w1 * l + (1 - l) * w2)
    return interps


# Load StyleGAN generator
if args.domain == "human_faces":
    model_path = paths_config.model_paths["stylegan2_ada_ffhq"]
elif args.domain == "churches":
    model_path = paths_config.model_paths["stylegan2_ada_church"]
else:
    raise Exception(f"{args.domain} is not supported!")

with open(model_path, "rb") as f:
    G_ckpt = pickle.load(f)["G_ema"]
    G_ckpt = G_ckpt.float()
G = Generator(**G_ckpt.init_kwargs)
G.load_state_dict(G_ckpt.state_dict())
G.cuda().eval()

for method in args.methods.split(","):
    print("Start to interpolate using: ", method)
    # Latent Creator
    if method == "e4e":
        latent_creator = E4ELatentCreator(domain=args.domain)
    elif method == "SG2_plus":
        latent_creator = SG2PlusLatentCreator(G=G, domain=args.domain)
    elif method == "SG2":
        latent_creator = SG2LatentCreator(G=G, domain=args.domain)
    elif method == "w_encoder":
        latent_creator = WEncoderLatentCreator(domain=args.domain)
    elif method == "hyper_inverter":
        latent_creator = HyperInverterLatentCreator(domain=args.domain)
    elif method == "psp":
        latent_creator = PSPLatentCreator(domain=args.domain)
    elif method == "restyle_e4e":
        latent_creator = ReStyle_E4ELatentCreator(domain=args.domain)
    else:
        raise ("Not implemented!")

    # Read images
    left_image = Image.open(args.left_image_path).convert("RGB")
    right_image = Image.open(args.right_image_path).convert("RGB")

    # Prepare input images
    left_image = latent_creator.inversion_pre_process(left_image).unsqueeze(0).cuda()
    right_image = latent_creator.inversion_pre_process(right_image).unsqueeze(0).cuda()

    # Inversion
    left_latent, left_added_weights = latent_creator.run_projection(left_image)
    right_latent, right_added_weights = latent_creator.run_projection(right_image)

    # Linear interpolate latent codes
    latent_interps = linear_interpolate(left_latent, right_latent, args.steps)

    interp_images = []
    if method == "hyper_inverter":
        # Linear interpolate added weights
        added_weights_interps = linear_interpolate_weights(left_added_weights, right_added_weights, args.steps)
        for latent, added_weights in zip(latent_interps, added_weights_interps):
            added_weights = convert_predicted_weights_to_dict(added_weights)
            interp_image = G.synthesis(latent, added_weights=added_weights, noise_mode="const")[0]
            interp_image = tensor2im(interp_image).resize(saved_image_size)
            interp_images.append(interp_image)
    else:
        for latent in latent_interps:
            interp_image = G.synthesis(latent, added_weights=None, noise_mode="const")[0]
            interp_image = tensor2im(interp_image).resize(saved_image_size)
            interp_images.append(interp_image)

    os.makedirs(f"{args.saved_dir}/{method}", exist_ok=True)
    os.makedirs(f"{args.saved_dir}/{method}/{args.saved_file_name}", exist_ok=True)

    concat_image = None
    if args.save_interpolated_images:
        interp_gammas = list(np.linspace(0, 1, args.steps))
        for gamma, interp_image in zip(interp_gammas, interp_images):
            gamma = np.round(gamma, 2)
            interp_image.save(f"{args.saved_dir}/{method}/{args.saved_file_name}/{gamma}.png")
            concat_image = get_concat_h(concat_image, interp_image, gap=10)
        concat_image.save(f"{args.saved_dir}/{method}/{args.saved_file_name}.png")

    # Save gif
    interp_images += interp_images[::-1]
    interp_images[0].save(
        f"{args.saved_dir}/{method}/{args.saved_file_name}.gif",
        save_all=True,
        append_images=interp_images[1:],
        optimize=False,
        duration=int(1 / args.gif_speed * 100),
        loop=0,
    )

    print("Done!")
