import argparse
import os
import sys


sys.path.append(".")
sys.path.append("..")

import torch  # noqa: E402
from evaluation.experiment_setting_creator import ExperimentRunner  # noqa: E402
from PIL import Image  # noqa: E402
from tqdm import tqdm  # noqa: E402
from utils.common import convert_predicted_weights_to_dict  # noqa: E402
from utils.log_utils import save_concat_image  # noqa: E402


class ReconstructionComparison:
    def __init__(self, args):
        self.args = args
        self.experiment_creator = ExperimentRunner(args)

        if self.args.domain == "human_faces":
            self.img_size = (1024, 1024)
        elif self.args.domain == "churches":
            self.img_size = (256, 256)

    def save_reconstruction_images(self, image_latents, added_weights, target_image):
        save_concat_image(
            self.concat_base_dir,
            image_latents,
            added_weights,
            self.experiment_creator.G,
            "reconstruction",
            target_image,
        )

    def create_output_dirs(self, full_image_name):

        output_base_dir_path = f"{self.args.output_dir}/{self.args.input_data_id}/{full_image_name}"
        os.makedirs(output_base_dir_path, exist_ok=True)

        self.concat_base_dir = f"{output_base_dir_path}/reconstruction"
        os.makedirs(self.concat_base_dir, exist_ok=True)

    def get_image_latent_codes(self, image_name):
        image_latents = []
        added_weights_list = []
        for method in self.args.methods:
            image_latents.append(
                torch.load(
                    f"{self.args.saved_embedding_dir}/{self.args.input_data_id}/{method}/{image_name}/latent_code.pt"
                )
            )

            if method == "hyper_inverter":
                added_weights = torch.load(
                    f"{self.args.saved_embedding_dir}/{self.args.input_data_id}/{method}/{image_name}/added_weights.pt"
                )
                added_weights = convert_predicted_weights_to_dict(added_weights)
            else:
                added_weights = None
            added_weights_list.append(added_weights)

        return image_latents, added_weights_list

    def run_experiment(self):
        images_counter = 0

        self.experiment_creator.run_experiment()

        for idx, image_path in tqdm(
            enumerate(self.experiment_creator.images_paths), total=len(self.experiment_creator.images_paths)
        ):

            if images_counter >= self.args.max_num_images:
                break

            image_name = image_path.split(".")[0].split("/")[-1]
            target_image = Image.open(self.experiment_creator.target_paths[idx]).resize(self.img_size)

            # Get latents
            image_latents, added_weights = self.get_image_latent_codes(image_name)
            self.create_output_dirs(image_name)

            # Get reconstruction images
            self.save_reconstruction_images(image_latents, added_weights, target_image)

            target_image.close()
            torch.cuda.empty_cache()
            images_counter += 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", type=str, default="hyper_inverter,e4e")
    parser.add_argument("--domain", type=str, default="")
    parser.add_argument("--input_data_dir", type=str, default="")
    parser.add_argument("--input_data_id", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--saved_embedding_dir", type=str, default="")
    parser.add_argument("--max_num_images", type=int, default=None)
    parser.add_argument("--resize", type=int, default=256)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.methods = args.methods.split(",")
    runner = ReconstructionComparison(args)
    runner.run_experiment()
