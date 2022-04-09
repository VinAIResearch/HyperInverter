import argparse
import os
import shutil
import sys


sys.path.append(".")
sys.path.append("..")

import torch  # noqa: E402
from editings.latent_editor_wrapper import LatentEditorWrapper  # noqa: E402
from evaluation.experiment_setting_creator import ExperimentRunner  # noqa: E402
from PIL import Image  # noqa: E402
from tqdm import tqdm  # noqa: E402
from utils.common import convert_predicted_weights_to_dict  # noqa: E402
from utils.log_utils import get_image_from_latent  # noqa: E402


class EditComparison:
    def __init__(self, args):
        self.args = args
        self.experiment_creator = ExperimentRunner(args)
        self.latent_editor = LatentEditorWrapper(args.domain)

    def create_output_dirs(self, full_image_name):

        output_base_dir_path = f"{self.args.output_dir}/{self.args.input_data_id}/{full_image_name}"
        os.makedirs(output_base_dir_path, exist_ok=True)

        self.edited_base_dir = f"{output_base_dir_path}/edited_images"
        os.makedirs(self.edited_base_dir, exist_ok=True)

        if self.args.save_edited_images:
            # For quantitative evalution
            self.eval_edited_base_dir = f"{self.args.output_dir}/{self.args.input_data_id}/eval_edited_images"
            os.makedirs(self.eval_edited_base_dir, exist_ok=True)

    def get_image_latent_codes(self, image_name, methods):
        image_latents = {}
        added_weights_dict = {}
        for method in methods:
            image_latents[method] = torch.load(
                f"{self.args.saved_embedding_dir}/{self.args.input_data_id}/{method}/{image_name}/latent_code.pt"
            )
            if method == "hyper_inverter":
                added_weights = torch.load(
                    f"{self.args.saved_embedding_dir}/{self.args.input_data_id}/{method}/{image_name}/added_weights.pt"
                )
                added_weights = convert_predicted_weights_to_dict(added_weights)
            else:
                added_weights = None
            added_weights_dict[method] = added_weights

        return image_latents, added_weights_dict

    def save_interfacegan_edits(
        self, image_latents, added_weights, factors, direction, resize, is_save_edited_images, image_name, gif_speed
    ):
        methods = image_latents.keys()
        inv_edits = {}
        for method, latent in image_latents.items():
            inv_edits[method] = self.latent_editor.get_single_interface_gan_edits_with_direction(
                latent, factors, direction
            )

        for method in inv_edits:
            method_saved_dir = os.path.join(self.edited_base_dir, method)
            os.makedirs(method_saved_dir, exist_ok=True)

            direction_saved_dir = os.path.join(method_saved_dir, direction)
            if os.path.exists(direction_saved_dir):
                shutil.rmtree(direction_saved_dir)
            os.makedirs(direction_saved_dir)

            if is_save_edited_images:
                edited_image_save_dir = os.path.join(direction_saved_dir, "edited_images")
                os.makedirs(edited_image_save_dir, exist_ok=True)

            if self.args.save_edited_images:
                quan_eval_editing_dir = os.path.join(self.eval_edited_base_dir, direction)
                os.makedirs(quan_eval_editing_dir, exist_ok=True)
                for factor in factors:
                    os.makedirs(f"{quan_eval_editing_dir}/{factor}", exist_ok=True)
                    for methods in methods:
                        os.makedirs(f"{quan_eval_editing_dir}/{factor}/{method}", exist_ok=True)

            edited_latents = inv_edits[method]
            edited_images = []
            for factor, edited_latent in zip(factors, edited_latents):
                saved_image = get_image_from_latent(
                    edited_latent, added_weights[method], self.experiment_creator.G, resize
                )
                edited_images.append(saved_image)
                if is_save_edited_images:
                    saved_image.save(os.path.join(edited_image_save_dir, f"{factor}.png"))
                    saved_image.save(f"{quan_eval_editing_dir}/{factor}/{method}/{image_name}.png")

            # Create gif animation
            edited_images += edited_images[::-1]
            edited_images[0].save(
                os.path.join(direction_saved_dir, f"{method}.gif"),
                save_all=True,
                append_images=edited_images[1:],
                optimize=False,
                duration=int(1 / gif_speed * 100),
                loop=0,
            )

    def save_ganspace_edits(
        self, image_latents, added_weights, factors, direction, resize, is_save_edited_images, image_name, gif_speed
    ):
        methods = image_latents.keys()
        inv_edits = {}
        for method, latent in image_latents.items():
            inv_edits[method] = self.latent_editor.get_single_ganspace_edits_with_direction(latent, factors, direction)

        for method in inv_edits:
            method_saved_dir = os.path.join(self.edited_base_dir, method)
            os.makedirs(method_saved_dir, exist_ok=True)

            direction_saved_dir = os.path.join(method_saved_dir, direction)
            if os.path.exists(direction_saved_dir):
                shutil.rmtree(direction_saved_dir)
            os.makedirs(direction_saved_dir)

            if is_save_edited_images:
                edited_image_save_dir = os.path.join(direction_saved_dir, "edited_images")
                os.makedirs(edited_image_save_dir, exist_ok=True)

            if self.args.save_edited_images:
                quan_eval_editing_dir = os.path.join(self.eval_edited_base_dir, direction)
                os.makedirs(quan_eval_editing_dir, exist_ok=True)
                for factor in factors:
                    os.makedirs(f"{quan_eval_editing_dir}/{factor}", exist_ok=True)
                    for methods in methods:
                        os.makedirs(f"{quan_eval_editing_dir}/{factor}/{method}", exist_ok=True)

            edited_latents = inv_edits[method]
            edited_images = []
            for factor, edited_latent in zip(factors, edited_latents):
                saved_image = get_image_from_latent(
                    edited_latent, added_weights[method], self.experiment_creator.G, resize
                )
                edited_images.append(saved_image)

                if is_save_edited_images:
                    saved_image.save(os.path.join(edited_image_save_dir, f"{factor}.png"))
                    saved_image.save(f"{quan_eval_editing_dir}/{factor}/{method}/{image_name}.png")

            # Create gif animation
            edited_images += edited_images[::-1]
            edited_images[0].save(
                os.path.join(direction_saved_dir, f"{method}.gif"),
                save_all=True,
                append_images=edited_images[1:],
                optimize=False,
                duration=int(1 / gif_speed * 100),
                loop=0,
            )

    def save_styleclip_latent_mapper_edits(
        self, image_latents, added_weights, factors, direction, resize, is_save_edited_images, image_name, gif_speed
    ):
        methods = image_latents.keys()
        inv_edits = {}
        for method, latent in image_latents.items():
            inv_edits[method] = self.latent_editor.get_single_styleclip_latent_mapper_edits_with_direction(
                latent, factors, direction
            )

        for method in inv_edits:
            method_saved_dir = os.path.join(self.edited_base_dir, method)
            os.makedirs(method_saved_dir, exist_ok=True)

            direction_saved_dir = os.path.join(method_saved_dir, direction)
            if os.path.exists(direction_saved_dir):
                shutil.rmtree(direction_saved_dir)
            os.makedirs(direction_saved_dir)

            if is_save_edited_images:
                edited_image_save_dir = os.path.join(direction_saved_dir, "edited_images")
                os.makedirs(edited_image_save_dir, exist_ok=True)

            if self.args.save_edited_images:
                quan_eval_editing_dir = os.path.join(self.eval_edited_base_dir, direction)
                os.makedirs(quan_eval_editing_dir, exist_ok=True)
                for factor in factors:
                    os.makedirs(f"{quan_eval_editing_dir}/{factor}", exist_ok=True)
                    for methods in methods:
                        os.makedirs(f"{quan_eval_editing_dir}/{factor}/{method}", exist_ok=True)

            edited_latents = inv_edits[method]
            edited_images = []
            for factor, edited_latent in zip(factors, edited_latents):
                saved_image = get_image_from_latent(
                    edited_latent, added_weights[method], self.experiment_creator.G, resize
                )
                edited_images.append(saved_image)

                if is_save_edited_images:
                    saved_image.save(os.path.join(edited_image_save_dir, f"{factor}.png"))
                    saved_image.save(f"{quan_eval_editing_dir}/{factor}/{method}/{image_name}.png")

            # Create gif animation
            edited_images += edited_images[::-1]
            edited_images[0].save(
                os.path.join(direction_saved_dir, f"{method}.gif"),
                save_all=True,
                append_images=edited_images[1:],
                optimize=False,
                duration=int(1 / gif_speed * 100),
                loop=0,
            )

    def run_experiment(self):
        images_counter = 0
        interfacegan_factors = [
            val / 10.0 for val in range(self.args.min_factor, self.args.max_factor + self.args.step, self.args.step)
        ]
        ganspace_factors = [
            val / 10.0 for val in range(self.args.min_factor, self.args.max_factor + self.args.step, self.args.step)
        ]
        styleclip_mapper_factors = [
            val / 100.0 for val in range(self.args.min_factor, self.args.max_factor + self.args.step, self.args.step)
        ]
        self.experiment_creator.run_experiment()

        for idx, image_path in tqdm(
            enumerate(self.experiment_creator.images_paths), total=len(self.experiment_creator.images_paths)
        ):

            if images_counter >= self.args.max_num_images:
                break

            image_name = image_path.split(".")[0].split("/")[-1]
            target_image = Image.open(self.experiment_creator.target_paths[idx])

            # Get latents
            image_latents, added_weights = self.get_image_latent_codes(image_name, self.args.methods)
            self.create_output_dirs(image_name)

            # Run image manipulation
            if self.args.domain == "human_faces":
                for direction in self.args.directions:
                    if direction in ["age", "smile", "rotation"]:  # INTERFACEGAN directions

                        self.save_interfacegan_edits(
                            image_latents,
                            added_weights,
                            interfacegan_factors,
                            direction,
                            self.args.resize,
                            self.args.save_edited_images,
                            image_name,
                            self.args.gif_speed,
                        )

                    elif direction in [
                        "surprised",
                        "afro",
                        "angry",
                        "beyonce",
                        "bobcut",
                        "bowlcut",
                        "curly_hair",
                        "hilary_clinton",
                        "depp",
                        "mohawk",
                        "purple_hair",
                        "taylor_swift",
                        "trump",
                        "zuckerberg",
                    ]:  # STYLECLIP directions from latent mappers

                        self.save_styleclip_latent_mapper_edits(
                            image_latents,
                            added_weights,
                            styleclip_mapper_factors,
                            direction,
                            self.args.resize,
                            self.args.save_edited_images,
                            image_name,
                            self.args.gif_speed,
                        )

                    elif direction in [
                        "eye_openness",
                        "trimmed_beard",
                        "lipstick",
                        "face_roundness",
                        "nose_length",
                        "eyebrow_thickness",
                        "head_angle_up",
                        "displeased",
                    ]:  # GANSPACE directions

                        self.save_ganspace_edits(
                            image_latents,
                            added_weights,
                            ganspace_factors,
                            direction,
                            self.args.resize,
                            self.args.save_edited_images,
                            image_name,
                            self.args.gif_speed,
                        )
                    else:
                        raise Exception(f"Direction {direction} is not supported!")

            elif self.args.domain == "churches":
                for direction in self.args.directions:
                    if direction in ["clouds", "vibrant", "blue_skies", "trees"]:  # GANSPACE directions
                        self.save_ganspace_edits(
                            image_latents,
                            added_weights,
                            ganspace_factors,
                            direction,
                            self.args.resize,
                            self.args.save_edited_images,
                            image_name,
                            self.args.gif_speed,
                        )
                    else:
                        raise Exception(f"Direction {direction} is not supported!")

            target_image.close()
            torch.cuda.empty_cache()
            images_counter += 1


def run_full_edit(args):
    edit_figure_creator = EditComparison(args=args)
    edit_figure_creator.run_experiment()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", type=str, default="hyper_inverter,e4e")
    parser.add_argument("--domain", type=str, default="")
    parser.add_argument("--input_data_dir", type=str, default="")
    parser.add_argument("--input_data_id", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--saved_embedding_dir", type=str, default="")
    parser.add_argument("--max_num_images", type=int, default=None)
    parser.add_argument("--min_factor", type=int, default=-30)
    parser.add_argument("--max_factor", type=int, default=35)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--directions", type=str, default="")
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--save_edited_images", action="store_true")
    parser.add_argument("--gif_speed", type=int, default=2)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.directions = args.directions.split(",")
    args.methods = args.methods.split(",")

    run_full_edit(args)
