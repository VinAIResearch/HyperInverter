import argparse
import os
import sys

import torch
from configs import paths_config
from editings.latent_editor import LatentEditor
from editings.styleclip.mapper.styleclip_mapper import StyleCLIPMapper
from utils.styleclip_utils import ensure_checkpoint_exists


sys.path.append(".")
sys.path.append("..")


class LatentEditorWrapper:
    def __init__(self, domain="human_faces"):

        self.domain = domain

        if self.domain == "human_faces":
            self.interfacegan_directions = {
                "age": f"{paths_config.editing_paths['interfacegan_age']}",
                "smile": f"{paths_config.editing_paths['interfacegan_smile']}",
                "rotation": f"{paths_config.editing_paths['interfacegan_rotation']}",
            }

            self.interfacegan_directions_tensors = {
                name: torch.load(path).cuda() for name, path in self.interfacegan_directions.items()
            }

            self.ganspace_pca = torch.load(f"{paths_config.editing_paths['ffhq_pca']}")

            self.ganspace_directions = {
                "eye_openness": (54, 7, 8, 5),
                "trimmed_beard": (58, 7, 9, 7),
                "lipstick": (34, 10, 11, 20),
                "face_roundness": (37, 0, 5, 20.0),
                "nose_length": (51, 4, 5, -30.0),
                "eyebrow_thickness": (37, 8, 9, 20.0),
                "head_angle_up": (11, 1, 4, -10.5),
                "displeased": (36, 4, 7, 10.0),
            }

            self.styleclip_meta_data = {
                "afro": [False, False, True],
                "angry": [False, False, True],
                "beyonce": [False, False, False],
                "bobcut": [False, False, True],
                "bowlcut": [False, False, True],
                "curly_hair": [False, False, True],
                "hilary_clinton": [False, False, False],
                "depp": [False, False, False],
                "mohawk": [False, False, True],
                "purple_hair": [False, False, False],
                "surprised": [False, False, True],
                "taylor_swift": [False, False, False],
                "trump": [False, False, False],
                "zuckerberg": [False, False, False],
            }

        elif self.domain == "churches":
            self.ganspace_pca = torch.load(f"{paths_config.editing_paths['church_pca']}")

            self.ganspace_directions = {
                "clouds": (20, 7, 9, -20.0),
                "vibrant": (8, 12, 14, -20.0),
                "blue_skies": (11, 9, 14, 9.9),
                "trees": (12, 5, 6, -19.1),
            }

        self.latent_editor = LatentEditor()

    def get_single_styleclip_latent_mapper_edits_with_direction(self, start_w, factors, direction):
        latents_to_display = []
        mapper_checkpoint_path = os.path.join(
            paths_config.styleclip_paths["style_clip_pretrained_mappers"], f"{direction}.pt"
        )
        ensure_checkpoint_exists(str(mapper_checkpoint_path))
        ckpt = torch.load(mapper_checkpoint_path, map_location="cpu")
        opts = ckpt["opts"]
        styleclip_opts = argparse.Namespace(
            **{
                "mapper_type": "LevelsMapper",
                "no_coarse_mapper": self.styleclip_meta_data[direction][0],
                "no_medium_mapper": self.styleclip_meta_data[direction][1],
                "no_fine_mapper": self.styleclip_meta_data[direction][2],
                "stylegan_size": 1024,
                "checkpoint_path": mapper_checkpoint_path,
            }
        )
        opts.update(vars(styleclip_opts))
        opts = argparse.Namespace(**opts)
        style_clip_net = StyleCLIPMapper(opts)
        style_clip_net.eval()
        style_clip_net.cuda()

        for factor in factors:
            edited_latent = start_w + factor * style_clip_net.mapper(start_w)
            latents_to_display.append(edited_latent)

        return latents_to_display

    def get_single_ganspace_edits_with_direction(self, start_w, factors, direction):
        latents_to_display = []
        for factor in factors:
            ganspace_direction = self.ganspace_directions[direction]
            edit_direction = list(ganspace_direction)
            edit_direction[-1] = factor
            edit_direction = tuple(edit_direction)
            new_w = self.latent_editor.apply_ganspace(start_w, self.ganspace_pca, [edit_direction])
            latents_to_display.append(new_w)
        return latents_to_display

    def get_single_interface_gan_edits_with_direction(self, start_w, factors, direction):
        latents_to_display = []
        for factor in factors:
            latents_to_display.append(
                self.latent_editor.apply_interfacegan(
                    start_w, self.interfacegan_directions_tensors[direction], factor / 2
                )
            )
        return latents_to_display
