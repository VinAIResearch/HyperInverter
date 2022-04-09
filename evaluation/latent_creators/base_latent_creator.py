import abc
import os

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from utils import data_utils


class InferenceDataset(Dataset):
    def __init__(self, root, transform=None):
        self.paths = sorted(data_utils.make_dataset(root))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        from_path = self.paths[index]
        fname = os.path.basename(from_path).split(".")[0]
        from_im = Image.open(from_path).convert("RGB")
        if self.transform:
            from_im = self.transform(from_im)
        return fname, from_im


class BaseLatentCreator:
    def __init__(self, method_name, data_preprocess=None):
        assert data_preprocess is not None, "Please define data pre-processing script!"
        self.projection_preprocess = data_preprocess
        self.method_name = method_name

    @abc.abstractmethod
    def run_projection(self, image):
        return None

    def create_latents(self, args):
        image_dataset = InferenceDataset(f"{args.input_data_dir}", self.projection_preprocess)
        self.image_dataloader = DataLoader(image_dataset, batch_size=1, shuffle=False)

        base_latent_folder_path = f"{args.saved_embedding_dir}/{args.input_data_id}"
        os.makedirs(base_latent_folder_path, exist_ok=True)
        self.latent_folder_path = f"{base_latent_folder_path}/{self.method_name}"
        os.makedirs(self.latent_folder_path, exist_ok=True)

        image_counter = 0
        for fname, image in tqdm(self.image_dataloader):
            image_counter += image.size(0)
            fname = fname[0]
            cur_latent_folder_path = f"{self.latent_folder_path}/{fname}"
            image = image.cuda()
            w, added_weights = self.run_projection(image)

            os.makedirs(cur_latent_folder_path, exist_ok=True)
            torch.save(w, f"{cur_latent_folder_path}/latent_code.pt")

            if added_weights is not None:
                torch.save(added_weights, f"{cur_latent_folder_path}/added_weights.pt")

            if image_counter >= args.max_num_images:
                break
