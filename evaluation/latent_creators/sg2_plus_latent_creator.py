import torch
from evaluation.latent_creators.base_latent_creator import BaseLatentCreator
from torchvision.transforms import transforms
from training.projectors import w_plus_projector


class SG2PlusLatentCreator(BaseLatentCreator):
    def __init__(self, G, domain, projection_steps=2000):  # 2000 for W plus space
        if domain == "human_faces":
            im_size = (1024, 1024)
        elif domain == "churches":
            im_size = (256, 256)

        self.data_preprocess = transforms.Compose(
            [transforms.Resize(im_size), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
        )

        super().__init__("SG2_plus", self.data_preprocess)

        self.G = G
        self.projection_steps = projection_steps

    def run_projection(self, image):
        image = torch.squeeze((image.cuda() + 1) / 2) * 255
        w = w_plus_projector.project(self.G, image, device=torch.device("cuda:0"), num_steps=self.projection_steps)

        return w, None


if __name__ == "__main__":
    latent_creator = SG2PlusLatentCreator()
    latent_creator.create_latents()
