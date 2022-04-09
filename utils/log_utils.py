import torch
from PIL import Image


def save_concat_image(base_dir, image_latents, added_weights, G, file_name, extra_image=None):
    images_to_save = []
    for latent, aws in zip(image_latents, added_weights):
        images_to_save.append(get_image_from_w(latent, aws, G))
    result_image = create_alongside_images(images_to_save)
    result_image = get_concat_h(extra_image, result_image)
    result_image.save(f"{base_dir}/{file_name}.jpg")


def get_image_from_latent(image_latent, added_weight, G, resize):
    image_to_save = get_image_from_w(image_latent, added_weight, G)
    image_to_save = Image.fromarray(image_to_save, mode="RGB").resize((resize, resize))
    return image_to_save


def get_concat_h(im1, im2, gap=10):
    if im1 is None:
        return im2
    dst = Image.new("RGB", (im1.width + im2.width + gap, im1.height), "WHITE")
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width + gap, 0))
    return dst


def create_alongside_images(images):
    res = None
    for img in images:
        res = get_concat_h(res, Image.fromarray(img, mode="RGB"))

    return res


def get_image_from_w(w, added_weights, G):
    if len(w.size()) <= 2:
        w = w.unsqueeze(0)
    with torch.no_grad():
        img = G.synthesis(w, added_weights=added_weights, noise_mode="const")
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).detach().cpu().numpy()
    return img[0]
