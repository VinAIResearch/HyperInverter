import argparse
import os

import piq
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", type=str, default=50, help="Batch size to use")
    parser.add_argument("--fake_path", type=str, default="")
    parser.add_argument("--real_path", type=str, default="")
    parser.add_argument("--result_file", type=str, default="")
    parser.add_argument("--image_height", type=int, default=1024)
    parser.add_argument("--image_width", type=int, default=1024)
    return parser.parse_args()


args = parse_args()

image_transform = transforms.ToTensor()
image_size = (args.image_width, args.image_height)
real_filenames = sorted(os.listdir(args.real_path))
fake_filenames = sorted(os.listdir(args.fake_path))

scores = []
for real_fn, fake_fn in tqdm(zip(real_filenames, fake_filenames)):
    gt_image = Image.open(os.path.join(args.real_path, real_fn)).convert("RGB").resize(image_size)
    gt_image = image_transform(gt_image).cuda().unsqueeze(0)

    pred_image = Image.open(os.path.join(args.fake_path, fake_fn)).convert("RGB").resize(image_size)
    pred_image = image_transform(pred_image).cuda().unsqueeze(0)

    if args.metric == "ms-ssim":
        score = piq.multi_scale_ssim(pred_image, gt_image, data_range=1.0)
    elif args.metric == "psnr":
        score = piq.psnr(pred_image, gt_image, data_range=1.0, reduction="none")
    scores.append(score.item())

mean = torch.mean(torch.tensor(scores)).item()
std = torch.std(torch.tensor(scores)).item()

result = f"{args.metric.upper()} = {mean} +- {std}"
print(result)
with open(args.result_file, "w") as f:
    f.write(result)
