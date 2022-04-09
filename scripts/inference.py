import argparse
import os
import sys
import time


sys.path.append(".")
sys.path.append("..")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from configs import data_configs  # noqa: E402
from datasets.inference_dataset import InferenceDataset  # noqa: E402
from models.hyper_inverter import HyperInverter  # noqa: E402
from options.test_options import TestOptions  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from tqdm import tqdm  # noqa: E402
from utils.common import log_input_image, tensor2im  # noqa: E402


def run():
    test_opts = TestOptions().parse()
    out_path_results = os.path.join(test_opts.exp_dir, "inference_results")
    out_path_coupled = os.path.join(test_opts.exp_dir, "inference_coupled")

    os.makedirs(out_path_results, exist_ok=True)
    os.makedirs(out_path_coupled, exist_ok=True)

    ckpt = torch.load(test_opts.checkpoint_path, map_location="cpu")
    opts = ckpt["opts"]
    opts.update(vars(test_opts))
    opts = argparse.Namespace(**opts)
    net = HyperInverter(opts)
    net.eval()
    net.cuda()

    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args["transforms"](opts).get_transforms()
    dataset = InferenceDataset(root=opts.data_path, transform=transforms_dict["transform_inference"], opts=opts)
    dataloader = DataLoader(
        dataset, batch_size=opts.batch_size, shuffle=False, num_workers=int(opts.workers), drop_last=False
    )

    if opts.n_images is None:
        opts.n_images = len(dataset)

    global_i = 0
    global_time = 0
    for input_batch in tqdm(dataloader):
        if global_i >= opts.n_images:
            break
        with torch.no_grad():
            input_cuda = input_batch.cuda().float()
            tic = time.time()
            w_images, final_images, predicted_weights = run_on_batch(input_cuda, net)
            toc = time.time()
            global_time += toc - tic

        bs = final_images.size(0)
        for i in range(bs):
            final_image = tensor2im(final_images[i])
            w_image = tensor2im(w_images[i])

            im_path = dataset.paths[global_i]

            if opts.couple_outputs or global_i % 100 == 0:
                input_im = log_input_image(input_batch[i])
                res = np.concatenate([np.array(input_im), np.array(w_image), np.array(final_image)], axis=1)
                Image.fromarray(res).save(os.path.join(out_path_coupled, os.path.basename(im_path)))

            im_save_path = os.path.join(out_path_results, os.path.basename(im_path))
            Image.fromarray(np.array(final_image)).save(im_save_path)

            global_i += 1

    stats_path = os.path.join(opts.exp_dir, "stats.txt")
    result_str = "Runtime {:.4f}".format(global_time / len(dataset))
    print(result_str)

    with open(stats_path, "w") as f:
        f.write(result_str)


def run_on_batch(inputs, net):
    result_batch = net(inputs, return_latents=False)
    return result_batch


if __name__ == "__main__":
    run()
