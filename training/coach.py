import os
import random
from shutil import copyfile

import numpy as np
import torch
import torch.nn.functional as F
from configs import data_configs
from criteria import id_loss, l2_loss, moco_loss
from criteria.lpips.lpips import LPIPS
from datasets.images_dataset import ImagesDataset
from models.hyper_inverter import HyperInverter
from torch.utils.data import DataLoader
from training.ranger import Ranger
from utils import train_utils
from utils.common import count_parameters, toogle_grad


class Coach:
    def __init__(self, opts):
        self.opts = opts
        self.device = "cuda:0"
        self.opts.device = self.device
        torch.backends.cudnn.benchmark = True
        self.global_step = 0
        self.best_val_loss = None

        # Fix random seed
        SEED = 2107
        random.seed(SEED)
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        torch.cuda.manual_seed_all(SEED)

        if self.opts.use_wandb:
            from utils.wandb_utils import WBLogger

            self.wb_logger = WBLogger(self.opts)

        # Initialize network
        self.net = HyperInverter(self.opts).to(self.device)

        # Log some information about network on terminal
        print("Number of parameters:")
        print("==> W Bar Encoder: ", count_parameters(self.net.w_bar_encoder))
        print("==> Hypernetwork: ", count_parameters(self.net.hypernet))
        print("==> Generator: ", count_parameters(self.net.decoder[0]))
        print("==> Discriminator: ", count_parameters(self.net.discriminator))
        print("StyleGAN layers to predict: ", self.net.target_shape.keys())
        print(
            "Number of parameter's weights in StyleGAN generator to predict: ", self.net.hypernet.num_predicted_weights
        )

        # Estimate latent_avg via dense sampling if latent_avg is not available
        if self.net.latent_avg is None:
            # Get mean latent by sampling
            z_samples = torch.randn(100000, self.net.decoder[0].z_dim, device=self.device)
            self.net.latent_avg = (
                self.net.decoder[0].mapping(z_samples, None)[:, :1, :].mean(0, keepdim=True).squeeze(0)
            )

        # Initialize loss
        if self.opts.hyper_lpips_lambda > 0:
            self.lpips_loss = LPIPS(net_type=self.opts.lpips_type).to(self.opts.device).eval()

        if self.opts.hyper_id_lambda > 0:
            if "ffhq" in self.opts.dataset_type or "celeb" in self.opts.dataset_type:
                self.id_loss = id_loss.IDLoss().to(self.device).eval()
            else:
                self.id_loss = moco_loss.MocoLoss(opts).to(self.device).eval()

        # Initialize optimizer
        self.encoder_optimizer = self.configure_encoder_optimizers()
        self.discriminator_optimizer = self.configure_discriminator_optimizers()

        # Resume training process from checkpoint path
        if self.opts.resume_training:
            ckpt = torch.load(self.opts.checkpoint_path, map_location="cpu")

            print("Load encoder optimizer from checkpoint")
            self.encoder_optimizer.load_state_dict(ckpt["encoder_optimizer"])

            if "discriminator_optimizer" in ckpt:
                print("Load discriminator optimizer from checkpoint")
                self.discriminator_optimizer.load_state_dict(ckpt["discriminator_optimizer"])

            # Resuming the global step and best val loss from checkpoint
            if "loss_dict" in ckpt:
                self.global_step = ckpt["loss_dict"]["global_step"]
                self.best_val_loss = ckpt["loss_dict"]["best_val_loss"]
                print(f"Resuming training process from step {self.global_step}")
                print(f"Current best val loss: {self.best_val_loss }")

        # Debugging purpose
        if self.global_step > self.opts.step_to_add_adversarial_loss:
            self.is_copy_best_version_non_adv = True
        else:
            self.is_copy_best_version_non_adv = False

        # Initialize dataset
        self.train_dataset, self.test_dataset = self.configure_datasets()
        self.configure_dataloaders(self.opts.batch_size)

        # Config num_cold_steps if has not been set.
        if self.opts.num_cold_steps is None:
            self.opts.num_cold_steps = int(self.opts.step_to_add_adversarial_loss / 10)

        # Initialize checkpoint dir
        self.checkpoint_dir = os.path.join(opts.exp_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        if self.opts.save_interval is None:
            self.opts.save_interval = self.opts.max_steps

    def toogle_mode(self, mode="train"):
        if mode == "train":
            self.net.w_bar_encoder.train()
            self.net.hypernet.train()
            self.net.discriminator.train()
        else:
            self.net.w_bar_encoder.eval()
            self.net.hypernet.eval()
            self.net.discriminator.eval()

    def train(self):
        self.toogle_mode("train")
        while self.global_step < self.opts.max_steps:
            for batch_idx, batch in enumerate(self.train_dataloader):
                # Generate images
                x, y = batch  # x: input image, # y: target image
                x, y = x.to(self.device).float(), y.to(self.device).float()
                w_y_hat, y_hat, predicted_weights = self.net.forward(x, return_latents=False)

                # Check the condition to use adversarial loss
                self.use_adv_loss = self.global_step >= self.opts.step_to_add_adversarial_loss

                # Log for debug
                if self.global_step == self.opts.step_to_add_adversarial_loss:
                    print(f"Start to train with adversarial loss at step {self.global_step}")

                # ===== Update G  ============
                g_loss, g_loss_dict, id_logs = self.calc_encoder_loss(y_hat, y, w_y_hat)
                self.encoder_optimizer.zero_grad()
                g_loss.backward()
                self.encoder_optimizer.step()
                loss_dict = g_loss_dict

                # For debugging purpose
                if (
                    not self.is_copy_best_version_non_adv
                    and self.global_step == self.opts.step_to_add_adversarial_loss
                ):
                    if os.path.exists(os.path.join(self.checkpoint_dir, "best_model.pt")):
                        copyfile(
                            os.path.join(self.checkpoint_dir, "best_model.pt"),
                            os.path.join(self.checkpoint_dir, f"non_adv_best_model_iter_{self.global_step}.pt"),
                        )
                    self.is_copy_best_version_non_adv = True

                    # Change batch size when we train with adv loss
                    self.configure_dataloaders(batch_size=self.opts.batch_size_used_with_adv_loss)
                    break

                if self.use_adv_loss:
                    # ===== Update D ============
                    toogle_grad(self.net.discriminator, True)
                    d_loss, d_loss_dict = self.calc_discriminator_loss(y_hat.detach(), y)
                    self.discriminator_optimizer.zero_grad()
                    d_loss.backward()
                    self.discriminator_optimizer.step()
                    loss_dict = {**g_loss_dict, **d_loss_dict}

                    # R1 Regularization
                    if self.opts.hyper_d_r1_gamma > 0:
                        if self.global_step % self.opts.hyper_d_reg_every == 0:
                            d_r1_loss, d_r1_loss_dict = self.calc_discriminator_r1_loss(y)
                            self.discriminator_optimizer.zero_grad()
                            d_r1_loss.backward()
                            self.discriminator_optimizer.step()
                            loss_dict = {**loss_dict, **d_r1_loss_dict}

                    toogle_grad(self.net.discriminator, False)

                self.net.latent_avg = self.net.latent_avg.detach()

                # Logging related
                if self.global_step % self.opts.print_interval == 0:
                    self.print_metrics(loss_dict, prefix="train")
                    self.log_metrics(loss_dict, prefix="train")

                # Log images of first batch to wandb
                if self.opts.use_wandb and self.global_step % self.opts.val_interval == 0 and batch_idx == 0:
                    with torch.no_grad():
                        w_y_hat = self.net.face_pool(w_y_hat)
                        y_hat = self.net.face_pool(y_hat)
                        y = self.net.face_pool(y)
                    self.wb_logger.log_images_to_wandb(
                        y, w_y_hat, y_hat, id_logs, prefix="train", step=self.global_step, opts=self.opts
                    )

                # Validation related
                val_loss_dict = None
                if self.global_step % self.opts.val_interval == 0 or self.global_step == self.opts.max_steps:
                    val_loss_dict = self.validate()

                    # Don't log the best model if the current step is in the cold period, to avoid log wrong the best model.
                    if self.global_step not in [
                        self.opts.step_to_add_adversarial_loss,
                        self.opts.step_to_add_adversarial_loss + self.opts.num_cold_steps,
                    ]:
                        if val_loss_dict and (
                            self.best_val_loss is None or val_loss_dict["loss"] < self.best_val_loss
                        ):
                            self.best_val_loss = val_loss_dict["loss"]
                            self.checkpoint_me(val_loss_dict, is_best=True)
                    else:
                        self.best_val_loss = None  # Restart best val loss again when adding adv loss

                if self.global_step % self.opts.save_interval == 0 or self.global_step == self.opts.max_steps:
                    if val_loss_dict is not None:
                        self.checkpoint_me(val_loss_dict, is_best=False)
                    else:
                        self.checkpoint_me(loss_dict, is_best=False)

                if self.global_step == self.opts.max_steps:
                    print("OMG, finished training!")
                    break

                self.global_step += 1

    def validate(self):
        self.toogle_mode("eval")
        agg_loss_dict = []
        for batch_idx, batch in enumerate(self.test_dataloader):
            x, y = batch

            with torch.no_grad():
                x, y = x.to(self.device).float(), y.to(self.device).float()
                w_y_hat, y_hat, predicted_weights = self.net.forward(x, return_latents=False)
                # Calc loss between final images and input images
                loss, cur_loss_dict, id_logs = self.calc_encoder_loss(y_hat, y, w_y_hat)

            agg_loss_dict.append(cur_loss_dict)

            # Log images of first batch to wandb
            if self.opts.use_wandb and batch_idx == 0:
                with torch.no_grad():
                    w_y_hat = self.net.face_pool(w_y_hat)
                    y_hat = self.net.face_pool(y_hat)
                    y = self.net.face_pool(y)
                self.wb_logger.log_images_to_wandb(
                    y, w_y_hat, y_hat, id_logs, prefix="test", step=self.global_step, opts=self.opts
                )

            # For first step just do sanity test on small amount of data
            if self.global_step == 0 and batch_idx >= 4:
                self.toogle_mode("train")
                return None  # Do not log, inaccurate in first batch

        loss_dict = train_utils.aggregate_loss_dict(agg_loss_dict)
        self.print_metrics(loss_dict, prefix="test")
        self.log_metrics(loss_dict, prefix="test")

        self.toogle_mode("train")
        return loss_dict

    def checkpoint_me(self, loss_dict, is_best):
        save_name = "best_model.pt" if is_best else f"iteration_{self.global_step}.pt"

        # Tracking current best val loss and global step for resume training process
        loss_dict["best_val_loss"] = self.best_val_loss
        loss_dict["global_step"] = self.global_step

        save_dict = self.__get_save_dict(loss_dict)
        checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
        torch.save(save_dict, checkpoint_path)

        # Remove the previous checkpoint
        previous_checkpoint_name = f"iteration_{self.global_step - self.opts.save_interval}.pt"
        if os.path.exists(os.path.join(self.checkpoint_dir, previous_checkpoint_name)) and not is_best:
            os.remove(os.path.join(self.checkpoint_dir, previous_checkpoint_name))

        with open(os.path.join(self.checkpoint_dir, "timestamp.txt"), "a") as f:
            if is_best:
                f.write(f"**Best**: Step - {self.global_step}, Loss - {self.best_val_loss} \n{loss_dict}\n")
                if self.opts.use_wandb:
                    self.wb_logger.log_best_model()
            else:
                f.write(f"Step - {self.global_step}, \n{loss_dict}\n")

    def configure_encoder_optimizers(self):
        params = list(self.net.w_bar_encoder.parameters())
        params += list(self.net.hypernet.parameters())
        if self.opts.encoder_optim_name == "adam":
            optimizer = torch.optim.Adam(params, lr=self.opts.encoder_learning_rate)
        elif self.opts.encoder_optim_name == "ranger":
            optimizer = Ranger(params, lr=self.opts.encoder_learning_rate)
        else:
            raise Exception(f"{self.opts.encoder_optim_name} optimizer is not defined.")
        return optimizer

    def configure_discriminator_optimizers(self):
        params = list(self.net.discriminator.parameters())
        if self.opts.discriminator_optim_name == "adam":
            optimizer = torch.optim.Adam(params, lr=self.opts.discriminator_learning_rate)
        elif self.opts.discriminator_optim_name == "ranger":
            optimizer = Ranger(params, lr=self.opts.discriminator_learning_rate)
        else:
            raise Exception(f"{self.opts.discriminator_optim_name} optimizer is not defined.")
        return optimizer

    def configure_datasets(self):
        if self.opts.dataset_type not in data_configs.DATASETS.keys():
            Exception(f"{self.opts.dataset_type} is not a valid dataset_type")
        print(f"Loading dataset for {self.opts.dataset_type}")
        dataset_args = data_configs.DATASETS[self.opts.dataset_type]
        transforms_dict = dataset_args["transforms"](self.opts).get_transforms()
        train_dataset = ImagesDataset(
            source_root=dataset_args["train_source_root"],
            target_root=dataset_args["train_target_root"],
            source_transform=transforms_dict["transform_source"],
            target_transform=transforms_dict["transform_gt_train"],
            opts=self.opts,
        )
        test_dataset = ImagesDataset(
            source_root=dataset_args["test_source_root"],
            target_root=dataset_args["test_target_root"],
            source_transform=transforms_dict["transform_source"],
            target_transform=transforms_dict["transform_test"],
            opts=self.opts,
        )
        if self.opts.use_wandb:
            self.wb_logger.log_dataset_wandb(train_dataset, dataset_name="Train")
            self.wb_logger.log_dataset_wandb(test_dataset, dataset_name="Test")
        print(f"Number of training samples: {len(train_dataset)}")
        print(f"Number of test samples: {len(test_dataset)}")
        return train_dataset, test_dataset

    def g_nonsaturating_loss(self, fake_preds):
        loss = F.softplus(-fake_preds).mean()
        return loss

    def configure_dataloaders(self, batch_size):
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=int(self.opts.workers),
            drop_last=True,
        )
        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=int(self.opts.workers),
            drop_last=True,
        )

    def calc_encoder_loss(self, generated_images, real_images, w_images=None):
        loss_dict = {}
        loss = 0.0
        id_logs = None

        # Adversarial loss
        if self.use_adv_loss:
            # Loss G
            fake_preds = self.net.discriminator(generated_images, None)
            loss_G_adv = self.g_nonsaturating_loss(fake_preds)
            loss_dict["loss_G_adv"] = float(loss_G_adv)
            loss += loss_G_adv * self.opts.hyper_adv_lambda

        # Resize to 256 x 256 to calculate for the remaining loss.
        w_images = self.net.face_pool(w_images)
        generated_images = self.net.face_pool(generated_images)
        real_images = self.net.face_pool(real_images)

        # L2 loss
        if self.opts.hyper_l2_lambda > 0:
            loss_l2 = l2_loss.l2_loss(generated_images, real_images)
            loss_dict["loss_l2"] = float(loss_l2)
            loss += loss_l2 * self.opts.hyper_l2_lambda

        # LPIPS loss
        if self.opts.hyper_lpips_lambda > 0:
            loss_lpips = self.lpips_loss(generated_images, real_images)
            loss_dict["loss_lpips"] = float(loss_lpips)
            loss += loss_lpips * self.opts.hyper_lpips_lambda

        # ID loss
        if self.opts.hyper_id_lambda > 0:
            loss_id, sim_improvement, id_logs = self.id_loss(generated_images, real_images, real_images)
            loss_dict["loss_id"] = float(loss_id)
            loss_dict["id_improve"] = float(sim_improvement)
            loss += loss_id * self.opts.hyper_id_lambda

        loss_dict["loss"] = float(loss)

        return loss, loss_dict, id_logs

    def d_logistic_loss(self, real_preds, fake_preds):
        real_loss = F.softplus(-real_preds)
        fake_loss = F.softplus(fake_preds)

        return (real_loss.mean() + fake_loss.mean()) / 2

    def calc_discriminator_loss(self, generated_images, real_images):
        loss_dict = {}
        fake_preds = self.net.discriminator(generated_images, None)
        real_preds = self.net.discriminator(real_images, None)
        loss = self.d_logistic_loss(real_preds, fake_preds)
        loss_dict["loss_D"] = float(loss)
        return loss, loss_dict

    def d_r1_loss(self, real_pred, real_img):
        (grad_real,) = torch.autograd.grad(outputs=real_pred.sum(), inputs=real_img, create_graph=True)
        grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

        return grad_penalty

    def calc_discriminator_r1_loss(self, real_images):
        loss_dict = {}
        real_images.requires_grad = True
        real_preds = self.net.discriminator(real_images, None)
        real_preds = real_preds.view(real_images.size(0), -1)
        real_preds = real_preds.mean(dim=1).unsqueeze(1)
        r1_loss = self.d_r1_loss(real_preds, real_images)
        loss_D_R1 = self.opts.hyper_d_r1_gamma / 2 * r1_loss * self.opts.hyper_d_reg_every + 0 * real_preds[0]
        loss_dict["loss_D_r1_reg"] = float(loss_D_R1)
        return loss_D_R1, loss_dict

    def log_metrics(self, metrics_dict, prefix):
        if self.opts.use_wandb:
            self.wb_logger.log(prefix, metrics_dict, self.global_step)

    def print_metrics(self, metrics_dict, prefix):
        print(f"Metrics for {prefix}, step {self.global_step}")
        for key, value in metrics_dict.items():
            print(f"\t{key} = ", value)

    def __get_save_dict(self, loss_dict):
        save_dict = {"state_dict": self.net.state_dict(), "opts": vars(self.opts)}

        if self.opts.save_checkpoint_for_resuming_training:
            save_dict["encoder_optimizer"] = self.encoder_optimizer.state_dict()
            save_dict["discriminator_optimizer"] = self.discriminator_optimizer.state_dict()
            save_dict["loss_dict"] = loss_dict

        return save_dict
