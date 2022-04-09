import argparse


class TrainOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialize()

    def initialize(self):
        self.parser.add_argument("--exp_dir", type=str, help="Path to experiment output directory")
        self.parser.add_argument(
            "--dataset_type", default="ffhq_encode", type=str, help="Type of dataset/experiment to run"
        )
        self.parser.add_argument("--encoder_type", default="LayerWiseEncoder", type=str, help="Which encoder to use")
        self.parser.add_argument("--output_size", default=1024, type=int, help="Output size of generator")

        self.parser.add_argument("--batch_size", default=4, type=int, help="Batch size for training")
        self.parser.add_argument(
            "--batch_size_used_with_adv_loss",
            default=4,
            type=int,
            help="The batch size for training with adversarial loss.",
        )
        self.parser.add_argument("--workers", default=4, type=int, help="Number of train dataloader workers")

        self.parser.add_argument(
            "--encoder_learning_rate", default=0.0001, type=float, help="Encoder optimizer learning rate"
        )
        self.parser.add_argument(
            "--discriminator_learning_rate", default=0.0001, type=float, help="Discriminator optimizer learning rate"
        )
        self.parser.add_argument(
            "--encoder_optim_name", default="adam", type=str, help="Which encoder optimizer to use"
        )
        self.parser.add_argument(
            "--discriminator_optim_name", default="adam", type=str, help="Which discriminator optimizer to use"
        )

        self.parser.add_argument("--target_shape_name", type=str, default="")
        self.parser.add_argument("--lpips_type", type=str, default="alex")

        self.parser.add_argument("--hyper_lpips_lambda", default=0.0, type=float, help="LPIPS loss multiplier factor")
        self.parser.add_argument("--hyper_l2_lambda", default=0.0, type=float, help="L2 loss multiplier factor")
        self.parser.add_argument("--hyper_id_lambda", default=0.0, type=float, help="ID loss multiplier factor")
        self.parser.add_argument(
            "--hyper_adv_lambda", default=0.0, type=float, help="Adversarial loss multiplier factor"
        )
        self.parser.add_argument(
            "--hyper_d_reg_every", type=int, default=0, help="Interval for applying r1 regularization on discriminator"
        )
        self.parser.add_argument("--hyper_d_r1_gamma", type=float, default=0.0, help="Weight of the r1 regularization")
        self.parser.add_argument(
            "--hidden_dim", type=int, default=64, help="The hyper-paramter D described in the main paper"
        )

        self.parser.add_argument("--step_to_add_adversarial_loss", type=int, default=150000)
        self.parser.add_argument(
            "--num_cold_steps",
            type=int,
            default=None,
            help="The period after we add adv loss to the training process, which we do not track the best model, since the loss after we add adv loss increase suddenly leading to tracking wrong best model.",
        )
        self.parser.add_argument("--resume_training", action="store_true", help="Resuming training process or not")
        self.parser.add_argument(
            "--checkpoint_path",
            default=None,
            type=str,
            help="Path to model checkpoint for resuming training or inference",
        )
        self.parser.add_argument(
            "--w_encoder_path",
            default=None,
            type=str,
            help="Path to pretrained w encoder.",
        )
        self.parser.add_argument("--max_steps", default=500000, type=int, help="Maximum number of training steps")
        self.parser.add_argument("--print_interval", default=50, type=int, help="Interval for print metrics")
        self.parser.add_argument("--val_interval", default=1000, type=int, help="Validation interval")
        self.parser.add_argument("--save_interval", default=None, type=int, help="Model checkpoint interval")
        self.parser.add_argument(
            "--save_checkpoint_for_resuming_training",
            action="store_true",
            help="Whether or not to save checkpoint for resuming training process.",
        )
        self.parser.add_argument(
            "--use_wandb", action="store_true", help="Whether to use Weights & Biases to track experiment."
        )

    def parse(self):
        opts = self.parser.parse_args()
        return opts
