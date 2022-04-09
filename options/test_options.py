from argparse import ArgumentParser


class TestOptions:
    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self):
        # arguments for inference script
        self.parser.add_argument("--exp_dir", type=str, help="Path to experiment output directory")
        self.parser.add_argument(
            "--checkpoint_path", default=None, type=str, help="Path to HyperInverter model checkpoint"
        )
        self.parser.add_argument(
            "--data_path", type=str, default="gt_images", help="Path to directory of images to evaluate"
        )
        self.parser.add_argument(
            "--couple_outputs", action="store_true", help="Whether to also save inputs + outputs side-by-side"
        )
        self.parser.add_argument(
            "--n_images", type=int, default=None, help="Number of images to output. If None, run on all data"
        )
        self.parser.add_argument("--batch_size", default=2, type=int, help="Batch size for testing and inference")
        self.parser.add_argument("--workers", default=2, type=int, help="Number of test/inference dataloader workers")

    def parse(self):
        opts = self.parser.parse_args()
        return opts
