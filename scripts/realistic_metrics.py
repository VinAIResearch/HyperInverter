import argparse

import torch_fidelity


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=50, help="Batch size to use")
    parser.add_argument("--fake_path", type=str, default="")
    parser.add_argument("--real_path", type=str, default="")
    parser.add_argument("--kid_subset_size", type=int, default=1000)
    parser.add_argument("--fid_saved_result_file", type=str, default="")
    parser.add_argument("--kid_saved_result_file", type=str, default="")
    return parser.parse_args()


args = parse_args()

# Calc metrics
metric_scores_dict = torch_fidelity.calculate_metrics(
    input1=args.fake_path,
    input2=args.real_path,
    cuda=True,
    batch_size=args.batch_size,
    fid=True,
    kid=True,
    verbose=False,
    kid_subset_size=args.kid_subset_size,
)

fid_score = metric_scores_dict["frechet_inception_distance"]
kid_mean = metric_scores_dict["kernel_inception_distance_mean"] * 1e3
kid_std = metric_scores_dict["kernel_inception_distance_std"] * 1e3

# Log on terminal
print(f"FID = {fid_score}")
print(f"KID (x 10^3) = {kid_mean} +- {kid_std}")

# Save to file
with open(args.fid_saved_result_file, "w") as f:
    f.write(f"FID = {fid_score}")

with open(args.kid_saved_result_file, "w") as f:
    f.write(f"KID (x 10^3) = {kid_mean} +- {kid_std}")
