# Configurations
### Set to folder saving results
EXPERIMENT_DIR=""
### Set the path to model path 
MODEL_PATH=""
### Set GPU ID 
GPU_ID=0
### Set path to TEST DATA folder
DATA_PATH='data/lsun_church/val'
### Set path to TEST DATA folder that test images are resized to 256 x 256 for computing FID, KID metrics.
RESIZED_DATA_PATH='data/lsun_church/val_256'

#======================================

# Inference on test set 
CUDA_VISIBLE_DEVICES="$GPU_ID" \
python scripts/inference.py \
--exp_dir="$EXPERIMENT_DIR" \
--checkpoint_path="$MODEL_PATH" \
--data_path="$DATA_PATH" \
--batch_size=4 \
--workers=4

# LPIPS 
CUDA_VISIBLE_DEVICES="$GPU_ID" \
python scripts/calc_losses_on_images.py \
--mode lpips \
--data_path="$EXPERIMENT_DIR"/inference_results \
--gt_path="$DATA_PATH"

# L2 
CUDA_VISIBLE_DEVICES="$GPU_ID" \
python scripts/calc_losses_on_images.py \
--mode l2 \
--data_path="$EXPERIMENT_DIR"/inference_results \
--gt_path="$DATA_PATH" 

# FID and KID
PATH_1="$RESIZED_DATA_PATH"
PATH_2="$EXPERIMENT_DIR"/inference_results
FID_SAVED_RESULT_FILE="$EXPERIMENT_DIR"/inference_metrics/stat_fid.txt 
KID_SAVED_RESULT_FILE="$EXPERIMENT_DIR"/inference_metrics/stat_kid.txt 
CUDA_VISIBLE_DEVICES="$GPU_ID" \
python scripts/realistic_metrics.py  \
--batch_size 50 \
--kid_subset_size 100 \
--real_path "$PATH_1" \
--fake_path "$PATH_2" \
--fid_saved_result_file "$FID_SAVED_RESULT_FILE" \
--kid_saved_result_file "$KID_SAVED_RESULT_FILE"

# PSNR 
CUDA_VISIBLE_DEVICES="$GPU_ID" \
python scripts/other_metrics.py \
--metric=psnr \
--fake_path="$EXPERIMENT_DIR"/inference_results \
--real_path="$DATA_PATH" \
--image_height=256 \
--image_width=256 \
--result_file="$EXPERIMENT_DIR"/inference_metrics/stat_psnr.txt

# MS-SSIM
CUDA_VISIBLE_DEVICES="$GPU_ID" \
python scripts/other_metrics.py \
--metric=ms-ssim \
--fake_path="$EXPERIMENT_DIR"/inference_results \
--real_path="$DATA_PATH" \
--image_height=256 \
--image_width=256 \
--result_file="$EXPERIMENT_DIR"/inference_metrics/stat_ms_ssim.txt