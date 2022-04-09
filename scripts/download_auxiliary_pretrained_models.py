import os
import zipfile

base_dir = "pretrained_models"

os.makedirs(base_dir, exist_ok=True)

google_drive_paths = {
    f"{base_dir}/shape_predictor_68_face_landmarks.dat": "https://drive.google.com/uc?id=1QMTvn2VJ9k3TYqbXBSiNgDxYz9T69udn",
    f"{base_dir}/CurricularFace_Backbone.pth": "https://drive.google.com/uc?id=1LMtF52CcjuhQffWUwzg17mbYl5oh_1mZ",
    f"{base_dir}/moco_v2_800ep_pretrain.pt": "https://drive.google.com/uc?id=10NBxMthLNigYXyTJwxd_0-1qCczizBnY",
    f"{base_dir}/model_ir_se50.pth": "https://drive.google.com/uc?id=1GPge9s_Jf3Xz6d4SeIP7_dVGw907epUP",
    f"{base_dir}/resnet34-333f7ec4.pth": "https://drive.google.com/uc?id=1JgbU4ztY66-U_bb6b3AXDVKr9waPJIGG",
    f"{base_dir}/stylegan2-church-config-f.pkl": "https://drive.google.com/uc?id=1fJUwO39r91YyJCtMobT75jWNcpDkBBY5",
    f"{base_dir}/stylegan2-ffhq-config-f.pkl": "https://drive.google.com/uc?id=1j6ekeBhMOnBRl2qEFlmfPUqrCNm-FkKh",
    f"{base_dir}/mtcnn.zip": "https://drive.google.com/uc?id=1Xv85mZrtOxVU381DjlsIjIcLCMlWSnL_",
}

for model_weights_filename in google_drive_paths:
    if not os.path.isfile(model_weights_filename):
        gdrive_url = google_drive_paths[model_weights_filename]
        try:
            from gdown import download as drive_download

            drive_download(gdrive_url, model_weights_filename, quiet=False)
        except ModuleNotFoundError:
            print(
                "gdown module not found.", "pip install gdown or, manually download the checkpoint file:", gdrive_url
            )

    if not os.path.isfile(model_weights_filename):
        print(model_weights_filename, " not found, you may need to manually download the model weights.")
        
    if model_weights_filename == f"{base_dir}/mtcnn.zip":
        with zipfile.ZipFile(model_weights_filename, 'r') as zip_ref:
            zip_ref.extractall(base_dir)
        

