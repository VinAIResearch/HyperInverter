import torch


STYLEGAN2_ADA_CONV_WEIGHT_WITHOUT_BIAS_SHAPES = {
    "b4.conv1.weight": {"shape": torch.Size([512, 512, 3, 3]), "w_idx": 0},
    "b4.torgb.weight": {"shape": torch.Size([3, 512, 1, 1]), "w_idx": 1},
    "b8.conv0.weight": {"shape": torch.Size([512, 512, 3, 3]), "w_idx": 2},
    "b8.conv1.weight": {"shape": torch.Size([512, 512, 3, 3]), "w_idx": 3},
    "b8.torgb.weight": {"shape": torch.Size([3, 512, 1, 1]), "w_idx": 3},
    "b16.conv0.weight": {"shape": torch.Size([512, 512, 3, 3]), "w_idx": 4},
    "b16.conv1.weight": {"shape": torch.Size([512, 512, 3, 3]), "w_idx": 5},
    "b16.torgb.weight": {"shape": torch.Size([3, 512, 1, 1]), "w_idx": 5},
    "b32.conv0.weight": {"shape": torch.Size([512, 512, 3, 3]), "w_idx": 6},
    "b32.conv1.weight": {"shape": torch.Size([512, 512, 3, 3]), "w_idx": 7},
    "b32.torgb.weight": {"shape": torch.Size([3, 512, 1, 1]), "w_idx": 7},
    "b64.conv0.weight": {"shape": torch.Size([512, 512, 3, 3]), "w_idx": 8},
    "b64.conv1.weight": {"shape": torch.Size([512, 512, 3, 3]), "w_idx": 9},
    "b64.torgb.weight": {"shape": torch.Size([3, 512, 1, 1]), "w_idx": 9},
    "b128.conv0.weight": {"shape": torch.Size([256, 512, 3, 3]), "w_idx": 10},
    "b128.conv1.weight": {"shape": torch.Size([256, 256, 3, 3]), "w_idx": 11},
    "b128.torgb.weight": {"shape": torch.Size([3, 256, 1, 1]), "w_idx": 11},
    "b256.conv0.weight": {"shape": torch.Size([128, 256, 3, 3]), "w_idx": 12},
    "b256.conv1.weight": {"shape": torch.Size([128, 128, 3, 3]), "w_idx": 13},
    "b256.torgb.weight": {"shape": torch.Size([3, 128, 1, 1]), "w_idx": 13},
    "b512.conv0.weight": {"shape": torch.Size([64, 128, 3, 3]), "w_idx": 14},
    "b512.conv1.weight": {"shape": torch.Size([64, 64, 3, 3]), "w_idx": 15},
    "b512.torgb.weight": {"shape": torch.Size([3, 64, 1, 1]), "w_idx": 15},
    "b1024.conv0.weight": {"shape": torch.Size([32, 64, 3, 3]), "w_idx": 16},
    "b1024.conv1.weight": {"shape": torch.Size([32, 32, 3, 3]), "w_idx": 17},
    "b1024.torgb.weight": {"shape": torch.Size([3, 32, 1, 1]), "w_idx": 17},
}

STYLEGAN2_ADA_CONV_WEIGHT_WITHOUT_BIAS_WITHOUT_TO_RGB_SHAPES = {
    "b4.conv1.weight": {"shape": torch.Size([512, 512, 3, 3]), "w_idx": 0},
    "b8.conv0.weight": {"shape": torch.Size([512, 512, 3, 3]), "w_idx": 2},
    "b8.conv1.weight": {"shape": torch.Size([512, 512, 3, 3]), "w_idx": 3},
    "b16.conv0.weight": {"shape": torch.Size([512, 512, 3, 3]), "w_idx": 4},
    "b16.conv1.weight": {"shape": torch.Size([512, 512, 3, 3]), "w_idx": 5},
    "b32.conv0.weight": {"shape": torch.Size([512, 512, 3, 3]), "w_idx": 6},
    "b32.conv1.weight": {"shape": torch.Size([512, 512, 3, 3]), "w_idx": 7},
    "b64.conv0.weight": {"shape": torch.Size([512, 512, 3, 3]), "w_idx": 8},
    "b64.conv1.weight": {"shape": torch.Size([512, 512, 3, 3]), "w_idx": 9},
    "b128.conv0.weight": {"shape": torch.Size([256, 512, 3, 3]), "w_idx": 10},
    "b128.conv1.weight": {"shape": torch.Size([256, 256, 3, 3]), "w_idx": 11},
    "b256.conv0.weight": {"shape": torch.Size([128, 256, 3, 3]), "w_idx": 12},
    "b256.conv1.weight": {"shape": torch.Size([128, 128, 3, 3]), "w_idx": 13},
    "b512.conv0.weight": {"shape": torch.Size([64, 128, 3, 3]), "w_idx": 14},
    "b512.conv1.weight": {"shape": torch.Size([64, 64, 3, 3]), "w_idx": 15},
    "b1024.conv0.weight": {"shape": torch.Size([32, 64, 3, 3]), "w_idx": 16},
    "b1024.conv1.weight": {"shape": torch.Size([32, 32, 3, 3]), "w_idx": 17},
}

STYLEGAN2_ADA_ALL_WEIGHT_WITHOUT_BIAS_SHAPES = {
    "b4.conv1.weight": {"shape": torch.Size([512, 512, 3, 3]), "w_idx": 0},
    "b4.torgb.weight": {"shape": torch.Size([3, 512, 1, 1]), "w_idx": 1},
    "b8.conv0.weight": {"shape": torch.Size([512, 512, 3, 3]), "w_idx": 2},
    "b8.conv1.weight": {"shape": torch.Size([512, 512, 3, 3]), "w_idx": 3},
    "b8.torgb.weight": {"shape": torch.Size([3, 512, 1, 1]), "w_idx": 3},
    "b16.conv0.weight": {"shape": torch.Size([512, 512, 3, 3]), "w_idx": 4},
    "b16.conv1.weight": {"shape": torch.Size([512, 512, 3, 3]), "w_idx": 5},
    "b16.torgb.weight": {"shape": torch.Size([3, 512, 1, 1]), "w_idx": 5},
    "b32.conv0.weight": {"shape": torch.Size([512, 512, 3, 3]), "w_idx": 6},
    "b32.conv1.weight": {"shape": torch.Size([512, 512, 3, 3]), "w_idx": 7},
    "b32.torgb.weight": {"shape": torch.Size([3, 512, 1, 1]), "w_idx": 7},
    "b64.conv0.weight": {"shape": torch.Size([512, 512, 3, 3]), "w_idx": 8},
    "b64.conv1.weight": {"shape": torch.Size([512, 512, 3, 3]), "w_idx": 9},
    "b64.torgb.weight": {"shape": torch.Size([3, 512, 1, 1]), "w_idx": 9},
    "b128.conv0.weight": {"shape": torch.Size([256, 512, 3, 3]), "w_idx": 10},
    "b128.conv1.weight": {"shape": torch.Size([256, 256, 3, 3]), "w_idx": 11},
    "b128.torgb.weight": {"shape": torch.Size([3, 256, 1, 1]), "w_idx": 11},
    "b256.conv0.weight": {"shape": torch.Size([128, 256, 3, 3]), "w_idx": 12},
    "b256.conv1.weight": {"shape": torch.Size([128, 128, 3, 3]), "w_idx": 13},
    "b256.torgb.weight": {"shape": torch.Size([3, 128, 1, 1]), "w_idx": 13},
    "b512.conv0.weight": {"shape": torch.Size([64, 128, 3, 3]), "w_idx": 14},
    "b512.conv1.weight": {"shape": torch.Size([64, 64, 3, 3]), "w_idx": 15},
    "b512.torgb.weight": {"shape": torch.Size([3, 64, 1, 1]), "w_idx": 15},
    "b1024.conv0.weight": {"shape": torch.Size([32, 64, 3, 3]), "w_idx": 16},
    "b1024.conv1.weight": {"shape": torch.Size([32, 32, 3, 3]), "w_idx": 17},
    "b1024.torgb.weight": {"shape": torch.Size([3, 32, 1, 1]), "w_idx": 17},
    # affine
    "b4.conv1.affine.weight": {"shape": torch.Size([512, 512]), "w_idx": 0},
    "b4.torgb.affine.weight": {"shape": torch.Size([512, 512]), "w_idx": 1},
    "b8.conv0.affine.weight": {"shape": torch.Size([512, 512]), "w_idx": 2},
    "b8.conv1.affine.weight": {"shape": torch.Size([512, 512]), "w_idx": 3},
    "b8.torgb.affine.weight": {"shape": torch.Size([512, 512]), "w_idx": 3},
    "b16.conv0.affine.weight": {"shape": torch.Size([512, 512]), "w_idx": 4},
    "b16.conv1.affine.weight": {"shape": torch.Size([512, 512]), "w_idx": 5},
    "b16.torgb.affine.weight": {"shape": torch.Size([512, 512]), "w_idx": 5},
    "b32.conv0.affine.weight": {"shape": torch.Size([512, 512]), "w_idx": 6},
    "b32.conv1.affine.weight": {"shape": torch.Size([512, 512]), "w_idx": 7},
    "b32.torgb.affine.weight": {"shape": torch.Size([512, 512]), "w_idx": 7},
    "b64.conv0.affine.weight": {"shape": torch.Size([512, 512]), "w_idx": 8},
    "b64.conv1.affine.weight": {"shape": torch.Size([512, 512]), "w_idx": 9},
    "b64.torgb.affine.weight": {"shape": torch.Size([512, 512]), "w_idx": 9},
    "b128.conv0.affine.weight": {"shape": torch.Size([512, 512]), "w_idx": 10},
    "b128.conv1.affine.weight": {"shape": torch.Size([256, 512]), "w_idx": 11},
    "b128.torgb.affine.weight": {"shape": torch.Size([256, 512]), "w_idx": 11},
    "b256.conv0.affine.weight": {"shape": torch.Size([256, 512]), "w_idx": 12},
    "b256.conv1.affine.weight": {"shape": torch.Size([128, 512]), "w_idx": 13},
    "b256.torgb.affine.weight": {"shape": torch.Size([128, 512]), "w_idx": 13},
    "b512.conv0.affine.weight": {"shape": torch.Size([128, 512]), "w_idx": 14},
    "b512.conv1.affine.weight": {"shape": torch.Size([64, 512]), "w_idx": 15},
    "b512.torgb.affine.weight": {"shape": torch.Size([64, 512]), "w_idx": 15},
    "b1024.conv0.affine.weight": {"shape": torch.Size([64, 512]), "w_idx": 16},
    "b1024.conv1.affine.weight": {"shape": torch.Size([32, 512]), "w_idx": 17},
    "b1024.torgb.affine.weight": {"shape": torch.Size([32, 512]), "w_idx": 17},
}
