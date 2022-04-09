from configs import transforms_config
from configs.paths_config import dataset_paths


DATASETS = {
    "ffhq_encode": {
        "transforms": transforms_config.EncodeTransforms,
        "train_source_root": dataset_paths["ffhq"],
        "train_target_root": dataset_paths["ffhq"],
        "test_source_root": dataset_paths["celeba_test"],
        "test_target_root": dataset_paths["celeba_test"],
    },
    "church_encode": {
        "transforms": transforms_config.EncodeTransforms,
        "train_source_root": dataset_paths["church_train"],
        "train_target_root": dataset_paths["church_train"],
        "test_source_root": dataset_paths["church_test"],
        "test_target_root": dataset_paths["church_test"],
    },
}
