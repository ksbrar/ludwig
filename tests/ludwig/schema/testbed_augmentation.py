import logging
import os

# import pprint
import shutil

# import sys
import warnings

import pandas as pd

from ludwig.api import LudwigModel
from ludwig.data.dataset_synthesizer import cli_synthesize_dataset

# from ludwig.encoders.registry import get_encoder_cls, get_encoder_registry

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
FEATURES_LIST = [
    {"name": "binary_output_feature", "type": "binary"},
    {"name": "category_output_feature", "type": "category"},
    {
        "name": "my_image",
        "type": "image",
        "destination_folder": os.path.join(os.getcwd(), "data2/images"),
        "preprocessing": {"height": 600, "width": 600, "num_channels": 3},
    },
]
shutil.rmtree("data2", ignore_errors=True)
os.makedirs("data2", exist_ok=True)
cli_synthesize_dataset(50, FEATURES_LIST, "data2/syn_train.csv")

config = {
    "input_features": [
        {"name": "category_output_feature", "type": "category"},
        {
            "name": "my_image",
            "type": "image",
            "augmentation": False,
            # "augmentation": True,
            # "augmentation": [
            #     {"type": "random_vertical_flip"},
            #     {"type": "random_rotate", "degree": 45},
            #     {"type": "random_blur", "kernel_size": 9},
            #     {"type": "random_contrast"},
            # ],
            "encoder": {
                # "type": "stacked_cnn",
                "type": "alexnet",
                "model_cache_dir": os.path.join(os.getcwd(), "tv_cache"),
            },
            "preprocessing": {"standardize_image": "imagenet1k", "width": 300, "height": 300},
        },
    ],
    "output_features": [
        {
            "name": "category_output_feature",
            "type": "category",
        }
    ],
    "trainer": {"epochs": 2, "batch_size": 7},
    "backend": {"type": "local"},
}
model = LudwigModel(config, logging_level=logging.INFO)
_, _, output_dir = model.train(dataset="data2/syn_train.csv", skip_save_processed_input=True)
print(f"encoder.input_shape: {model.model.input_features['my_image'].encoder_obj.input_shape}")
model.save("./saved_model")
print("after training")
print(">>>>before prediction")
model2 = LudwigModel.load(os.path.join(output_dir, "model"), logging_level=logging.INFO)
predictions = model2.predict(dataset="data2/syn_train.csv")
print(">>>> after prediction")
print(predictions[0].shape)
print(predictions[0].columns)
print(predictions[0].head())
print("all done.")
