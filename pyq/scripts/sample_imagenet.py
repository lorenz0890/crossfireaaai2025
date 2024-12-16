import os
import warnings
from pathlib import Path
import shutil

import argparse
from tqdm.notebook import tqdm

TRAIN_PATH_NAME = "/train/"
VAL_PATH_NAME = "/val/"


def sample_imagenet(data_path, ratio):
    for path in tqdm(os.listdir(data_path)):
        class_path = os.path.join(data_path, path)
        images_names = os.listdir(class_path)
        images_names.sort()
        number_of_images = len(images_names)
        remove_images_names = images_names[: int(number_of_images * (1 - ratio))]

        if len(remove_images_names) == number_of_images:
            warnings.warn("{} will have no classes and all images in it will removed.".format(path))

        # iterate over the list to remove each image
        for remove_image_name in remove_images_names:
            remove_image_path = os.path.join(class_path, remove_image_name)
            new_path = class_path.replace("train", "removed_train").replace("val", "removed_val")
            Path(new_path).mkdir(parents=True, exist_ok=True)
            shutil.move(remove_image_path, os.path.join(new_path, remove_image_name))


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser(description="PyQ")
    args_parser.add_argument("--imagenet_path", type=str, required=True)
    args_parser.add_argument("--ratio", type=float, required=False)
    args = args_parser.parse_args()

    ratio = args.ratio if hasattr(args, "ratio") and args.ratio else 0.1

    imagenet_path = args.imagenet_path
    imagenet_training_path = imagenet_path + TRAIN_PATH_NAME
    imagenet_validation_path = imagenet_path + VAL_PATH_NAME

    sample_imagenet(imagenet_training_path, ratio)
    sample_imagenet(imagenet_validation_path, ratio)
