import cv2
import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor
from typing import Dict
import os
import torch


def get_masked_image(image: str, masks: Tensor) -> Tensor:
    """
    Get masked image from image and masks

    Args:
        image: The path to the image.
        masks: The masks of the image.

    Returns:
        The masked image.
    """

    image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    image = torch.from_numpy(image)

    # (num_objects,num_masks_per_object,height,width)
    merged_mask = torch.max(masks, dim=0).values[0]
    merged_mask = merged_mask.unsqueeze(2)

    masked_image = image * merged_mask
    return masked_image


def get_masked_images(dataset: str, masks: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """
    Get masked images from dataset and masks

    Args:
        dataset: The path to the dataset.
        masks: The masks of the dataset.

    Returns:
        The masked images.
    """

    masked_images = {}
    if not os.path.isdir(dataset):
        image = os.path.basename(dataset)
        masked_images[image] = get_masked_image(dataset, masks[image])
    else:
        for image in os.listdir(dataset):
            masked_images[image] = get_masked_image(os.path.join(dataset, image), masks[image])
    return masked_images
