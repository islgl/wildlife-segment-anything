import cv2
from PIL.Image import fromarray
from PIL import Image
from torch import Tensor
from typing import Dict
import os
import torch


def get_masked_image(image: str, masks: Tensor) -> Image:
    """
    Get masked image from image and masks

    Args:
        image: The path to the image.
        masks: The masks of the image.

    Returns:
        The masked image.
    """

    image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(image)

    # (num_objects,num_masks_per_object,height,width)
    merged_mask = torch.max(masks, dim=0).values[0]
    merged_mask = merged_mask.unsqueeze(2)

    if torch.cuda.is_available():
        image = image.cuda()
        merged_mask = merged_mask.cuda()

    masked_image = image * merged_mask
    masked_image = masked_image.cpu()
    masked_image = masked_image.numpy()
    masked_image = fromarray(masked_image)

    return masked_image


def get_masked_images(dataset: str, masks: Dict) -> Dict:
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
            try:
                masked_images[image] = get_masked_image(os.path.join(dataset, image), masks[image])
            except KeyError:
                continue
    return masked_images
