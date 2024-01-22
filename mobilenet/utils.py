from torchvision import transforms
from typing import Dict, List
from PIL import Image
from torch import Tensor
import torch


def preprocess(dataset: Dict) -> Dict:
    """
    Preprocesses images for MobileNet inference.
    Args:
        dataset: The dataset to preprocess.

    Returns:
        The preprocessed dataset.
    """

    img_preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    for k, v in dataset.items():
        input_tensor = img_preprocess(v)
        input_batch = input_tensor.unsqueeze(0)
        dataset[k] = input_batch

    return dataset