import numpy as np
import torch
from matplotlib import pyplot as plt
from segment_anything import SamPredictor, sam_model_registry
from typing import List, Dict
from torch import Tensor
import cv2
import os


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def sam_register(checkpoint: str, model_type: str = 'vit_b', device: str = 'cpu') -> SamPredictor:
    """
    Register SAM model

    Args:
        checkpoint: The path to the checkpoint file.
        model_type: The type of the model, vit_b , vit_l or vit_h.
        device: The device to run the model on, cpu or cuda.

    Returns:
        SAM predictor
    """
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    return predictor


def sam_inference_single(
        image: str,
        prompt: List[List],
        predictor: SamPredictor,
        multimask_output: bool = False,
) -> Tensor:
    """
    Run SAM inference on a single image.

    Args:
        image: The path to the image file.
        prompt: Prompt boxes, [[xmin, ymin, xmax, ymax], ...].
        predictor: SAM predictor.
        multimask_output: Whether to output multimask.

    Returns:
        Masks (num_boxes) x (num_predicted_masks_per_input) x H x W
    """

    # TODO: multimask_output is not supported yet.
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    predictor.set_image(img)

    input_boxes = torch.tensor(prompt, device=predictor.device)
    transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, img.shape[:2])

    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=multimask_output,
    )

    return masks


def sam_inference(
        dataset: str,
        prompt: Dict[str, List[List]],
        predictor: SamPredictor,
        multimask_output: bool = False,
) -> Dict[str, Tensor]:
    """
    Run SAM inference on a dataset.

    Args:
        images: The path to the image dataset directory.
        prompt: Prompt boxes of each image, {image.jpg: [[xmin, ymin, xmax, ymax], ...], ...}.
        predictor: SAM predictor.
        multimask_output: Whether to output multimask.

    Returns:
        Masks of each image (num_boxes) x (num_predicted_masks_per_input) x H x W.
    """

    # TODO: multimask_output is not supported yet.
    results = {}
    filenames = os.listdir(dataset)
    length = len(filenames)

    for filename in filenames:
        assert filename in prompt.keys(), "Can't find prompt for image {}".format(filename)
        filepath = os.path.join(dataset, filename)
        masks = sam_inference_single(filepath, prompt[filename], predictor, multimask_output)
        results[filename] = masks
        print("Processed {}/{} images".format(len(results), length))

    return results
