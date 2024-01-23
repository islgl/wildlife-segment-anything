import torch, gc
from typing import List, Dict
from torch import Tensor
import cv2
import os
from tools import release_memory

from .build_sam import (
    build_sam,
    build_sam_vit_h,
    build_sam_vit_l,
    build_sam_vit_b,
    sam_model_registry,
)
from .predictor import SamPredictor


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
        prompt: List,
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

    gc.collect()
    torch.cuda.empty_cache()

    return masks


def sam_run(
        dataset: str,
        prompt: Dict,
        predictor: SamPredictor,
        multimask_output: bool = False,
        if_log: bool = False,
        log_path: str = None,
) -> Dict:
    """
    Run SAM inference on a dataset.

    Args:
        images: The path to the image dataset directory.
        prompt: Prompt boxes of each image, {image.jpg: [[xmin, ymin, xmax, ymax], ...], ...}.
        predictor: SAM predictor.
        multimask_output: Whether to output multimask.
        if_log: Whether to log the results.
        log_path: The path to the log file. Only used when if_log is True.

    Returns:
        Masks of each image (num_boxes) x (num_predicted_masks_per_input) x H x W.
    """

    # TODO: multimask_output is not supported yet.
    results = {}
    if os.path.isdir(dataset):
        filenames = os.listdir(dataset)
        length = len(filenames)

        for filename in filenames:
            filepath = os.path.join(dataset, filename)
            if prompt[filename] == [] or prompt[filename] is None:
                msg = 'No prompt for image {}'.format(filename)
                print(msg)
                if if_log and log_path is not None:
                    if not os.path.exists(os.path.dirname(log_path)):
                        os.makedirs(os.path.dirname(log_path))
                    with open(log_path, 'a') as f:
                        f.write(msg + '\n')
                continue
            masks = sam_inference_single(filepath, prompt[filename], predictor, multimask_output)
            results[filename] = masks
            print("Processed {}/{} images".format(len(results), length))
    else:
        filename = os.path.basename(dataset)
        masks = sam_inference_single(dataset, prompt[filename], predictor, multimask_output)
        results[filename] = masks

    return results
