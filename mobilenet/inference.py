import torch
from torch import Tensor
from typing import Dict


def run_single(input_batch: Tensor) -> Tensor:
    # Load MobileNet model
    model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
    model.eval()

    # Run inference
    with torch.no_grad():
        output = model(input_batch)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    return probabilities


def run(input_batches: Dict, label_path: str, topn: int = 1) -> Dict:
    """
    Run inference on the input images.

    Args:
        input_batches: The preprocessed input images to run inference on.
        label_path: The path to the categories label file.
        topn: The number of top categories to return.

    Returns:
        The dictionary of categories and their probabilities.
    """

    with open(label_path, "r") as f:
        categories = [s.strip() for s in f.readlines()]

    results = {}

    for k, v in input_batches.items():
        probabilities = run_single(v)
        top_categories = {}
        top_prob, top_catid = torch.topk(probabilities, topn)
        for i in range(top_prob.size(0)):
            top_categories[categories[top_catid[i]]] = top_prob[i].item()
        results[k] = top_categories

    return results
