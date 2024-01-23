import torch
import yaml
import json

from yolov5 import run as yolo_run
from segment_anything import sam_register, sam_run
from segment_anything.utils import get_masked_images
from mobilenet.utils import preprocess
from mobilenet import run as mobilenet_run

if __name__ == '__main__':
    # Load configuration
    config_path = './config/inference.yaml'
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Run YOLOv5 inference
    prompts = yolo_run(
        weights=config['yolo'],
        source=config['dataset'],
        data=config['config'],
        device=config['device'],
    )

    # Run SAM inference
    sam = sam_register(
        checkpoint=config['sam_checkpoint'],
        model_type=config['sam_model_type'],
        device=config['device'],
    )
    masks = sam_run(
        dataset=config['dataset'],
        prompt=prompts,
        predictor=sam,
        if_log=True,
        log_path=config['sam_log'],
    )

    # Get masked images
    masked_images = get_masked_images(
        dataset=config['dataset'],
        masks=masks,
    )

    # Run MobileNet inference
    # Load MobileNet model
    mobilenet = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
    mobilenet.eval()

    # Preprocess images
    mobilenet_input = preprocess(masked_images)

    # Run inference
    results = mobilenet_run(
        input_batches=mobilenet_input,
        label_path=config['labels'],
    )

    # Save masks
    with open(config['save_path'], 'w') as f:
        json.dump(results, f)