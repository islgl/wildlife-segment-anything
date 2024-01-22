import cv2

from yolov5 import run as yolo_run
from segment_anything import sam_register, sam_inference
from segment_anything.utils import get_masked_images
import yaml

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
    masks = sam_inference(
        dataset=config['dataset'],
        prompt=prompts,
        predictor=sam,
    )

    # Get masked images
    masked_images = get_masked_images(
        dataset=config['dataset'],
        masks=masks,
    )

for image, masked_image in masked_images.items():
    filename='/Users/lgl/code/machine_learning/wildlife-segment-anything/results/'+image.split('.')[0]+'.png'
    cv2.imwrite(filename, masked_image.numpy())
