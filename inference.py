from yolov5 import run as yolo_run
from segment_anything import sam_register, sam_inference
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




