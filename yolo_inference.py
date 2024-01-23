import yaml
import json
import os

from yolov5 import run as yolo_run

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

    # Save prompts
    if not os.path.exists(os.path.dirname(config['prompts'])):
        os.makedirs(os.path.dirname(config['prompts']))
    with open(config['prompts'], 'w') as f:
        json.dump(prompts, f)
