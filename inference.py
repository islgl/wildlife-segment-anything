from yolov5 import run as yolo_run
import yaml

if __name__ == '__main__':
    # Load configuration
    config_path = './config/inference.yaml'
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Run YOLOv5 inference
    result = yolo_run(
        weights=config['yolo'],
        source=config['dataset'],
        data=config['config'],
        device=config['device'],
    )
    print(result)
