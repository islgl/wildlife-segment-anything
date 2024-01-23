import yaml
import json
import os

from segment_anything import sam_register, sam_run
from segment_anything.utils import get_masked_images

if __name__ == '__main__':
    # Load configuration
    config_path = './config/inference.yaml'
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Load prompts
    with open(config['prompts']) as f:
        prompts = json.load(f)

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
        if_log=False,
        log_path=config['sam_log'],
    )

    # Get masked images
    masked_images = get_masked_images(
        dataset=config['dataset'],
        masks=masks,
    )

    if not os.path.exists(config['save_path']):
        os.makedirs(config['save_path'])

    # 以RGB形式写入图片
    for i, img in masked_images.items():
        save_path = config['save_path'] + i
        img.save(save_path)


