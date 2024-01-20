import numpy as np
import yaml
import cv2
from segment_anything import sam_model_registry, SamPredictor
import matplotlib.pyplot as plt


# Load configuration
config_path = './config/inference.yaml'
with open(config_path) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

device = config['device']

# Register SAM
sam_checkpoint = config['sam_checkpoint']
sam_model_type = config['sam_model_type']

sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)

# Run SAM inference
image = cv2.imread('/Users/lgl/code/machine_learning/wildlife-segment-anything/data/images/coco/bird.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

predictor.set_image(image)

input_box = np.array([124, 124, 482, 468])

masks, _, _ = predictor.predict(
    point_coords=None,
    point_labels=None,
    box=input_box[None, :],
    multimask_output=False,
)

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.imshow(image)
show_box(input_box, ax)
show_mask(masks[0], ax)
plt.show()
