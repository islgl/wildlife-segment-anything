import os

dir='/Users/lgl/code/machine_learning/wildlife-segment-anything/results'
for img in os.listdir(dir):
    if img.endswith('.png'):
        os.rename(os.path.join(dir, img), os.path.join(dir, img.split('.')[0]+'.jpg'))
