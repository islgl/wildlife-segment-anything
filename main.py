import json

if __name__ == '__main__':
    filepath = '/Users/lgl/code/machine_learning/wildlife-segment-anything/prompts/voc.json'
    with open(filepath) as f:
        prompts = json.load(f)
    print(len(prompts))