import json
import os


info_json_path = r'../dair_v2x/cooperative-vehicle-infrastructure-example'
info_name = 'example-cooperative-split-data.json'

splits = ['train', 'val', 'test']


if __name__ == '__main__':
    with open(os.path.join(info_json_path, info_name), 'rb') as f:
        split_data = json.load(f)

    for split in splits:
        split_info = split_data["cooperative_split"][split]
        split_json_name = f'{split}.json'
        with open(os.path.join(info_json_path, split_json_name), 'w') as f:
            json.dump(split_info, f)
