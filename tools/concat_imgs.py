import glob
import os

from PIL import Image
from tqdm import tqdm


def concat_images(img_files, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    img_size = Image.open(img_files[0][0]).size

    for imgs in tqdm(list(zip(*img_files))):
        img_concat = Image.new('RGB', (img_size[0], img_size[1] * len(imgs)))
        for i, img in enumerate(imgs):
            img_concat.paste(Image.open(img), (0, i * img_size[1]))
        img_concat.save(os.path.join(output_dir, os.path.basename(imgs[0])))


if __name__ == '__main__':
    img_dirs = ['vis_noadapt', 'vis_da']
    output_dir = 'vis'

    img_files_3d = []
    img_files_bev = []

    for img_dir in img_dirs:
        img_files_3d.append(sorted(glob.glob(os.path.join(img_dir, '3d/*.png'))))
        img_files_bev.append(sorted(glob.glob(os.path.join(img_dir, 'bev/*.png'))))

    concat_images(img_files_3d, os.path.join(output_dir, '3d'))
    concat_images(img_files_bev, os.path.join(output_dir, 'bev'))
