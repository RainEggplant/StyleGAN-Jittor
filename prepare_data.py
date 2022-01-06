import argparse
import multiprocessing
from functools import partial
from pathlib import Path

from PIL import Image


def resize(file_in, dir_out, size, base_path=None):
    img = Image.open(file_in).convert('RGB')
    for s in size:
        img_resized = img.resize((s, s))
        file_out = file_in.relative_to(base_path) if base_path is not None else file_in.name
        file_out = dir_out / str(s) / file_out
        img_resized.save(file_out)


def main(input_dir, output_dir, n_worker, size):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    for s in size:
        (output_dir / str(s)).mkdir(parents=True, exist_ok=True)

    resize_fn = partial(resize, dir_out=output_dir, size=size, base_path=input_dir)

    ext_img = ['.png', '.jpg', '.jpeg', '.bmp']
    file_list = []
    for ext in ext_img:
        file_list.extend(input_dir.rglob(f'*{ext}'))

    with multiprocessing.Pool(n_worker) as pool:
        pool.map(resize_fn, file_list)


if __name__ == '__main__':
    def parse_size(size_str):
        size = tuple([int(x) for x in size_str.split(',')])
        valid = [x > 0 for x in size]
        if not all(valid):
            raise AttributeError()
        return size

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True)
    parser.add_argument('-o', '--output', type=str, required=True)
    parser.add_argument('--size', type=parse_size, default=(8, 16, 32, 64, 128))
    parser.add_argument('--n_worker', type=int, default=8)
    args = parser.parse_args()

    main(args.input, args.output, args.n_worker, args.size)
