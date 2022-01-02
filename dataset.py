from pathlib import Path

from PIL import Image
from jittor.dataset import Dataset


class MultiResolutionDataset(Dataset):
    def __init__(self, root, transform, resolution_level=8):
        super().__init__()
        self.root = Path(root)
        self.transform = transform
        self._resolution = (2 ** resolution_level, 2 ** resolution_level)

        ext_img = ['.png', '.jpg', '.jpeg', '.bmp']
        file_list = []
        for ext in ext_img:
            file_list.extend(self.root.rglob(f'*{ext}'))
        self.files = file_list

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img = Image.open(self.files[index]).resize(self._resolution)
        img = self.transform(img)
        return img

    @property
    def resolution(self):
        return self._resolution

    def set_resolution(self, resolution_level):
        self._resolution = (2 ** resolution_level, 2 ** resolution_level)
        self.reset()
