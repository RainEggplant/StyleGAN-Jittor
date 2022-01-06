from pathlib import Path

from PIL import Image
from jittor.dataset import Dataset


class MultiResolutionDataset(Dataset):
    def __init__(self, root, transform, resolution=8):
        super().__init__()
        self.root = Path(root)
        self.transform = transform
        self._resolution = (resolution, resolution)

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
        return self._resolution[0]

    def set_resolution(self, resolution):
        self._resolution = (resolution, resolution)

        # cannot change resolution when n_workers > 0 due to Jittor's bug,
        # or it will raise "'Dataset' object has no attribute 'gid_obj'" error
        self.reset()


class FolderDataset(Dataset):
    def __init__(self, root, transform):
        super().__init__()
        self.root = Path(root)
        self.transform = transform

        ext_img = ['.png', '.jpg', '.jpeg', '.bmp']
        file_list = []
        for ext in ext_img:
            file_list.extend(self.root.rglob(f'*{ext}'))
        self.files = file_list

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img = Image.open(self.files[index])
        img = self.transform(img)
        return img
