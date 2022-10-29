import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class SMOKEBASE(Dataset):
    def __init__(self,
                txt_file,
                data_root,
                size=None,
                interpolation="bicubic",
                flip_p = 0.5
                ):
        
        self.data_paths = txt_file
        self.data_root = data_root
        with open(self.data_paths, "r") as f:
            self.image_paths = f.read().splitlines()
        self._length = len(self.image_paths)
        self.labels = {
            "relative_file_path_": [l for l in self.image_paths],
            "file_path_": [os.path.join(self.data_root, l)
                           for l in self.image_paths],
        }

        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]

        self.flip = transforms.RandomHorizontalFlip(p=flip_p)
        self.flip_2 = transforms.RandomVerticalFlip(p=flip_p)


    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)

        image = Image.open(example["file_path_"])
        # mask = Image.open(os.path.splitext(example["file_path_"])[0] + "_mask" + os.path.splitext(os.path.basename(example["file_path_"]))[1] )

        if not image.mode == "RGB":
            image = image.convert("RGB")

        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)
            # mask = mask.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip(image)
        image = self.flip_2(image)
        image = np.array(image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        
        return example


class SMOKETrain(SMOKEBASE):
    def __init__(self, **kwargs):
        super().__init__(txt_file="data/smoke/smoke_train.txt", data_root="/home/mateo/pyronear/latent-diffusion/data/smoke", **kwargs)


class SMOKEValidation(SMOKEBASE):
    def __init__(self, flip_p=0.0, **kwargs):
        super().__init__(txt_file="data/smoke/smoke_val.txt", data_root="/home/mateo/pyronear/latent-diffusion/data/smoke",
                         flip_p=flip_p, **kwargs)