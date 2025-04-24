import os
from PIL import Image
import torch.utils.data as data
from self_evaluate.utils.data_path import Urban1k

class urban1k_dataset_4_LLM(data.Dataset):
    def __init__(self):
        self.image_root = Urban1k["image_root"]
        self.caption_root = Urban1k["caption_root"]
        self.total_image = os.listdir(self.image_root)
        self.total_caption = os.listdir(self.caption_root)
    def __len__(self):
        return len(self.total_caption)

    def __getitem__(self, index):
        caption_name = self.total_caption[index]
        image_name = self.total_caption[index][:-4] + '.jpg'
        image = Image.open(self.image_root + image_name)
        f=open(self.caption_root + caption_name)
        caption = f.readlines()[0]
        return image, caption