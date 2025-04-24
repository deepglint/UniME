import json
from PIL import Image
import torch.utils.data as data
from self_evaluate.utils.data_path import ShareGPT4v

class share4v_val_dataset_4_LLM(data.Dataset):
    def __init__(self):
        self.data4v_root = ShareGPT4v["data4v_root"]
        self.json_name = ShareGPT4v["json_name"]
        self.image_root = ShareGPT4v["image_root"]
        self.total_len = 1000
        with open(self.data4v_root + self.json_name, 'r',encoding='utf8')as fp:
            self.json_data = json.load(fp)[:self.total_len]
    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        caption = self.json_data[index]['conversations'][1]['value']
        caption = caption.replace("\n", " ")
        image_name = self.image_root + self.json_data[index]['image']
        image = Image.open(image_name)
        return image, caption