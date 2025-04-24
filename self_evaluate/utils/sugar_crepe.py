import os
import json
from PIL import Image
import torch.utils.data as data
from self_evaluate.utils.data_path import SugarCrepe

data_root = SugarCrepe["data_root"]
data_dict = {
    'add_obj'    : f'{data_root}/add_obj.json',
    'add_att'    : f'{data_root}/add_att.json',
    'replace_obj': f'{data_root}/replace_obj.json',
    'replace_att': f'{data_root}/replace_att.json',
    'replace_rel': f'{data_root}/replace_rel.json',
    'swap_obj'   : f'{data_root}/swap_obj.json',
    'swap_att'   : f'{data_root}/swap_att.json',
}

class surgar_crepe_dataset_4_LLM(data.Dataset):
    def __init__(self, name):
        self.name = name
        self.image_root = SugarCrepe["image_root"]
        self.dataset = json.load(open(data_dict[name], 'r', encoding='utf-8'))
        self.dataset_keys = list(self.dataset.keys())
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        index = self.dataset_keys[index]
        pos = self.dataset[index]['caption']
        neg = self.dataset[index]['negative_caption']
        image_path = os.path.join(self.image_root, self.dataset[index]['filename'])
        image = Image.open(image_path)
        return pos, neg, image