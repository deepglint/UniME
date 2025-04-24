import torch
from tqdm import tqdm
from torch.nn import functional as F

from datasets import Dataset as HFDataset
from transformers import AutoProcessor, AutoModelForCausalLM, LlavaNextProcessor, LlavaNextForConditionalGeneration

from accelerate import Accelerator
accelerator = Accelerator()

def dataset_2_HFdataset(dataset):
    images = []
    captions = []
    for i in range(len(dataset)):
        image, caption = dataset[i]
        images.append(image) 
        captions.append(caption)
    data_dict = {
        'img': images,
        'caption': captions
    }
    dataset = HFDataset.from_dict(data_dict)
    return dataset

def recall_at_k(scores, 
                positive_pairs, 
                k):
    nb_texts, nb_images = scores.shape
    topk_indices = torch.topk(scores, k, dim=1)[1]
    nb_positive = positive_pairs.sum(dim=1)
    topk_indices_onehot = torch.nn.functional.one_hot(topk_indices, num_classes=nb_images)
    positive_pairs_reshaped = positive_pairs.view(nb_texts, 1, nb_images)
    nb_true_positive = (topk_indices_onehot * positive_pairs_reshaped).sum(dim=(1,2))
    recall_at_k = (nb_true_positive / nb_positive)
    return recall_at_k

def batchify(func, 
             X, 
             Y, 
             batch_size, 
             device, 
             *args, 
             **kwargs):
    results = []
    for start in range(0, len(X), batch_size):
        end = start + batch_size
        x = X[start:end].to(device)
        y = Y[start:end].to(device)
        result = func(x, y, *args, **kwargs).cpu()
        results.append(result)
    return torch.cat(results)

def emb_data(model, 
             transform, 
             dataset, 
             device,
             MODEL_TYPE,
             emb_type='text', 
             prompt=None, 
             bsz=4,
             dataset_name="flickr30k",
             text_column='caption', 
             img_column='img'):
    # emb img
    def custom_collate_fn(batch):
        collated_batch = {}
        for key in batch[0].keys():
            collated_batch[key] = [b[key] for b in batch]
        return collated_batch
    
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=3*bsz if emb_type == 'text' else bsz,
        shuffle=False, num_workers=1,
        collate_fn=custom_collate_fn
    )
    
    dataloader = accelerator.prepare(dataloader)
    embs = []
    bar = tqdm(total=len(dataloader))
    for batch in dataloader:
        if emb_type == 'text':
            if dataset_name == "sharegpt4v" or dataset_name == "Urban200K":
                input_texts = [prompt.replace('<sent>', text) for text in batch[text_column]]
            else:
                input_texts = [prompt.replace('<sent>', text) for text in sum(batch[text_column], start=[])]
            inputs = transform(text=input_texts,
                               images=None,
                               return_tensors="pt", 
                               padding=True)
            for key in inputs:
                if inputs[key] is not None:
                   inputs[key] = inputs[key].to(device)
        else:
            input_texts = [prompt]*len(batch[img_column])
            if MODEL_TYPE == 'phi35V':
                #! phi3 only support 1 bsz for image
                assert len(input_texts) == 1
                input_texts = input_texts[0]
            inputs = transform(text=input_texts,
                               images=batch[img_column], 
                               return_tensors="pt", 
                               padding=True).to(device)
        with torch.no_grad():
            emb = model(**inputs, output_hidden_states=True, return_dict=True).hidden_states[-1][:, -1, :]
            emb = F.normalize(emb, dim=-1)
        emb = accelerator.gather(emb)
        embs.append(emb.cpu().float())
        bar.update(1)
    embs = torch.cat(embs)
    total = 0
    for i in dataset:
        if emb_type == 'text' and type(i[text_column]) is list:
            total += len(i[text_column])
        else:
            total += 1
    bar.close()
    return embs[:total]

def log_to_file(data, 
                metrics, 
                checkpoint_name):
    if data == 'flickr30k' or data == 'coco2014' or data == "sharegpt4v" or data == "Urban200K" or data == "coco2017":
        output = f"{data} Retrieval: {str(metrics)}" 

    if checkpoint_name is not None:
        with open(checkpoint_name, 'a') as f:
            print(output, file=f)
    return output

def init_model_and_transform(model_name, base_model_path): 
    if model_name == 'phi35V':
        transform = AutoProcessor.from_pretrained("microsoft/Phi-3.5-vision-instruct", trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(base_model_path,
                                                device_map="cuda", trust_remote_code=True,
                                                torch_dtype=torch.float16, _attn_implementation='flash_attention_2')
    elif model_name == "llava_16": 
        transform = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
        model = LlavaNextForConditionalGeneration.from_pretrained(base_model_path, device_map="cuda", torch_dtype=torch.float16, low_cpu_mem_usage=True) 
    transform.tokenizer.padding_side = "left"
    transform.tokenizer.padding = True
    return model, transform


def dataset_2_HFdataset_4_sugarcrepe(dataset):
    images = []
    pos_list = []
    neg_list = []
    for i in range(len(dataset)):
        pos, neg, image = dataset[i]
        images.append(image)
        pos_list.append(pos)
        neg_list.append(neg)
    data_dict = {
        'img': images,
        'pos': pos_list,
        'neg': neg_list
    }
    dataset = HFDataset.from_dict(data_dict)
    return dataset