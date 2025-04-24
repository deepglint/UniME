import os
import sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['HF_ENDPOINT'] = ''
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = "

from datasets import load_dataset, disable_caching
import torch
import torch.nn.functional as F
from self_evaluate.utils.utils import recall_at_k, batchify, emb_data, log_to_file, init_model_and_transform, dataset_2_HFdataset
from self_evaluate.utils.sharegpt4v import share4v_val_dataset_4_LLM
from self_evaluate.utils.urban1k import urban1k_dataset_4_LLM

import warnings
warnings.filterwarnings("ignore")
from accelerate import Accelerator
accelerator = Accelerator()

def ir(model,
       transform,
       model_name,
       img_prompt, 
       text_prompt,
       data, 
       device,
       batch_size=4):
    
    if data == "coco2014":
        dataset = load_dataset(f'royokong/coco_test', split='test')
    elif data == "flickr30k":
        dataset = load_dataset(f'royokong/{data}_test', split='test')
    elif data == "sharegpt4v":
        dataset = share4v_val_dataset_4_LLM()
        dataset = dataset_2_HFdataset(dataset)
    elif data == "Urban200K":
        dataset = urban1k_dataset_4_LLM()
        dataset = dataset_2_HFdataset(dataset)
    
    if data == 'coco2014' or data == "flickr30k":
        dataset = dataset.rename_column('text', 'caption')
        dataset = dataset.rename_column('image', 'img')
    
    if data == 'coco2014':
        dataset = dataset.map(lambda x: {'caption': x['caption'][:5]}, num_proc=4)
    
    text_embs = emb_data(model, transform, dataset, device, MODEL_TYPE = model_name, dataset_name=data, emb_type='text', prompt=text_prompt, bsz=batch_size)
    img_embs = emb_data(model, transform, dataset, device,  MODEL_TYPE = model_name, dataset_name=data, emb_type='image', prompt=img_prompt, bsz=batch_size)
    if data == "coco2014" or data == "flickr30k":
        texts_image_index = [i // 5 for i in range(img_embs.shape[0]*5)]
    else:
        texts_image_index = [i for i in range(img_embs.shape[0])]
        
    assert len(texts_image_index) == len(text_embs)
    assert img_embs.isnan().sum().item() == 0, f'nan in images emb: {img_embs.isnan().sum().item()}/{img_embs.size(0)}'
    assert text_embs.isnan().sum().item() == 0, f'nan in retrieve emb: {text_embs.isnan().sum().item()}/{text_embs.size(0)}'

    scores  = text_embs @ img_embs.t()
    positive_pairs = torch.zeros_like(scores, dtype=bool)
    positive_pairs[torch.arange(len(scores)), texts_image_index] = True
    metrics = {}
    recall_k_list = [1, 5, 10]
    batch_size = 64
    for recall_k in recall_k_list:
        metrics[f"image_retrieval_recall@{recall_k}"] = (batchify(recall_at_k, scores, positive_pairs, batch_size, device, k=recall_k)>0).float().mean().item()
        metrics[f"text_retrieval_recall@{recall_k}"] = (batchify(recall_at_k, scores.T, positive_pairs.T, batch_size, device, k=recall_k)>0).float().mean().item()

    return metrics

def main(base_model_path: str="",
        batch_size: int = 1,
        data: str = "",
        model_name: str = "phi35V", # llava_16
        output_path: str = "",
):
    device=accelerator.device 
    model, transform = init_model_and_transform(model_name, base_model_path)

    model.to(device)
    disable_caching()

    datasets = ['flickr30k', 'coco2014','sharegpt4v', 'Urban200K']
    
    if data: datasets = [str(each).replace("_"," ") for each in data]
    if accelerator.is_main_process: print(f"Data for testing: {str(datasets)}")

    all_results = []
    for data in datasets: 
        if model_name == "phi35V":
            img_prompt = '<|user|>\n<|image_1|>\nSummary above image in one word:<|end|>\n<|assistant|>\n'
            text_prompt = '<|user|>\n<sent>\nSummary above sentence in one word:<|end|>\n<|assistant|>\n'
        elif model_name == "llava_16":
            img_prompt = "[INST] <image>\nSummary above image in one word: [/INST]"
            text_prompt = "[INST] <sent>\nSummary above sentence in one word: [/INST]"    
        else:
            LLAMA3_TEMPLATE = '<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n \n'
            img_prompt = LLAMA3_TEMPLATE.format('<image>\nSummary above image in one word: ')
            text_prompt = LLAMA3_TEMPLATE.format('<sent>\nSummary above sentence in one word: ')

        if accelerator.is_main_process:
            print(img_prompt)
            print(text_prompt)
           
        metrics = ir(model, 
                    transform, 
                    model_name,
                    img_prompt, 
                    text_prompt,
                    data, device,  
                    batch_size)
            
        if accelerator.is_main_process:
            print(metrics)
            if output_path is not None:
                assert output_path.endswith(".txt"), print("output path needs to end with .txt")
                checkpoint_name = output_path
            else:
                checkpoint_name = f"evaluate_result/{model_name}.txt"
            all_results.append(log_to_file(data, metrics, checkpoint_name))

    if accelerator.is_main_process:
        print('\n'.join(all_results))

if __name__ == '__main__':
    from fire import Fire
    Fire(main)
