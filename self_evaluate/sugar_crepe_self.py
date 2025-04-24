import argparse
import os
import sys
os.environ['HF_ENDPOINT'] = ''
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = "
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from tqdm import tqdm
from torch.nn import functional as F
from accelerate import Accelerator
accelerator = Accelerator()

from self_evaluate.utils.utils import init_model_and_transform, dataset_2_HFdataset_4_sugarcrepe
from self_evaluate.utils.sugar_crepe import surgar_crepe_dataset_4_LLM
import warnings
warnings.filterwarnings("ignore")

def process_text(text_prompt, input_text_pos, transform):
        input_texts_pos = [text_prompt.replace('<sent>', text) for text in input_text_pos]
        inputs_text_pos = transform(input_texts_pos,
                            return_tensors="pt", 
                            padding=True)
        for key in inputs_text_pos:
            if inputs_text_pos[key] is not None:
                inputs_text_pos[key] = inputs_text_pos[key].to(device)
        return inputs_text_pos

def text_retrieval_llm(input_text_pos,
                        input_text_neg,
                        input_image, 
                        model, 
                        transform, 
                        model_name,
                        device="cuda"):    
        
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
    
    inputs_text_pos = process_text(text_prompt, input_text_pos, transform)
    inputs_text_neg = process_text(text_prompt, input_text_neg, transform)
    input_image_prompt = [img_prompt]*len(input_image)
    
    if model_name == "phi35V":
        assert len(input_image_prompt) == 1
        input_image_prompt = input_image_prompt[0]
        
    input_image = transform(input_image_prompt,
                        input_image, 
                        return_tensors="pt", 
                        padding=True).to(device)
    
    with torch.no_grad():
        emb_text_pos = model(**inputs_text_pos, output_hidden_states=True, return_dict=True).hidden_states[-1][:, -1, :]
        emb_text_pos = F.normalize(emb_text_pos, dim=-1)
        emb_text_neg = model(**inputs_text_neg, output_hidden_states=True, return_dict=True).hidden_states[-1][:, -1, :]
        emb_text_neg = F.normalize(emb_text_neg, dim=-1)
        image_embedding = model(**input_image, output_hidden_states=True, return_dict=True).hidden_states[-1][:, -1, :]
        image_embedding = F.normalize(image_embedding, dim=-1)
    
    emb_text_pos = accelerator.gather(emb_text_pos)
    emb_text_neg = accelerator.gather(emb_text_neg)
    image_embedding = accelerator.gather(image_embedding)
    
    pos_score = emb_text_pos @ image_embedding.t()
    neg_score = emb_text_neg @ image_embedding.t()
    
    pos_score = pos_score.diagonal()
    neg_score = neg_score.diagonal()
    
    ans = sum([1 if a.item() > b.item() else 0 for a,b in zip(pos_score, neg_score)])
    return ans

def evaluate_llm(dataset_name, 
                 model, 
                 transform, 
                 model_name,
                 device,
                 bsz=1):
    
    metrics = {}
    for data_name in dataset_name:
        correct_cnt = 0
        dataset = surgar_crepe_dataset_4_LLM(data_name)
        count = dataset.__len__()
        dataset = dataset_2_HFdataset_4_sugarcrepe(dataset)
        
        def custom_collate_fn(batch):
            collated_batch = {}
            for key in batch[0].keys():
                collated_batch[key] = [b[key] for b in batch]
            return collated_batch

        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=bsz,
            shuffle=False, num_workers=1,
            collate_fn=custom_collate_fn
        )
        
        dataloader = accelerator.prepare(dataloader)
        if accelerator.is_main_process: 
            bar = tqdm(total=len(dataloader), desc=f'evaluating {data_name}')
        for batch in dataloader:
            correct = text_retrieval_llm(input_text_pos = batch['pos'], 
                                         input_text_neg = batch['neg'], 
                                         input_image = batch['img'], 
                                         model = model, 
                                         transform = transform, 
                                         model_name = model_name,
                                         device = device)
            correct_cnt += correct
            if accelerator.is_main_process: bar.update(1)
        metrics[data_name] = correct_cnt / count
    return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model_path', type=str, default="", help="")
    parser.add_argument('--output_path', default="", type=str, help="") 
    parser.add_argument('--model_name', default="", type=str, help="") # phi35V, llava_16
    args = parser.parse_args()

    device=accelerator.device
    data_name = ['replace_obj', 'replace_att', 'replace_rel', 'swap_obj', 'swap_att', 'add_obj', 'add_att']
    
    if accelerator.is_main_process: 
        print(" =========== Evaluation Dataset =========== ")
        print(str(data_name))
        print(" =========== Evaluation Dataset =========== ")

    model, transform = init_model_and_transform(args.model_name, args.base_model_path)
    model.to(device)

    metrics = evaluate_llm(data_name, 
                            model, 
                            transform, 
                            args.model_name,
                            device)
    if accelerator.is_main_process: 
        print(metrics)
        if args.output_path is not None:
            assert args.output_path.endswith(".txt"), print("output path needs to end with .txt")
            output_files_name = args.output_path
        else:
            output_files_name = f"evaluate_result/{args.model_name}.txt"
        with open(output_files_name, 'a') as f:
            print(" ============ Retrieval Testing ============ ", file=f)
            for k,v in metrics.items():
                print(f"{str(k)}:{str(v)}", file=f)
            
