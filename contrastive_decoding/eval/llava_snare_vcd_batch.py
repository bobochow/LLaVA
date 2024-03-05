import argparse
import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import json
from tqdm import tqdm
import shortuuid
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math
import random
import numpy as np
import copy
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Tuple

import transformers
from transformers import set_seed
from contrastive_decoding.decoding_utils.vcd_decoding import add_diffusion_noise
from contrastive_decoding.decoding_utils.vcd_decoding import evolve_vcd_sampling
evolve_vcd_sampling()

class CustomDataset(Dataset):
    def __init__(self, questions_folder, image_folder, tokenizer, image_processor, model_config, noise_step, max_instances, subclausal, conv_mode):
        
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.noise_step = noise_step
        self.subclausal = subclausal
        self.conv_mode = conv_mode
        
        with open(questions_folder, "r") as f:
            self.dataset = json.load(f)
        
        if max_instances != -1:
            random.seed(42)
            random.shuffle(self.dataset)
            self.dataset = self.dataset[:max_instances]
        
        for item in self.dataset:
            item["image_path"] = os.path.join(image_folder, item["image_path"])
        
        self.all_attributes = [f"{item['attributes'][0]}_{item['attributes'][1]}" for item in self.dataset]
        
        if self.subclausal:
            self.cla_name = ["negative1", "negative2"]
        else:
            self.cla_name = ["correct", "negative"]
        print(self.cla_name)



    def __getitem__(self, index):
        line = self.dataset[index]
        
        qs_options = []
        if self.subclausal:
            negative1 = "Is the {} not {} and the {} {}?".format(line["obj1_name"],
                                                                    line["attributes"][0],
                                                                    line["obj2_name"],
                                                                    line["attributes"][1])
            
            negative2 = "Is the {} {} and the {} not {}?".format(line["obj1_name"],
                                                                    line["attributes"][0],
                                                                    line["obj2_name"],
                                                                    line["attributes"][1])
            
            qs_options = [negative1,negative2]
        else:
            positive = "Is the {} {} and the {} {}?".format(line["obj1_name"],
                                                                    line["attributes"][0],
                                                                    line["obj2_name"],
                                                                    line["attributes"][1])
            
            negative = "Is the {} not {} and the {} not {}?".format(line["obj1_name"],
                                                                    line["attributes"][0],
                                                                    line["obj2_name"],
                                                                    line["attributes"][1])
            qs_options = [positive,negative]
        
        input_ids_list=[]
        for qs in qs_options:
            
            if self.model_config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs + '\nAnswer the question using a single word or phrase.'
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs + '\nAnswer the question using a single word or phrase.'
            
            conv = conv_templates[self.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
            input_ids_list.append(input_ids)

        image = Image.open(line["image_path"]).convert('RGB')
        image = image.crop((line["bbox_x"], line["bbox_y"], line["bbox_x"] + line["bbox_w"],
                            line["bbox_y"] + line["bbox_h"]))
        
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]
        
        image_tensor_cd = add_diffusion_noise(image_tensor, self.noise_step)
        
        return index, input_ids_list[0], input_ids_list[1], image_tensor, image_tensor_cd

    def __len__(self):
        return len(self.dataset)
    
    def evaluate_vllm_scores(self, scores):
        """
        Scores: M x 1 x N, i.e. first caption is the perturbed one, second is the positive one
        """
        
        cla_acc = {self.cla_name[i]: sum(scores[:, i])/len(scores) for i in range(len(self.cla_name))}
        result_records = []
        # 子类别数据
        all_attributes = np.array(self.all_attributes)
        for attr in np.unique(all_attributes):
            attr_mask = (all_attributes == attr)
            score_sub = scores[attr_mask]
            if attr_mask.sum() < 25:
                continue
            res_dict = {
                "Attributes": attr,
                "Count": attr_mask.sum(),
            }
            for i in range(len(self.cla_name)):
                res_dict.update({self.cla_name[i]: sum(score_sub[:, i])/len(score_sub)})
            result_records.append(res_dict)

        # 总体数据
        for key, value in cla_acc.items():
            res_dict = {
                "Attributes": key,
                key: value,
            }
            result_records.append(res_dict)
        return result_records


@dataclass
class DataCollatorForVisualTextGeneration(object):
    tokenizer: transformers.PreTrainedTokenizer

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids] 
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=batch_first,
            padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    def __call__(self,
                batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        indices, pos_input_ids, neg_input_ids, image_tensor, image_tensor_cd = zip(*batch)
        pos_input_ids = self.pad_sequence(
            pos_input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        neg_input_ids = self.pad_sequence(
            neg_input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        
        image_tensor = torch.stack(image_tensor, dim=0)
        image_tensor_cd = torch.stack(image_tensor_cd, dim=0)
        return indices, [pos_input_ids, neg_input_ids], image_tensor, image_tensor_cd



def eval_model(args):
    # Model
    print(args.subclausal)
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    
    # model.config.tokenizer_padding_side = tokenizer.padding_side = "left"
    
    dataset = CustomDataset(args.question_file, args.image_folder, tokenizer, image_processor, model.config, args.noise_step, args.max_instances, args.subclausal, args.conv_mode)
    data_loader = DataLoader(dataset, batch_size=args.batch, num_workers=4, shuffle=False, collate_fn=DataCollatorForVisualTextGeneration(tokenizer=tokenizer))
    
    os.makedirs(args.output_dir) if not os.path.exists(args.output_dir) else None
    
    scores=[]
    for (indices, input_ids_list, image_tensor, image_tensor_cd) in tqdm(data_loader, desc="Evaluating:  "):
        
        batch_scores = []
        for input_ids in input_ids_list:
            score=[]
            input_ids = input_ids.to(device='cuda', non_blocking=True)
            stop_str = conv_templates[args.conv_mode].sep if conv_templates[args.conv_mode].sep_style != SeparatorStyle.TWO else conv_templates[args.conv_mode].sep2
            
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                    images_cd=(image_tensor_cd.to(dtype=torch.float16, device='cuda', non_blocking=True) if image_tensor_cd is not None else None),
                    
                    cd_alpha = args.cd_alpha,
                    cd_beta = args.cd_beta,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True)

            
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            
            for output in outputs:
            
                output = output.strip()
                if output.endswith(stop_str):
                    output = output[:-len(stop_str)]
                output = output.strip()
                print(f'{output}\n')
                if 'yes' in output.lower():
                    score.append(1)
                elif 'no' in output.lower():
                    score.append(0)
                else:
                    score.append(0)
                    print(f"There are not \"Yes\" or \"No\" in answer. \n The answer is: {output}\n")
            batch_scores.append(score)
        batch_scores_flip=[list(item) for item in zip(*batch_scores)]
        scores.append(batch_scores_flip)
    
    all_scores = np.concatenate(scores, axis=0)
    result_records = dataset.evaluate_vllm_scores(all_scores)
    
    for record in result_records:
        record.update({"Model": args.model_path.split('/')[1], "Dataset": args.dataset, "Seed": args.seed})
    if args.extra_info is None:
        output_file = os.path.join(args.output_dir, f"{args.dataset}_{args.model_path.split('/')[1]}_seed-{args.seed}.csv")
    else:
        output_file = os.path.join(args.output_dir, f"{args.dataset}_{args.model_path.split('/')[1]}_seed-{args.seed}_{args.extra_info}.csv")
    df = pd.DataFrame(result_records)
    
    print(f"Saving results to {output_file}")
    if os.path.exists(output_file):
        all_df = pd.read_csv(output_file, index_col=0)
        all_df = pd.concat([all_df, df])
        all_df.to_csv(output_file)

    else:
        df.to_csv(output_file)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="/home/Visual-Negation-Reasoning/data/prerelease_bow/images")
    parser.add_argument("--question-file", type=str, default="/home/Visual-Negation-Reasoning/data/prerelease_bow/visual_genome_attribution.json")
    
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--dataset", default="Negation_Logic", type=str)
    parser.add_argument("--output_dir", default="llava_eval/snare/test/", type=str)
    parser.add_argument("--extra_info", default=None, type=str)
    parser.add_argument("--cot_type", type=str, default=None)
    parser.add_argument("--subclausal", action='store_true', default=False)
    parser.add_argument("--max_instances", type=int, default=16)

    parser.add_argument("--noise_step", type=int, default=500)
    
    parser.add_argument("--cd_alpha", type=float, default=1)
    parser.add_argument("--cd_beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()
    set_seed(args.seed)
    eval_model(args)