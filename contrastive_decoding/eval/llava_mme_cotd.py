import argparse
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
import re
from transformers import set_seed

def get_next_token_logit(model, tokenizer, input_ids, image_tensor, image_sizes):
    
    with torch.inference_mode():
        gen_out = model.generate(input_ids,
                            images=image_tensor,
                            image_sizes=image_sizes,
                            output_scores=True, return_dict_in_generate=True, max_new_tokens=1)
    return gen_out.scores[-1]


def generate_branching_responses(model, tokenizer, input_ids, image_tensors, image_sizes, prompt, k, max_new_tokens=80):
    logit = get_next_token_logit(model, tokenizer, input_ids, image_tensors, image_sizes)
    k_token = logit[0].argsort()[-k:]
    k_prob = torch.nn.functional.softmax(logit[0][logit[0].argsort()[-k:]], dim=0)
    k_response = []
    response_probs = []
    for init_token, init_prob in zip(k_token, k_prob):
        init_text = tokenizer.decode(init_token, skip_special_tokens=True)
        new_query = prompt + init_text
        candidate_inputs = tokenizer_image_token(new_query, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        candidate_inputs = torch.stack((candidate_inputs,), dim=0).to(device='cuda', non_blocking=True)
        gen_out = model.generate(candidate_inputs, 
                                images=image_tensors,
                                image_sizes=image_sizes,
                                output_scores=True, return_dict_in_generate=True, max_new_tokens=max_new_tokens)
        
        logits = gen_out.scores
        num_output = len(logits)
        output_ids = gen_out.sequences[0][-num_output:]
        response = tokenizer.decode(torch.cat([init_token.unsqueeze(0), output_ids]), skip_special_tokens=True).strip()
        
        path_probs = [round(init_prob.item(), 4)]
        for score in logits:
            probabilities = torch.softmax(score, dim=-1)
            topk_values, topk_indicies = torch.topk(probabilities, 2)
            prob_diff = topk_values[:, 0] - topk_values[:, 1]
            # decode_seq = tokenizer.decode(topk_indices[:, 0], skip_special_tokens=True)
            # response.append(decode_seq)
            path_probs.append(round(prob_diff.item(), 4))
            # print(f"{decode_seq}  {round(prob_diff.item(), 4)}")
        # print('----'*5)
        k_response.append(response)
        response_probs.append(sum(path_probs) / len(path_probs))
    return k_response, response_probs



class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config


    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        
        return input_ids, image_tensor, image.size, prompt

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes, prompt = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    
    return input_ids, image_tensors, image_sizes, list(prompt)

# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader

def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    
    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config)
    
    for (input_ids, image_tensors, image_sizes, prompt), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["question_id"]
        cur_prompt = line["text"]
        gt = line["GT"]
        
        input_ids = input_ids.to(device='cuda', non_blocking=True)
        image_tensors = image_tensors.to(dtype=torch.float16, device='cuda', non_blocking=True)
        
        k_response, response_probs = generate_branching_responses(model, tokenizer, input_ids, image_tensors, image_sizes, prompt[0], args.num_branches, args.max_new_tokens)
        
        pos_score = 0.0
        neg_score = 0.0
        for i ,response, prob in zip(range(len(k_response)), k_response, response_probs):
            
            if 'yes' in response.lower().strip():
                pos_score += prob
            elif re.search(r'\bno\b', response, re.IGNORECASE):
                neg_score += prob
        
        if pos_score > neg_score:
            outputs = "Yes"
        else:
            outputs = "No"
        
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "GT": gt,
                                   "metadata": {}}) + "\n")
        
        # ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="/home/dataset/MME_Benchmark_release_version")
    parser.add_argument("--question-file", type=str, default="llava_eval/MME/llava_mme_gt.jsonl")
    parser.add_argument("--answers-file", type=str, default="llava_eval/MME/answers/test.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    
    parser.add_argument("--max_new_tokens", type=int, default=128)

    parser.add_argument("--num_branches", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    eval_model(args)