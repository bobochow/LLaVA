import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

from tqdm import tqdm
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math
import re
from typing import List, Tuple
from transformers import set_seed
import torch

def get_next_token_logit(model, tokenizer, input_ids, image_tensor, image_sizes):
    
    with torch.inference_mode():
        gen_out = model.generate(input_ids,
                            images=image_tensor,
                            image_sizes=image_sizes,
                            output_scores=True, return_dict_in_generate=True, max_new_tokens=1, pad_token_id=tokenizer.eos_token_id)
    return gen_out.scores[-1]

def get_token_path_prob(gen_out, num_append:int = 0):
    logits = gen_out.scores
    num_output = len(logits)
    output_ids = gen_out.sequences[0][-num_output-num_append:]
    # output = tokenizer.decode(output_ids, skip_special_tokens=True)
    path_prob = torch.stack([score[0].max() for score in logits])
    path_prob = torch.nn.functional.softmax(path_prob, dim=0)
    # path_logprob = torch.log(path_prob)
    return output_ids, path_prob

def get_path_prob(gen_out, init_token_prob=None):
    if init_token_prob is None:
        token_ids, probs = get_token_path_prob(gen_out, num_append=0)
    else:
        token_ids, probs = get_token_path_prob(gen_out)
        probs = torch.concat([init_token_prob.unsqueeze(-1), probs])
    current_n_words = 0
    current_prob = 0
    word_probs = []
    ids = []
    current_n_tokens = 0
    word_prob = 0
    current_n_words = 0
    for token_id, prob in zip(token_ids, probs):
        ids.append(token_id)
        decode_seq = tokenizer.decode(ids, skip_special_tokens=True)
        # print('Decode Sequence: ', decode_seq)
        words = re.split(r' |\n|\.\|:', decode_seq)
        # print('Splitted Words: ')
        # print(words)
        word = words[-1]
        if len(words) == current_n_words:
            word_prob += prob
            current_n_tokens += 1
            # more than one tokens correspond to the same word, word gets updated
            word_probs[-1] = (word, word_prob / current_n_tokens) # replace the previous word in the word prob list
        elif len(words) > current_n_words: # A old word is determined
            word_prob = prob
            current_n_tokens = 1
            word_probs.append((word, word_prob / current_n_tokens))
            current_n_words += 1
    return word_probs

def get_follow_up_output(model, tokenizer, follow_up_template, gen_out, image_tensors, image_sizes,max_new_tokens=40):
    construct_input = lambda new_ids: {'input_ids': new_ids, "attention_mask":torch.ones_like(new_ids)}
    output_ids = gen_out.sequences
    
    follow_up_ids = tokenizer(follow_up_template, return_tensors="pt")['input_ids']
    
    new_ids = torch.concat([output_ids, follow_up_ids.to(device='cuda', non_blocking=True)], axis=1)
    # inputs = construct_input(new_ids)
    gen_out = model.generate(new_ids,
                            images=image_tensors,
                            image_sizes=image_sizes,
                            output_scores=True, return_dict_in_generate=True, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)
    return gen_out

def get_k_path_prob_follow_up(model, tokenizer, input_ids, image_tensors, image_sizes, prompt, k, max_new_tokens=80, 
                                follow_up_template="\nUSER: Summarize the answer as yes, no, or uncertain. So the answer is: \nASSISTANT:"):
    logit = get_next_token_logit(model, tokenizer, input_ids, image_tensors, image_sizes)
    k_token = logit[0].argsort()[-k:]
    k_prob = torch.nn.functional.softmax(logit[0][logit[0].argsort()[-k:]], dim=0)
    k_response = []
    for token,prob in zip(k_token,k_prob):
        # token =token.unsqueeze(0).unsqueeze(0)
        new_query = prompt + tokenizer.decode(token, skip_special_tokens=True)
        candidate_inputs = tokenizer_image_token(new_query, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        # candidate_inputs = tokenizer(new_query, return_tensors="pt")
        # candidate_inputs = torch.cat([input_ids, token], dim=1)
        # attention_mask = torch.ones_like(candidate_inputs)
        candidate_inputs = torch.stack((candidate_inputs,), dim=0).to(device='cuda', non_blocking=True)
        gen_out = model.generate(candidate_inputs, 
                                images=image_tensors,
                                image_sizes=image_sizes,
                                output_scores=True, return_dict_in_generate=True, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)
        
        output_ids=gen_out.sequences[0]
        final = torch.cat([token.unsqueeze(0), output_ids], dim=0)
        result = tokenizer.decode(gen_out.sequences[0], skip_special_tokens=True)
        
        # follow_up_out = get_follow_up_output(model, tokenizer, follow_up_template, gen_out, image_tensors, image_sizes)
        # path_probs = get_path_prob(follow_up_out, k_prob)
        path_probs = get_path_prob(gen_out, None)
        print(f"{tokenizer.decode(token, skip_special_tokens=True)}  {prob.item()}")
        for p in path_probs:
            print(f"{p[0]}  {p[1]}")
        # print(path_probs)
        print('----'*5)
        k_response.append(path_probs)
    return k_response

def generate_branching_responses(model, tokenizer, input_ids, image_tensors, image_sizes, prompt, k, max_new_tokens=80, 
                                follow_up_template="\nUSER: Summarize the answer as yes, no, or uncertain. So the answer is: \nASSISTANT:"):
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
        
        # print(f"{init_text}  {round(init_prob.item(), 5)}")
        # response = [init_text]
        path_probs = [round(init_prob.item(), 4)]
        for score in logits:
            probabilities = torch.softmax(score, dim=-1)
            topk_values, topk_indices = torch.topk(probabilities, 2)
            prob_diff = topk_values[:, 0] - topk_values[:, 1]
            # decode_seq = tokenizer.decode(topk_indices[:, 0], skip_special_tokens=True)
            # response.append(decode_seq)
            path_probs.append(round(prob_diff.item(), 4))
            # print(f"{decode_seq}  {round(prob_diff.item(), 4)}")
        # print('----'*5)
        k_response.append(response)
        response_probs.append(sum(path_probs) / len(path_probs))
    return k_response, response_probs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    args = parser.parse_args()
    
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    qs='Is a c++ code shown in the picture?'
    raw_image = Image.open('../dataset/MME_Benchmark_release_version/code_reasoning/0020.png')
    #+ ' Answer the question using a single word or phrase.\n'
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs + ' Answer the question using a single word or phrase.\n'
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs + ' Answer the question using a single word or phrase.\n'
    
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    image_tensor = process_images([raw_image], image_processor, model.config)[0]
    image_size = [raw_image.size]
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
    
    input_ids = torch.stack((input_ids,), dim=0)
    image_tensors = torch.stack((image_tensor,), dim=0)
    
    input_ids = input_ids.to(device='cuda', non_blocking=True)
    image_tensors=image_tensors.to(dtype=torch.float16, device='cuda', non_blocking=True)
    # k_response = get_k_path_prob_follow_up(model, tokenizer, input_ids, image_tensors, image_size, prompt,5)
    k_response, response_probs = generate_branching_responses(model, tokenizer, input_ids, image_tensors, image_size, prompt,5)
    
    # print(f'{k_response}\n')
    # print(f'\n{response_probs}\n')
    
    for k , response in enumerate(k_response):
        print(f'\nResponse k={k}:\n\n'+response)
        
        print('\nScore:', response_probs[k])
        print('----'*5)
    