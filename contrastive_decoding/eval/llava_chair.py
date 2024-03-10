import argparse
import os
import torch
from tqdm import tqdm

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from transformers import set_seed
import random
import json

def eval_model(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    img_files = os.listdir(args.data_path)
    random.shuffle(img_files)
    
    with open(os.path.expanduser(args.question_file), 'r') as f:
        lines = f.readlines()
    coco_anns = json.loads(lines[0])
    
    img_dict = {}

    categories = coco_anns["categories"]
    category_names = [c["name"] for c in categories]
    category_dict = {int(c["id"]): c["name"] for c in categories}

    for img_info in coco_anns["images"]:
        img_dict[img_info["id"]] = {"name": img_info["file_name"], "anns": []}

    for ann_info in coco_anns["annotations"]:
        img_dict[ann_info["image_id"]]["anns"].append(
            category_dict[ann_info["category_id"]]
        )
    answers_file = os.path.expanduser(args.answers_file)
    ans_file = open(answers_file, "w")
    
    for img_id in tqdm(range(args.max_instances)):
        # if img_id == 500:
        #     break
        img_file = img_files[img_id]
        img_id = int(img_file.split(".jpg")[0][-6:])
        img_info = img_dict[img_id]
        assert img_info["name"] == img_file
        img_anns = set(img_info["anns"])
        img_save = {}
        img_save["image_id"] = img_id

        image_path = args.data_path + img_file
        raw_image = Image.open(image_path).convert("RGB")
        image_tensor = process_images([raw_image], image_processor, model.config)[0].unsqueeze(0)
        
        stop_str = conv_templates[args.conv_mode].sep if conv_templates[args.conv_mode].sep_style != SeparatorStyle.TWO else conv_templates[args.conv_mode].sep2
        
        qs = DEFAULT_IMAGE_TOKEN + '\n' + "Please describe this image in detail."
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
        input_ids = input_ids.to(device='cuda', non_blocking=True)
        
        
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
            )
        input_token_len = input_ids.shape[1]
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        
        img_save["caption"] = outputs

        # dump metric file
        # with open(os.path.join(base_dir, 'greedy_ours-s_{}-t_{}-num_can_{}-p_{}.jsonl'.format(args.scale_factor, args.threshold, args.num_attn_candidates, args.penalty_weights)), "a") as f:
        #     json.dump(img_save, f)
        #     f.write('\n')
        ans_file.write(
            json.dumps(img_save) + "\n"
        )
    
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CHAIR evaluation on LVLMs.")
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--data_path", type=str, default="../dataset/coco/val2014/", help="data path")
    parser.add_argument("--question-file", type=str, default="data/CHAIR/annotations/instances_val2014.json")
    parser.add_argument("--answers-file", type=str, default="llava_eval/MME/answers/test.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=55)
    parser.add_argument("--max_instances", type=int, default=10)
    args = parser.parse_args()
    set_seed(args.seed)
    eval_model(args)