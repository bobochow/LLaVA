export CUDA_VISIBLE_DEVICES=2

seed=${1:-55}

model_name=llava-v1.5-7b
# model_name=llava-v1.5-13b
# model_name=llava-v1.6-mistral-7b
model_path=liuhaotian/${model_name}

temperature=0

max_new_tokens=512

max_instances=500

experiment=CHAIR-${model_name}-greedy-t${temperature}-seed${seed}

answers_file=llava_eval/CHAIR/answers/${experiment}.jsonl

echo "CHAIR Experiment: $experiment"

python contrastive_decoding/eval/llava_chair.py \
    --model-path ${model_path} \
    --answers-file ${answers_file} \
    --seed ${seed} \
    --temperature ${temperature} \
    --max_new_tokens ${max_new_tokens} \
    --max_instances ${max_instances}

results_file=llava_eval/CHAIR/results/${experiment}.jsonl

python contrastive_decoding/eval/chair.py \
    --cap_file $answers_file \
    --coco_path data/CHAIR/annotations/ \
    --save_path $results_file