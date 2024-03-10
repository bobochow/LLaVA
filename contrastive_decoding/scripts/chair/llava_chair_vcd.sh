export CUDA_VISIBLE_DEVICES=1

seed=${1:-55}

model_name=llava-v1.5-7b
# model_name=llava-v1.5-13b
# model_name=llava-v1.6-mistral-7b
model_path=liuhaotian/${model_name}

temperature=1

max_new_tokens=512

max_instances=500

cd_alpha=${5:-1}
cd_beta=${6:-0.1}
noise_step=${7:-500}

experiment=CHAIR-${model_name}-vcd-t${temperature}-a${cd_alpha}-b${cd_beta}-seed${seed}

answers_file=llava_eval/CHAIR/answers/${experiment}.jsonl

echo "MME Experiment: $experiment"

python contrastive_decoding/eval/llava_chair.py \
    --model-path ${model_path} \
    --answers-file ${answers_file} \
    --seed ${seed} \
    --temperature ${temperature} \
    --max_new_tokens ${max_new_tokens} \
    --max_instances ${max_instances} \
    --cd_alpha $cd_alpha \
    --cd_beta $cd_beta \
    --noise_step $noise_step \

results_file=llava_eval/CHAIR/results/${experiment}.jsonl

python contrastive_decoding/eval/chair.py \
    --cap_file $answers_file \
    --coco_path data/CHAIR/annotations/ \
    --save_path $results_file