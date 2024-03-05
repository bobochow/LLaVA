#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

seed=${1:-55}
dataset_name=${2:-"mme"}
model_name=llava-v1.5-7b
model_path=liuhaotian/${model_name}

image_folder=/home/dataset/MME_Benchmark_release_version

temperature=1

neg=false

if [[ $neg == false ]]; then
    question_file=llava_eval/MME/llava_mme_gt.jsonl
    experiment=${model_name}-sample-t${temperature}-seed${seed}
else
    question_file=llava_eval/MME/llava_mme_neg.jsonl
    experiment=NEG-${model_name}-sample-t${temperature}-seed${seed}
fi

answers_file=llava_eval/MME/answers/${experiment}.jsonl
echo "Running experiment: $experiment"

python -m llava.eval.model_vqa_loader \
    --model-path ${model_path} \
    --question-file ${question_file} \
    --image-folder ${image_folder} \
    --answers-file ${answers_file} \
    --temperature ${temperature} \
    --conv-mode vicuna_v1 \
    --seed ${seed}

cd llava_eval/MME

python convert_answer_to_mme.py --experiment $experiment

cd eval_tool

python calculation.py --results_dir answers/$experiment
