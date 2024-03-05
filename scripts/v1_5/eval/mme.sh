#!/bin/bash
export CUDA_VISIBLE_DEVICES=2

seed=55
model_path=liuhaotian/llava-v1.5-13b

image_folder=/home/dataset/MME_Benchmark_release_version

temperature=1

neg=false

if [[ $neg == false ]]; then
    question_file=llava_eval/MME/llava_mme_gt.jsonl
    experiment=llava-v1.5-13b-sample-t${temperature}-seed${seed}_v1.6
else
    question_file=llava_eval/MME/llava_mme_neg.jsonl
    experiment=NEG-llava-v1.5-13b-sample-t${temperature}-seed${seed}_v1.6
fi

answers_file=llava_eval/MME/answers/${experiment}.jsonl

python -m llava.eval.model_vqa_loader \
    --model-path $model_path \
    --question-file $question_file \
    --image-folder $image_folder \
    --answers-file $answers_file \
    --temperature $temperature \
    --seed ${seed}


cd llava_eval/MME

python convert_answer_to_mme.py --experiment $experiment

cd eval_tool

echo "MME Experiment: $experiment"

python calculation.py --results_dir answers/${experiment}
